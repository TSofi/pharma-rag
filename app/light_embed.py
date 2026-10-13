"""Memory-light question embedder for the multilingual model (paraphrase-multilingual-MiniLM-L12-v2).

The Hugging Face `tokenizers` library needs ~280 MB of RAM for this model's 250k-piece Unigram vocabulary,
more than the int8 model itself, so on the 512 MB server the same tokenization is done here in plain Python
(~60 MB): NFKC normalisation, whitespace split with the "▁" prefix, Viterbi over the piece scores
(SentencePiece Unigram), <s> ... </s>. Then ONNX Runtime, mean pooling and L2 normalisation, exactly what
fastembed / sentence-transformers do for this model. Checked against fastembed by scripts/check_int8.py.
"""
import json
import os
import unicodedata

import numpy as np

BOS, EOS, UNK = 0, 2, 3
MAX_TOKENS = 128


class UnigramTokenizer:
    def __init__(self, tokenizer_json: str):
        with open(tokenizer_json, encoding="utf-8") as f:
            vocab = json.load(f)["model"]["vocab"]
        # 250k Python strings in a dict cost ~100 MB, so pieces are kept only as sorted 64-bit hashes
        # (numpy, ~6 MB in total). The UTF-8 bytes are hashed, not the str: CPython hashes a str's internal
        # buffer, so "na" (1 byte/char) and "慮" (U+616E, 2 bytes/char) would collide.
        h = np.array([hash(p.encode()) for p, _ in vocab], dtype=np.int64)
        order = np.argsort(h, kind="stable")
        self.hashes, self.piece_ids = h[order], order.astype(np.int32)
        self.scores = np.array([sc for _, sc in vocab], dtype=np.float64)
        self.max_len = max(len(p) for p, _ in vocab)
        del vocab
        self.unk_score = float(self.scores.min()) - 10.0

    def _lookup(self, piece: str) -> int | None:
        h = hash(piece.encode())
        i = int(np.searchsorted(self.hashes, h, side="right")) - 1  # a few pieces appear twice: last id wins
        return int(self.piece_ids[i]) if i >= 0 and self.hashes[i] == h else None

    def _word(self, w: str) -> list[int]:
        n = len(w)
        best = [(-1e18, -1, UNK)] * (n + 1)  # (score, start, id)
        best[0] = (0.0, -1, UNK)
        for i in range(n):
            if best[i][0] <= -1e17:
                continue
            base, found_single = best[i][0], False
            for j in range(i + 1, min(n, i + self.max_len) + 1):
                pid = self._lookup(w[i:j])
                if pid is not None:
                    if j == i + 1:
                        found_single = True
                    sc = base + self.scores[pid]
                    if sc > best[j][0]:
                        best[j] = (sc, i, pid)
            if not found_single and base + self.unk_score > best[i + 1][0]:
                best[i + 1] = (base + self.unk_score, i, UNK)
        ids, j = [], n
        while j > 0:
            _, i, pid = best[j]
            ids.append(pid)
            j = i
        ids.reverse()
        out = []  # consecutive unknown characters become one <unk>, like SentencePiece
        for t in ids:
            if not (t == UNK and out and out[-1] == UNK):
                out.append(t)
        return out

    def encode(self, text: str) -> list[int]:
        text = unicodedata.normalize("NFKC", text)
        ids = [BOS]
        for w in text.split():
            ids += self._word("▁" + w)
        return ids[:MAX_TOKENS - 1] + [EOS]


class LightEmbedder:
    def __init__(self, model_dir: str, threads: int | None = None):
        import onnxruntime as ort
        self.tok = UnigramTokenizer(os.path.join(model_dir, "tokenizer.json"))
        o = ort.SessionOptions()
        o.enable_cpu_mem_arena = False
        if threads:
            o.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(os.path.join(model_dir, "model_optimized.onnx"), o,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def embed(self, text: str) -> list[float]:
        ids = np.array([self.tok.encode(text)], dtype=np.int64)
        mask = np.ones_like(ids)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        h = self.sess.run(None, feeds)[0][0]  # (tokens, 384)
        v = h.mean(axis=0)
        return (v / (np.linalg.norm(v) or 1)).tolist()
