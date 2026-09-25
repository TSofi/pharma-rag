const API = (window.API_BASE || "").replace(/\/$/, "");
const t = (k, v) => I18N.t(k, v);
const $ = (s) => document.querySelector(s);

const form = $("#askForm"), q = $("#q"), btn = $("#askBtn");
const result = $("#result"), answerEl = $("#answer"), sourcesEl = $("#sources");

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// Minimal markdown: bullets, **bold**, and [n] citations -> clickable badges.
function renderAnswer(text) {
  const inline = (s) =>
    esc(s)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*\w])\*([^*\n]+?)\*(?!\w)/g, "$1<em>$2</em>")
      .replace(/\[(\d+)\]/g, '<button type="button" class="cite" data-src="$1" aria-label="Source $1">$1</button>');
  const html = [];
  let list = false;
  for (const raw of text.split(/\n+/)) {
    const line = raw.trim();
    if (!line) continue;
    const m = line.match(/^[-*•]\s+(.*)/);
    if (m) { if (!list) { html.push("<ul>"); list = true; } html.push(`<li>${inline(m[1])}</li>`); }
    else { if (list) { html.push("</ul>"); list = false; } html.push(`<p>${inline(line)}</p>`); }
  }
  if (list) html.push("</ul>");
  return html.join("");
}

function fmtDate(yyyymmdd) {
  if (!yyyymmdd || yyyymmdd.length !== 8) return "";
  const d = new Date(`${yyyymmdd.slice(0, 4)}-${yyyymmdd.slice(4, 6)}-${yyyymmdd.slice(6)}`);
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short" });
}

function renderSources(sources) {
  sourcesEl.innerHTML = sources.map((s) => `
    <li class="card src ${s.cited ? "" : "uncited"}" id="src-${s.id}">
      <div class="src-top">
        <span class="cite" aria-hidden="true">${s.id}</span>
        <span class="src-drug">${esc(s.drug)}</span>
        ${s.brands?.length ? `<span class="muted">${esc(s.brands.slice(0, 2).join(", "))}</span>` : ""}
        <span class="src-sec">${esc(s.section)}</span>
        <span class="src-score" title="${esc(t("scoreTitle"))}">
          <span class="bar"><i style="width:${Math.max(0, Math.min(1, s.score)) * 100}%"></i></span>${s.score.toFixed(2)}
        </span>
      </div>
      <p class="src-text">${esc(s.text)}</p>
      <div class="src-foot">
        <button type="button" class="linkish" data-expand>${t("showFull")}</button>
        <a href="${esc(s.url)}" target="_blank" rel="noopener">${t("openLabel")}</a>
      </div>
    </li>`).join("");
}

function flashSource(n) {
  const el = document.getElementById(`src-${n}`);
  if (!el) return;
  el.classList.remove("uncited");
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 1400);
}

async function ask(question) {
  $("#hero").classList.add("compact");
  $("#chips").hidden = true;
  autosize();
  result.hidden = false;
  btn.disabled = true;
  $("#detected").innerHTML = "";
  $("#meta").textContent = t("searching");
  $("#notice").hidden = true;
  document.querySelector(".sources-head").hidden = true;
  answerEl.className = "answer-body";
  const waiting = Waiting.start(answerEl, API);
  sourcesEl.innerHTML = "";
  const t0 = performance.now();
  try {
    const r = await fetch(`${API}/api/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, market: $("#market").value, answer_language: $("#answerLang").value }),
    });
    const data = await r.json();
    const score = waiting.stop();
    if (!r.ok) throw new Error(data.detail || r.statusText);

    answerEl.innerHTML = renderAnswer(data.answer);
    answerEl.classList.toggle("notfound", !data.found && !data.playful);
    answerEl.classList.toggle("playful", !!data.playful);
    $("#detected").innerHTML = (data.detected_drugs || []).map((d) => `<span>${esc(d)}</span>`).join("");
    renderSources(data.sources);
    const cited = data.sources.filter((s) => s.cited).length;
    const dates = [...new Set(data.sources.filter((s) => s.cited).map((s) => fmtDate(s.effective_time)).filter(Boolean))];
    $("#meta").textContent = (data.found
      ? t("metaFound", { cited, total: data.sources.length, secs: ((performance.now() - t0) / 1000).toFixed(1) }) +
        (dates.length ? " · " + t("metaVersions", { dates: dates.join(", ") }) : "") + (data.model ? ` · ${data.model}` : "")
      : data.playful ? t("metaPlayful") : t("metaNone")) + (score ? " · " + t("metaScore", { n: score }) : "");

    // Transparency: show how the question was interpreted (typo fixes, translation).
    const notes = [];
    for (const c of data.corrections || []) notes.push(t("interpreted", { typed: esc(c.typed), matched: esc(c.matched) }));
    if (data.search_query) notes.push(t("searchedFor", { q: esc(data.search_query) }));
    $("#notice").innerHTML = notes.join(" ");
    $("#notice").hidden = !notes.length;
    document.querySelector(".sources-head").hidden = !data.sources.length;
  } catch (e) {
    waiting.stop();
    answerEl.className = "answer-body notfound";
    const busy = /503|UNAVAILABLE|overload|high demand|429|RESOURCE_EXHAUSTED/i.test(e.message);
    answerEl.innerHTML = busy
      ? `<p>${t("busy")}</p>
         <button type="button" class="retry">${t("retry")}</button>
         <details class="err"><summary>${t("details")}</summary><code>${esc(e.message)}</code></details>`
      : `<p>${t("wrong")}</p><button type="button" class="retry">${t("retry")}</button>
         <details class="err" open><summary>${t("details")}</summary><code>${esc(e.message)}</code></details>`;
    answerEl.querySelector(".retry").onclick = () => ask(question);
    document.querySelector(".sources-head").hidden = true;
    $("#meta").textContent = e instanceof TypeError ? t("noApi") : busy ? "" : t("apiErr");
  } finally {
    btn.disabled = false;
  }
}

form.addEventListener("submit", (e) => { e.preventDefault(); const v = q.value.trim(); if (v) ask(v); });
q.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); } });
const autosize = () => { q.style.height = "auto"; q.style.height = q.scrollHeight + "px"; };
q.addEventListener("input", autosize);

$("#chips").addEventListener("click", (e) => {
  if (e.target.tagName !== "BUTTON") return;
  q.value = e.target.textContent;
  form.requestSubmit();
});

document.addEventListener("click", (e) => {
  const c = e.target.closest(".answer-body .cite");
  if (c) flashSource(c.dataset.src);
  const x = e.target.closest("[data-expand]");
  if (x) {
    const txt = x.closest(".src").querySelector(".src-text");
    txt.classList.toggle("open");
    x.textContent = txt.classList.contains("open") ? t("collapse") : t("showFull");
  }
});

// The "Answer in" choice also sets the interface language ("Same as my question" = English UI).
const applyLang = () => I18N.apply($("#answerLang").value === "auto" ? "en" : $("#answerLang").value);
$("#answerLang").addEventListener("change", applyLang);

for (const id of ["answerLang"]) {
  const el = $("#" + id);
  try { const v = localStorage.getItem(id); if (v && [...el.options].some((o) => o.value === v)) el.value = v; } catch { /* no storage */ }
  el.addEventListener("change", () => { try { localStorage.setItem(id, el.value); } catch { /* no storage */ } });
}
applyLang();

$("#showAll").addEventListener("change", (e) => sourcesEl.classList.toggle("hide-uncited", !e.target.checked));
sourcesEl.classList.add("hide-uncited");

// ----- market selector: the main one and the one in the library drawer stay in sync -----
for (const [from, to] of [["#market", "#libMarket"], ["#libMarket", "#market"]]) {
  $(from).addEventListener("change", (e) => { $(to).value = e.target.value; });
}

// ----- drug library drawer -----
const drawer = $("#library");
let drugs = [];
function renderLib(filter = "") {
  const f = filter.toLowerCase();
  $("#libList").innerHTML = drugs
    .filter((d) => !f || d.drug.includes(f) || d.brands.some((b) => b.toLowerCase().includes(f)))
    .map((d) => `<li><button type="button" data-drug="${esc(d.drug)}"><b>${esc(d.drug)}</b><small>${esc(d.brands.join(", ") || "—")}</small></button></li>`)
    .join("");
}
function toggleLib(open) { drawer.hidden = !open; $("#libraryBtn").setAttribute("aria-expanded", open); if (open) $("#libFilter").focus(); }
$("#libraryBtn").onclick = () => toggleLib(true);
$("#closeLib").onclick = () => toggleLib(false);
drawer.addEventListener("click", (e) => { if (e.target === drawer) toggleLib(false); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") toggleLib(false); });
$("#libFilter").addEventListener("input", (e) => renderLib(e.target.value));
$("#libList").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  toggleLib(false);
  q.value = t("libAsk", { drug: b.dataset.drug });
  autosize();
  q.focus();
});

// Wake the backend up as soon as the page opens: a free Render instance sleeps after 15 min idle
// and needs ~50 s to start, so we start that clock while the visitor is still reading/typing.
fetch(`${API}/api/health`).catch(() => {});

fetch(`${API}/api/drugs`).then((r) => r.json()).then((d) => {
  drugs = d; $("#drugCount").textContent = d.length; renderLib();
}).catch(() => { $("#drugCount").textContent = "offline"; });
