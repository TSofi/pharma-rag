// Loading screen shown while the backend retrieves + generates an answer.
// Two modes: "game" (a pill jumping over germs, default) and "fact" (short curated drug facts).
// Usage: const w = Waiting.start(container, apiBase); ... const score = w.stop();
window.Waiting = (() => {
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const getMode = () => { try { return localStorage.getItem("waitMode") || "game"; } catch { return "game"; } };
  const setMode = (m) => { try { localStorage.setItem("waitMode", m); } catch { /* storage unavailable */ } };
  const seen = [];  // drugs whose facts were already shown, so facts don't repeat

  function start(container, API = "") {
    container.innerHTML = `
      <div class="wait">
        <div class="wait-top">
          <div class="wait-tabs" role="tablist">
            <button type="button" data-mode="game" role="tab">${I18N.t("tabGame")}</button>
            <button type="button" data-mode="fact" role="tab">${I18N.t("tabFact")}</button>
          </div>
          <span class="wait-timer" aria-live="off">0s</span>
        </div>
        <div class="progress"><i></i></div>
        <figure class="fact" hidden><span class="fact-emoji" aria-hidden="true">💡</span><blockquote></blockquote><figcaption></figcaption></figure>
        <div class="game">
          <canvas height="150" aria-label="Mini game: press Space or tap to jump over obstacles"></canvas>
          <p class="game-hint">${I18N.t("gameHint", { score: '<b class="g-score">0</b>', best: `<b class="g-best">${window.__pillBest || 0}</b>` })}</p>
        </div>
      </div>`;

    const $ = (s) => container.querySelector(s);
    const t0 = performance.now();
    let alive = true, factTimer;
    const tick = setInterval(() => { $(".wait-timer").textContent = `${Math.round((performance.now() - t0) / 1000)}s`; }, 500);

    const game = PillGame.create($("canvas"), {
      onScore: (n) => { $(".g-score").textContent = n; },
      onBest: (n) => { $(".g-best").textContent = n; },
    });
    const jump = () => game.jump();

    const onKey = (e) => {
      if (getMode() !== "game") return;
      if (e.code !== "Space" && e.code !== "ArrowUp") return;
      if (document.activeElement?.tagName === "TEXTAREA") return;
      e.preventDefault(); jump();
    };
    document.addEventListener("keydown", onKey);

    // ---- facts ----
    async function nextFact() {
      try {
        const f = await (await fetch(`${API}/api/fact?lang=${I18N.lang}&seen=${encodeURIComponent(seen.slice(-30).join(","))}`)).json();
        if (!alive || !f.text) return;
        seen.push(f.drug);
        const fig = $(".fact");
        fig.classList.remove("in"); void fig.offsetWidth; fig.classList.add("in");
        fig.querySelector("blockquote").textContent = f.text;
        fig.querySelector("figcaption").innerHTML =
          `<b>${esc(f.drug)}</b> · <a href="${esc(f.url)}" target="_blank" rel="noopener">${esc(f.source)} ↗</a>`;
      } catch { /* facts are decorative */ }
    }

    // ---- mode switching ----
    function show(mode) {
      setMode(mode);
      container.querySelectorAll(".wait-tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.mode === mode));
      $(".fact").hidden = mode !== "fact";
      $(".game").hidden = mode !== "game";
      game.pause(); clearInterval(factTimer);
      if (mode === "game") { document.activeElement?.blur?.(); game.start(); }
      else { nextFact(); factTimer = setInterval(nextFact, 9000); }
    }
    container.querySelector(".wait-tabs").addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (b) show(b.dataset.mode);
    });
    show(getMode());

    return {
      stop() {
        alive = false;
        clearInterval(tick); clearInterval(factTimer);
        document.removeEventListener("keydown", onKey);
        const score = game.stop();
        window.__pillBest = game.best;
        return getMode() === "game" ? score : 0;
      },
    };
  }
  return { start };
})();
