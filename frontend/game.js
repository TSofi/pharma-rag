// "Pill runner": the mini-game shown while an answer is loading.
// 12 worlds = player pill × floor × germs. Every new game picks a different world.
// In 4 of them the rules are secretly reversed: you have to HIT the germs (no instructions given).
window.PillGame = (() => {
  // ---------------------------------------------------------------- players
  const pill = {
    capsule(ctx, a = "#5fb8a8", b = "#e9e4d8") {
      ctx.beginPath(); ctx.roundRect(-18, -9, 36, 18, 9); ctx.fillStyle = a; ctx.fill();
      ctx.beginPath(); ctx.roundRect(0, -9, 18, 18, [0, 9, 9, 0]); ctx.fillStyle = b; ctx.fill();
      ctx.fillStyle = "rgba(255,255,255,.35)"; ctx.fillRect(-12, -6, 20, 3);
    },
    round(ctx, fill, score = "rgba(0,0,0,.18)") {
      ctx.beginPath(); ctx.arc(0, 0, 12, 0, Math.PI * 2); ctx.fillStyle = fill; ctx.fill();
      ctx.strokeStyle = "rgba(150,150,150,.55)"; ctx.lineWidth = 1.2; ctx.stroke();  // visible on dark and light bg
      ctx.strokeStyle = score; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(-8, 0); ctx.lineTo(8, 0); ctx.stroke();
      ctx.fillStyle = "rgba(255,255,255,.35)"; ctx.beginPath(); ctx.arc(-4, -5, 3, 0, Math.PI * 2); ctx.fill();
    },
    caplet(ctx) {
      ctx.beginPath(); ctx.roundRect(-20, -8, 40, 16, 8); ctx.fillStyle = "#f4f1ea"; ctx.fill();
      ctx.strokeStyle = "rgba(0,0,0,.15)"; ctx.lineWidth = 1.2; ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, -6); ctx.lineTo(0, 6); ctx.stroke();
    },
    softgel(ctx) {
      ctx.beginPath(); ctx.ellipse(0, 0, 19, 11, 0, 0, Math.PI * 2);
      const gr = ctx.createRadialGradient(-6, -4, 2, 0, 0, 19);
      gr.addColorStop(0, "#fff3b0"); gr.addColorStop(1, "#d9a521"); ctx.fillStyle = gr; ctx.fill();
    },
    ampoule(ctx) {
      ctx.fillStyle = "rgba(170,210,230,.75)"; ctx.strokeStyle = "rgba(120,170,200,.9)"; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.roundRect(-8, -4, 16, 18, 5); ctx.fill(); ctx.stroke();       // body
      ctx.beginPath(); ctx.roundRect(-3, -12, 6, 9, 2); ctx.fill(); ctx.stroke();         // neck
      ctx.beginPath(); ctx.arc(0, -15, 4, 0, Math.PI * 2); ctx.fill(); ctx.stroke();       // head
      ctx.fillStyle = "#e46a6a"; ctx.fillRect(-8, 6, 16, 3);                                 // label stripe
    },
  };

  // ---------------------------------------------------------------- floors
  const floor = {
    dashed(ctx, W, y, off, c) {
      ctx.strokeStyle = c.line; ctx.lineWidth = 2; ctx.setLineDash([6, 8]); ctx.lineDashOffset = off % 14;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); ctx.setLineDash([]);
    },
    cushions(ctx, W, y, off, c) {  // padded-cell pillows
      const w = 38, o = off % w;
      for (let x = -o - w; x < W + w; x += w) {
        ctx.beginPath(); ctx.roundRect(x + 2, y, w - 4, 20, 9); ctx.fillStyle = c.cushion; ctx.fill();
        ctx.fillStyle = c.line; ctx.beginPath(); ctx.arc(x + w / 2, y + 10, 2, 0, Math.PI * 2); ctx.fill();
      }
    },
    tiles(ctx, W, y, off, c) {  // pharmacy floor
      const s = 12, o = off % (s * 2);
      for (let x = -o - s * 2; x < W + s; x += s)
        for (let r = 0; r < 2; r++) {
          ctx.fillStyle = ((Math.round((x + o) / s) + r) % 2) ? c.line : "transparent";
          ctx.fillRect(x, y + r * s, s, s);
        }
    },
    blister(ctx, W, y, off, c) {  // blister pack
      ctx.fillStyle = c.line; ctx.fillRect(0, y + 4, W, 16);
      const s = 26, o = off % s;
      for (let x = -o; x < W + s; x += s) {
        ctx.beginPath(); ctx.ellipse(x, y + 6, 9, 6, 0, Math.PI, 0); ctx.fillStyle = c.cushion; ctx.fill();
      }
    },
    syrup(ctx, W, y, off, c) {  // wavy cough syrup
      ctx.fillStyle = c.syrup; ctx.beginPath(); ctx.moveTo(0, y + 22);
      for (let x = 0; x <= W; x += 6) ctx.lineTo(x, y + 3 + Math.sin((x + off) / 14) * 3);
      ctx.lineTo(W, y + 22); ctx.fill();
    },
    stripes(ctx, W, y, off, c) {  // hospital linoleum
      const s = 22, o = off % s;
      ctx.strokeStyle = c.line; ctx.lineWidth = 3;
      for (let x = -o; x < W + s; x += s) { ctx.beginPath(); ctx.moveTo(x, y + 18); ctx.lineTo(x + 10, y); ctx.stroke(); }
    },
  };

  // ---------------------------------------------------------------- germs
  const germ = {
    spiky(ctx, r, t, col = "#d9825b") {
      ctx.rotate(t / 20); ctx.fillStyle = col;
      ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI * 2); ctx.fill();
      for (let k = 0; k < 8; k++) { ctx.rotate(Math.PI / 4); ctx.fillRect(r - 1, -1.5, 5, 3); }
    },
    blob(ctx, r, t) {  // purple amoeba
      ctx.fillStyle = "#9b7fc4"; ctx.beginPath();
      for (let a = 0; a <= Math.PI * 2 + 0.01; a += Math.PI / 10) {
        const rr = r + Math.sin(a * 3 + t / 8) * 3;
        ctx.lineTo(Math.cos(a) * rr, Math.sin(a) * rr);
      }
      ctx.fill(); ctx.fillStyle = "rgba(255,255,255,.5)"; ctx.beginPath(); ctx.arc(-3, -2, 3, 0, Math.PI * 2); ctx.fill();
    },
    bacillus(ctx, r, t) {  // green rod with flagella
      ctx.rotate(Math.sin(t / 15) * 0.3);
      ctx.strokeStyle = "#7fae6a"; ctx.lineWidth = 1.5;
      for (const s of [-1, 1]) { ctx.beginPath(); ctx.moveTo(s * r * 1.3, 0); ctx.quadraticCurveTo(s * (r * 1.3 + 6), Math.sin(t / 5) * 6, s * (r * 1.3 + 12), 0); ctx.stroke(); }
      ctx.fillStyle = "#8cc27a"; ctx.beginPath(); ctx.roundRect(-r * 1.3, -r * 0.6, r * 2.6, r * 1.2, r * 0.6); ctx.fill();
    },
    cocci(ctx, r) {  // yellow cluster
      ctx.fillStyle = "#e0b64f";
      for (const [x, y] of [[0, 0], [-r * .6, -r * .5], [r * .6, -r * .4], [-r * .5, r * .6], [r * .6, r * .5]]) {
        ctx.beginPath(); ctx.arc(x, y, r * 0.5, 0, Math.PI * 2); ctx.fill();
      }
    },
    corona(ctx, r, t) {  // knobby virus
      ctx.rotate(-t / 30); ctx.fillStyle = "#c96f8a";
      for (let k = 0; k < 10; k++) {
        ctx.rotate(Math.PI / 5);
        ctx.beginPath(); ctx.arc(r + 2, 0, 2.6, 0, Math.PI * 2); ctx.fill(); ctx.fillRect(r - 2, -0.8, 4, 1.6);
      }
      ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI * 2); ctx.fill();
    },
    phage(ctx, r, t) {  // bacteriophage: hexagon head, tail, legs
      ctx.fillStyle = "#6f9fc9"; ctx.strokeStyle = "#6f9fc9"; ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (let k = 0; k < 6; k++) ctx.lineTo(Math.cos(k * Math.PI / 3) * r * .7, -r * .5 + Math.sin(k * Math.PI / 3) * r * .7);
      ctx.fill();
      ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(0, r * .6); ctx.stroke();
      for (const s of [-1, 1]) { ctx.beginPath(); ctx.moveTo(0, r * .6); ctx.lineTo(s * r * .8, r + Math.sin(t / 4) * 2); ctx.stroke(); }
    },
  };

  // ---------------------------------------------------------------- 12 worlds
  const W_ = (name, p, f, gm, reverse = false) => ({ name, p, f, g: gm, reverse });
  const WORLDS = [
    W_("Classic", (c) => pill.capsule(c), "dashed", (c, r, t) => germ.spiky(c, r, t)),
    W_("Padded cell", (c) => pill.round(c, "#f7f5f0"), "cushions", germ.blob),
    W_("Night shift", (c) => pill.round(c, "#26292b", "rgba(255,255,255,.25)"), "stripes", germ.bacillus),
    W_("Pink pill", (c) => pill.round(c, "#f1a7c0"), "blister", germ.cocci),
    W_("Ampoule", pill.ampoule, "tiles", germ.phage),
    W_("Caplet", pill.caplet, "syrup", germ.corona),
    W_("Softgel", pill.softgel, "cushions", (c, r, t) => germ.spiky(c, r, t, "#c46a4a")),
    W_("Orange capsule", (c) => pill.capsule(c, "#ef9a4a", "#f6efe2"), "tiles", germ.bacillus),
    // secretly reversed: hit the germs!
    W_("Red & white", (c) => pill.capsule(c, "#d95c5c", "#f6efe2"), "dashed", (c, r, t) => germ.spiky(c, r, t, "#8fbf6f"), true),
    W_("Glass hero", pill.ampoule, "cushions", germ.blob, true),
    W_("Midnight", (c) => pill.round(c, "#26292b", "rgba(255,255,255,.25)"), "blister", germ.phage, true),
    W_("Bubblegum", (c) => pill.round(c, "#f1a7c0"), "syrup", germ.cocci, true),
  ];

  let lastWorld = -1, best = 0;
  const pickWorld = () => {
    let i; do { i = Math.floor(Math.random() * WORLDS.length); } while (i === lastWorld);
    lastWorld = i; return WORLDS[i];
  };

  function create(canvas, { onScore = () => {}, onBest = () => {} } = {}) {
    const ctx = canvas.getContext("2d");
    let g, raf, running = false, fx = [], last = 0;
    // Speed is in "pixels per 60fps-frame". Every run starts slow and speeds up gradually,
    // and all movement is scaled by real elapsed time, so 120/144 Hz screens aren't faster.
    const START_SPEED = 2.4, MAX_SPEED = 7.5, ACCEL = 0.0016;

    function reset() {
      g = { y: 0, vy: 0, obs: [], t: 0, speed: START_SPEED, score: 0, next: 90, over: 0, escaped: 0, world: pickWorld() };
      fx = [];
    }
    const jump = () => { if (g && g.y === 0 && !g.over) g.vy = 10.5; };
    const end = (msg) => {
      g.over = 50; g.msg = msg; best = Math.max(best, Math.floor(g.score)); onBest(best);
    };

    function frame(now) {
      if (!running) return;
      const dt = last ? Math.min(3, (now - last) / (1000 / 60)) : 1;  // 1.0 == one 60 fps frame
      last = now;
      const css = getComputedStyle(document.documentElement);
      const c = {
        line: css.getPropertyValue("--line").trim(), ink: css.getPropertyValue("--muted").trim(),
        accent: css.getPropertyValue("--accent").trim(),
        cushion: css.getPropertyValue("--accent-soft").trim(), syrup: "rgba(201,111,138,.35)",
      };
      const dpr = window.devicePixelRatio || 1, W = canvas.clientWidth, H = 150, ground = H - 24;
      if (canvas.width !== Math.round(W * dpr)) { canvas.width = Math.round(W * dpr); canvas.height = H * dpr; }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      const world = g.world, px = 48;

      if (g.over) { g.over -= dt; if (g.over <= 0) reset(); }
      else {
        g.t += dt; g.speed = Math.min(MAX_SPEED, g.speed + ACCEL * dt); g.score += 0.1 * dt;
        g.vy -= 0.62 * dt; g.y = Math.max(0, g.y + g.vy * dt); if (g.y === 0) g.vy = 0;
        g.next -= dt;
        if (g.next <= 0) {
          g.obs.push({ x: W + 14, r: 9 + Math.random() * 6, fly: Math.random() < (world.reverse ? 0.35 : 0.18) });
          g.next = 70 + Math.random() * 70 - Math.min(30, g.speed * 3);
        }
        for (const o of g.obs) o.x -= g.speed * dt;
        for (const o of g.obs) if (world.reverse && !o.hit && !o.gone && o.x < px - 26) {
          o.gone = true; g.escaped++;
          if (g.escaped >= 3) end("They got away!");
        }
        g.obs = g.obs.filter((o) => o.x > -30 && !o.hit);
      }

      floor[world.f](ctx, W, ground + 1, g.t * g.speed, c);

      // player
      const py = ground - 14 - g.y;
      ctx.save(); ctx.translate(px, py); ctx.rotate(g.y > 0 ? -0.35 : Math.sin(g.t / 6) * 0.04);
      world.p(ctx); ctx.restore();

      // germs
      for (const o of g.obs) {
        const oy = o.fly ? ground - 58 : ground - o.r;
        ctx.save(); ctx.translate(o.x, oy); world.g(ctx, o.r, g.t); ctx.restore();
        if (!g.over && Math.hypot(o.x - px, oy - py) < o.r + 12) {
          if (world.reverse) {  // pop!
            o.hit = true; g.score += 10;
            for (let k = 0; k < 10; k++) fx.push({ x: o.x, y: oy, vx: (Math.random() - .5) * 5, vy: (Math.random() - .7) * 5, life: 30 });
            fx.push({ x: o.x, y: oy - 18, text: "+10", life: 40 });
          } else end("Ouch! Restarting…");
        }
      }

      // pop particles / "+10"
      for (const p of fx) {
        p.life -= dt;
        if (p.text) { ctx.fillStyle = c.accent; ctx.font = "700 13px Inter, sans-serif"; ctx.textAlign = "center"; ctx.fillText(p.text, p.x, p.y - (40 - p.life) / 2); }
        else { p.x += p.vx * dt; p.y += p.vy * dt; p.vy += 0.2 * dt; ctx.fillStyle = c.accent; ctx.globalAlpha = p.life / 30; ctx.fillRect(p.x, p.y, 3, 3); ctx.globalAlpha = 1; }
      }
      fx = fx.filter((p) => p.life > 0);

      // HUD: world name only -- the reversed rules are a secret
      ctx.fillStyle = c.ink; ctx.font = "500 11px Inter, sans-serif"; ctx.textAlign = "left";
      ctx.fillText(world.name, 10, 16);
      if (g.over) { ctx.font = "600 14px Inter, sans-serif"; ctx.textAlign = "center"; ctx.fillText(g.msg, W / 2, 44); }

      onScore(Math.floor(g.score));
      raf = requestAnimationFrame(frame);
    }

    canvas.addEventListener("pointerdown", jump);
    return {
      jump,
      start() { if (!g) reset(); if (!running) { running = true; last = 0; raf = requestAnimationFrame(frame); } },
      pause() { running = false; cancelAnimationFrame(raf); },
      stop() { running = false; cancelAnimationFrame(raf); const s = g ? Math.floor(g.score) : 0; best = Math.max(best, s); return s; },
      get best() { return best; },
    };
  }

  return { create, WORLDS };
})();
