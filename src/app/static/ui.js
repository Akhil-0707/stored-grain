"use strict";

// Presentation-only behaviour: theme, the "grain field" background, the
// process rail, and small transitions. It only observes the DOM that app.js
// renders and never touches measurement, API or i18n logic.
(function () {

  var root = document.documentElement;
  var body = document.body;
  var reduceMotion = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)");
  var coarse = window.matchMedia && matchMedia("(pointer: coarse)").matches;
  var lowEnd = coarse || (navigator.hardwareConcurrency || 8) <= 4 || (navigator.deviceMemory || 8) <= 4;

  function reduced() { return !!(reduceMotion && reduceMotion.matches); }

  // ------------------------------------------------------------ theme

  var themeMeta = document.querySelector('meta[name="theme-color"]');

  function setTheme(theme) {
    root.setAttribute("data-theme", theme);
    if (themeMeta) themeMeta.setAttribute("content", theme === "light" ? "#f3efe3" : "#0b0e0a");
    try { localStorage.setItem("grain-theme", theme); } catch (e) { /* private mode */ }
    if (field) field.recolour();
  }

  var toggle = document.getElementById("theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      setTheme(root.getAttribute("data-theme") === "light" ? "dark" : "light");
    });
  }
  if (themeMeta) themeMeta.setAttribute("content", root.getAttribute("data-theme") === "light" ? "#f3efe3" : "#0b0e0a");

  // ---------------------------------------------------- grain field

  // Kernel-shaped grains drifting slowly, faintly linked when close (a
  // sample being drawn), pushed aside by the cursor or a finger like grain
  // sifted by hand. Speeds up while a tray is measured; ripples when a
  // reading lands. Sized to the screen, capped for phones, paused when the
  // tab is hidden, and drawn once (still) under reduced motion.
  var field = (function () {
    var canvas = document.getElementById("field");
    if (!canvas || !canvas.getContext) return null;
    var ctx = canvas.getContext("2d");
    var dpr = Math.min(window.devicePixelRatio || 1, lowEnd ? 1 : 1.5);
    var W = 0, H = 0, grains = [], pointer = null, ripples = [];
    var energy = 1, energyTarget = 1, frame = 0, last = 0, running = false;
    var frameGap = lowEnd ? 1000 / 30 : 0;
    var LINK = lowEnd ? 90 : 120, PUSH_R = 130, PUSH = 70;
    var palette = { rgb: "242 182 50", alpha: 0.42, line: 0.11 };

    function recolour() {
      var s = getComputedStyle(root);
      palette.rgb = s.getPropertyValue("--field-rgb").trim() || palette.rgb;
      palette.alpha = parseFloat(s.getPropertyValue("--field-alpha")) || palette.alpha;
      palette.line = parseFloat(s.getPropertyValue("--field-line")) || palette.line;
      if (!running) draw();
    }

    function size() {
      W = window.innerWidth; H = window.innerHeight;
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      var want = Math.max(16, Math.min(lowEnd ? 30 : 64, Math.round((W * H) / 19000)));
      while (grains.length < want) grains.push(grain(Math.random() * W, Math.random() * H));
      grains.length = want;
    }

    function grain(x, y) {
      var a = Math.random() * Math.PI * 2, v = 5 + Math.random() * 9;
      return { x: x, y: y, vx: Math.cos(a) * v, vy: Math.sin(a) * v,
               r: 1.6 + Math.random() * 1.5, rot: Math.random() * Math.PI * 2,
               spin: (Math.random() - 0.5) * 0.4 };
    }

    function step(dt) {
      energy += (energyTarget - energy) * Math.min(1, dt * 2.5);
      for (var i = 0; i < grains.length; i++) {
        var g = grains[i];
        g.x += g.vx * dt * energy; g.y += g.vy * dt * energy; g.rot += g.spin * dt * energy;
        if (pointer) push(g, pointer.x, pointer.y, PUSH_R, PUSH * dt);
        for (var k = 0; k < ripples.length; k++) {
          var rp = ripples[k], dx = g.x - rp.x, dy = g.y - rp.y, d = Math.sqrt(dx * dx + dy * dy);
          if (d > 0 && Math.abs(d - rp.r) < 40) { g.x += dx / d * 40 * dt * rp.life; g.y += dy / d * 40 * dt * rp.life; }
        }
        if (g.x < -20) g.x += W + 40; else if (g.x > W + 20) g.x -= W + 40;
        if (g.y < -20) g.y += H + 40; else if (g.y > H + 20) g.y -= H + 40;
      }
      for (var j = ripples.length - 1; j >= 0; j--) {
        ripples[j].r += 260 * dt; ripples[j].life -= dt * 0.8;
        if (ripples[j].life <= 0) ripples.splice(j, 1);
      }
    }

    function push(g, px, py, radius, amount) {
      var dx = g.x - px, dy = g.y - py, d = Math.sqrt(dx * dx + dy * dy);
      if (d > 0 && d < radius) {
        var f = (1 - d / radius); f = f * f * amount;
        g.x += dx / d * f; g.y += dy / d * f;
      }
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      var rgb = palette.rgb, lineA = palette.line * (0.8 + 0.4 * (energy - 1));
      ctx.lineWidth = 1;
      for (var i = 0; i < grains.length; i++) {
        var a = grains[i];
        for (var j = i + 1; j < grains.length; j++) {
          var b = grains[j], dx = a.x - b.x, dy = a.y - b.y;
          if (dx > LINK || dx < -LINK || dy > LINK || dy < -LINK) continue;
          var d = Math.sqrt(dx * dx + dy * dy);
          if (d >= LINK) continue;
          ctx.strokeStyle = "rgb(" + rgb + " / " + ((1 - d / LINK) * lineA).toFixed(3) + ")";
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        }
      }
      ctx.fillStyle = "rgb(" + rgb + " / " + Math.min(0.9, palette.alpha * (0.85 + 0.25 * energy)).toFixed(3) + ")";
      for (var k = 0; k < grains.length; k++) {
        var g = grains[k];
        ctx.beginPath();
        // A maize kernel: a slightly pointed ellipse.
        ctx.ellipse(g.x, g.y, g.r, g.r * 1.38, g.rot, 0, Math.PI * 2);
        ctx.fill();
      }
      for (var m = 0; m < ripples.length; m++) {
        var rp = ripples[m];
        ctx.strokeStyle = "rgb(" + rgb + " / " + (0.22 * rp.life).toFixed(3) + ")";
        ctx.beginPath(); ctx.arc(rp.x, rp.y, rp.r, 0, Math.PI * 2); ctx.stroke();
      }
    }

    function tick(now) {
      frame = 0;
      if (!running) return;
      if (frameGap && now - last < frameGap) { frame = requestAnimationFrame(tick); return; }
      var dt = last ? Math.min(0.05, (now - last) / 1000) : 0.016;
      last = now;
      step(dt); draw();
      frame = requestAnimationFrame(tick);
    }

    function start() {
      if (running || reduced() || document.hidden) { draw(); return; }
      running = true; last = 0; frame = requestAnimationFrame(tick);
    }
    function stop() { running = false; if (frame) cancelAnimationFrame(frame); frame = 0; }

    size(); recolour();
    var resizeTimer = 0;
    window.addEventListener("resize", function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () { size(); draw(); }, 150);
    });
    document.addEventListener("visibilitychange", function () { document.hidden ? stop() : start(); });
    if (reduceMotion && reduceMotion.addEventListener) {
      reduceMotion.addEventListener("change", function () { reduced() ? stop() : start(); });
    }
    window.addEventListener("pointermove", function (e) {
      if (e.pointerType === "touch") return;
      pointer = { x: e.clientX, y: e.clientY };
    }, { passive: true });
    window.addEventListener("pointerleave", function () { pointer = null; });
    document.addEventListener("mouseleave", function () { pointer = null; });
    window.addEventListener("pointerdown", function (e) {
      if (e.pointerType === "touch") ripple(e.clientX, e.clientY);
    }, { passive: true });

    function ripple(x, y) {
      if (reduced()) return;
      ripples.push({ x: x, y: y, r: 6, life: 1 });
      if (ripples.length > 4) ripples.shift();
    }

    return {
      start: start,
      recolour: recolour,
      excite: function (on) { energyTarget = on ? 3.2 : 1; },
      ripple: ripple
    };
  }());

  // Start drawing after first paint so it never delays the page.
  if (field) {
    var go = function () { field.start(); };
    if (window.requestIdleCallback) requestIdleCallback(go, { timeout: 600 });
    else setTimeout(go, 200);
  }

  // ---------------------------------------------------- cursor spotlight

  if (window.matchMedia && matchMedia("(hover: hover)").matches) {
    var spotEvent = null, spotFrame = 0;
    document.addEventListener("pointermove", function (e) {
      spotEvent = e;
      if (!spotFrame) spotFrame = requestAnimationFrame(function () {
        spotFrame = 0;
        var panel = spotEvent.target.closest && spotEvent.target.closest(".panel");
        if (!panel) return;
        var r = panel.getBoundingClientRect();
        panel.style.setProperty("--spot-x", Math.round(spotEvent.clientX - r.left) + "px");
        panel.style.setProperty("--spot-y", Math.round(spotEvent.clientY - r.top) + "px");
      });
    }, { passive: true });
  }

  // ----------------------------------------------------- process rail

  var dropzone = document.getElementById("dropzone");
  var resultBody = document.getElementById("result-body");
  var rankingBody = document.getElementById("ranking-body");
  var resultPanel = document.getElementById("result");
  var wasMeasuring = false, lastReading = "";

  function updateStages() {
    var photo = dropzone && dropzone.getAttribute("data-empty") === "false";
    var measuring = !!resultBody.querySelector('.shot[data-state="loading"]');
    var reading = !!resultBody.querySelector(".readout");
    var ranked = !!rankingBody.querySelector(".rung");

    // capture -> measure -> rank; after a ranking exists, capture invites the next lot.
    var s;
    if (measuring) s = ["done", "active", ranked ? "done" : "idle"];
    else if (photo) s = ["done", "active", ranked ? "done" : "idle"];
    else if (ranked || reading) s = ["active", "done", "done"];
    else s = ["active", "idle", "idle"];
    body.setAttribute("data-s1", s[0]);
    body.setAttribute("data-s2", s[1]);
    body.setAttribute("data-s3", s[2]);

    if (field) {
      if (measuring && !wasMeasuring) field.excite(true);
      if (!measuring && wasMeasuring) {
        field.excite(false);
        if (reading && resultPanel) {
          var r = resultPanel.getBoundingClientRect();
          field.ripple(r.left + r.width / 2, Math.max(80, r.top + 120));
        }
      }
    }
    wasMeasuring = measuring;
  }

  // ------------------------------------------------- count-up readout

  function countUp() {
    var el = resultBody.querySelector(".readout-value .num");
    if (!el) return;
    var target = parseFloat(el.textContent);
    var key = el.textContent + "|" + (resultBody.querySelector(".readout-band") || {}).textContent;
    if (!isFinite(target) || key === lastReading) { lastReading = key; return; }
    lastReading = key;
    if (reduced()) return;
    var decimals = (el.textContent.split(".")[1] || "").length;
    var t0 = performance.now(), dur = 750;
    (function frame(now) {
      var p = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - p, 3);
      el.textContent = (target * e).toFixed(decimals);
      if (p < 1) requestAnimationFrame(frame);
    }(t0));
  }

  // -------------------------------------------- ranking reorder (FLIP)

  var lastRects = {};

  function rungKey(rung) {
    var b = rung.querySelector(".rung-remove");
    return b ? b.getAttribute("data-lot") : null;
  }

  function flipRanking() {
    var base = rankingBody.getBoundingClientRect().top;
    var rungs = rankingBody.querySelectorAll(".rung");
    var next = {};
    rungs.forEach(function (rung) {
      var key = rungKey(rung);
      if (!key) return;
      var top = rung.getBoundingClientRect().top - base;
      next[key] = top;
      if (!reduced() && Object.prototype.hasOwnProperty.call(lastRects, key)) {
        var delta = lastRects[key] - top;
        if (Math.abs(delta) > 2) {
          rung.style.animation = "none";
          rung.style.transform = "translateY(" + delta + "px)";
          rung.style.transition = "none";
          requestAnimationFrame(function () {
            rung.style.transition = "transform 0.55s cubic-bezier(0.22, 1, 0.36, 1)";
            rung.style.transform = "";
          });
        } else {
          rung.style.animation = "none";
        }
      }
    });
    lastRects = next;
  }

  // ---------------------------------------------------------- observe

  var pending = 0;
  function schedule(fn) {
    return function () {
      if (pending) return;
      pending = requestAnimationFrame(function () { pending = 0; fn(); });
    };
  }

  var onResult = schedule(function () { updateStages(); countUp(); });
  var onRanking = function () { updateStages(); flipRanking(); };

  if (window.MutationObserver) {
    // Direct children only: app.js replaces them per render, while the
    // count-up edits text deeper down and must not re-trigger itself.
    new MutationObserver(onResult).observe(resultBody, { childList: true });
    new MutationObserver(onRanking).observe(rankingBody, { childList: true });
    new MutationObserver(updateStages).observe(dropzone, { attributes: true, attributeFilter: ["data-empty"] });
  }
  updateStages();

}());
