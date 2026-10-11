// KDAA handbook: theme toggle, search over search.json, copy buttons, "/" to search.
(function () {
  var root = document.documentElement;

  // theme: system by default; the button cycles system -> dark -> light; remembered per browser
  var KEY = "kdaa-handbook-theme";
  function applyTheme(t) {
    if (t === "dark" || t === "light") root.setAttribute("data-theme", t);
    else root.removeAttribute("data-theme");
  }
  try { applyTheme(localStorage.getItem(KEY)); } catch (e) {}
  var themeBtn = document.getElementById("theme");
  if (themeBtn) themeBtn.addEventListener("click", function () {
    var cur = root.getAttribute("data-theme");
    var next = cur === null ? "dark" : cur === "dark" ? "light" : null;
    applyTheme(next);
    try { if (next) localStorage.setItem(KEY, next); else localStorage.removeItem(KEY); } catch (e) {}
    themeBtn.title = next ? (next === "dark" ? "어두운 화면" : "밝은 화면") : "기기 설정 따름";
  });

  // copy buttons on .code blocks
  document.querySelectorAll(".code").forEach(function (box) {
    var pre = box.querySelector("pre");
    if (!pre) return;
    var b = document.createElement("button");
    b.type = "button"; b.className = "copy"; b.textContent = "복사";
    b.addEventListener("click", function () {
      var text = pre.innerText.replace(/\s+$/, "");
      var done = function () { b.textContent = "복사됨"; setTimeout(function () { b.textContent = "복사"; }, 1400); };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () { select(pre); });
      } else { select(pre); }
    });
    box.appendChild(b);
  });
  function select(el) {
    var r = document.createRange(); r.selectNodeContents(el);
    var s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
  }

  // search
  var q = document.getElementById("q"), list = document.getElementById("results");
  if (!q || !list) return;
  var prefix = q.getAttribute("data-prefix") || "";
  var index = null, loading = null, sel = -1;
  function load() {
    if (index || loading) return loading;
    loading = fetch(prefix + "search.json").then(function (r) { return r.json(); })
      .then(function (d) { index = d; return d; })
      .catch(function () { index = []; return index; });
    return loading;
  }
  function esc(s) { return s.replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function render() {
    var term = q.value.trim().toLowerCase();
    if (!term) { list.hidden = true; return; }
    load().then(function () {
      var words = term.split(/\s+/);
      var hits = index.filter(function (e) {
        var hay = (e.t + " " + e.g + " " + e.x).toLowerCase();
        return words.every(function (w) { return hay.indexOf(w) >= 0; });
      }).slice(0, 30);
      sel = -1;
      list.innerHTML = hits.length ? hits.map(function (e) {
        var x = e.x ? e.x.slice(0, 110) + (e.x.length > 110 ? "…" : "") : "";
        return '<li><a href="' + prefix + e.u + '">' + esc(e.t) + "<small>" + esc(e.g) + (x ? " · " + esc(x) : "") + "</small></a></li>";
      }).join("") : '<li class="none">찾는 내용이 없습니다. 다른 낱말로 찾아 보세요.</li>';
      list.hidden = false;
    });
  }
  q.addEventListener("input", render);
  q.addEventListener("focus", function () { load(); if (q.value.trim()) render(); });
  q.addEventListener("keydown", function (ev) {
    var items = list.querySelectorAll("a");
    if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
      if (!items.length) return;
      ev.preventDefault();
      sel = (sel + (ev.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      items.forEach(function (a, i) { a.classList.toggle("sel", i === sel); });
    } else if (ev.key === "Enter" && items.length) {
      ev.preventDefault();
      location.href = items[Math.max(sel, 0)].href;
    } else if (ev.key === "Escape") {
      list.hidden = true; q.blur();
    }
  });
  document.addEventListener("click", function (ev) { if (!ev.target.closest(".search")) list.hidden = true; });
  document.addEventListener("keydown", function (ev) {
    if (ev.key === "/" && document.activeElement !== q && !/input|textarea/i.test(document.activeElement.tagName)) {
      ev.preventDefault(); q.focus();
    }
  });
})();
