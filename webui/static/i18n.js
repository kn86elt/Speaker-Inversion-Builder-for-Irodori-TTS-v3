// i18n module – loaded before app.js (no defer)
(function () {
  var _lang = localStorage.getItem('ui_lang') || 'ja';
  var _dict = {};

  function t(key) {
    return Object.prototype.hasOwnProperty.call(_dict, key) ? _dict[key] : key;
  }

  function apply() {
    document.querySelectorAll('[data-i18n]').forEach(function (el) {
      el.textContent = t(el.dataset.i18n);
    });
    document.querySelectorAll('[data-i18n-html]').forEach(function (el) {
      el.innerHTML = t(el.dataset.i18nHtml);
    });
    document.querySelectorAll('[data-i18n-attr]').forEach(function (el) {
      var parts = el.dataset.i18nAttr.split(':');
      // format: "attr:key"  e.g. "title:btn_clear_sel_title"
      el.setAttribute(parts[0], t(parts[1]));
    });
    var sel = document.getElementById('lang-select');
    if (sel && sel.value !== _lang) sel.value = _lang;
    document.documentElement.lang = _lang;
  }

  function load(lang) {
    return fetch('/static/i18n/' + lang + '.json?v=1')
      .then(function (r) {
        if (!r.ok) throw new Error('i18n load failed: ' + r.status);
        return r.json();
      })
      .then(function (d) {
        _dict = d;
        _lang = lang;
        localStorage.setItem('ui_lang', lang);
        apply();
      });
  }

  function setLang(lang) {
    return load(lang);
  }

  function getLang() { return _lang; }

  var _ready = load(_lang).catch(function () {});

  window.I18N = { t: t, setLang: setLang, getLang: getLang, apply: apply, _ready: _ready };
  window.t = t;
})();
