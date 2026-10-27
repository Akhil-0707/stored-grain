(function (root, factory) {
  var api = factory(root && root.document);
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.GrainI18n = api;
}(typeof window !== "undefined" ? window : globalThis, function (document) {
  "use strict";

  var supported = ["en", "hi", "te"];
  var catalogs = Object.create(null);
  var locale = "en";
  var listeners = [];
  var englishFallback = {
    "confirm.clear_session": "Clear all measured lots from this session?",
    "confirm.remove_lot": "Remove {lot} from this session?",
    "error.not_image": "Choose a PNG or JPEG image.",
    "error.no_kernels": "No kernels were detected in that image.",
    "error.server_request": "The measurement server could not process that request.",
    "error.photo_too_large": "Choose a photo smaller than 32 MB.",
    "error.too_many": "Too many requests. Try again shortly.",
    "error.server_unavailable": "The measurement server is unavailable.",
    "error.rejected": "The server rejected that request ({status}).",
    "error.bad_reply": "The server returned an invalid response.",
    "error.network": "Network error. Check your connection and try again.",
    "error.unknown": "An unknown error occurred.",
    "error.language_load": "Could not load the selected language.",
    "validation.lot": "Lot ID is required.",
    "validation.photo": "A sample photo is required.",
    "validation.temperature": "Enter a temperature between -10 and 60 °C.",
    "validation.moisture": "Enter moisture between 1 and 60% wb.",
    "status.measuring": "Measuring…",
    "status.measured": "Measured {lot}.",
    "status.overlay_failed": "Measured {lot}, but the outlined image did not load. Reload to see it.",
    "status.removing": "Removing {lot}…",
    "status.removed": "Removed {lot}.",
    "status.clearing": "Clearing session…",
    "status.cleared": "Session cleared.",
    "mode.ranking_absolute": "Absolute estimate",
    "ranking.empty": "Measure at least one lot to see a ranking.",
    "ranking.rate": "Degradation rate",
    "ranking.remove": "Remove {lot}",
    "ranking.tie": "Cannot be separated at this sample size — the damage gap is smaller than {gap} points.",
    "reading.biological": "Biological damage",
    "reading.by_count": "By count",
    "reading.damage_by_weight": "Damage by weight",
    "reading.days_to_loss": "Days to threshold",
    "reading.legend.damage": "Mechanical damage",
    "reading.legend.sound": "Sound kernels",
    "reading.mode": "Mode",
    "reading.moisture": "Moisture",
    "reading.none_reported": "None reported",
    "reading.outside_range": "Outside the published validity range: {reasons}",
    "reading.overlay_alt": "Outlined kernels and detected damage",
    "reading.range_unknown": "Validity range unknown",
    "reading.resolvable": "±{gap} resolvable · {count} kernels counted",
    "reading.temperature": "Temperature",
    "reading.threshold_withheld": "Days to threshold withheld: {reasons}",
    "reading.uncorrected": "Uncorrected"
  };

  function normalize(value) {
    var code = String(value || "").trim().toLowerCase().split(/[-_]/)[0];
    return supported.indexOf(code) >= 0 ? code : "en";
  }

  function load(code) {
    if (catalogs[code]) return Promise.resolve(catalogs[code]);
    return fetch("/static/locales/" + code + ".json").then(function (response) {
      if (!response.ok) throw new Error("catalog unavailable");
      return response.json();
    }).then(function (catalog) {
      catalogs[code] = catalog;
      return catalog;
    });
  }

  function interpolate(text, values) {
    return String(text).replace(/\{([^{}]+)\}/g, function (match, name) {
      return values && Object.prototype.hasOwnProperty.call(values, name)
        ? String(values[name])
        : match;
    });
  }

  function t(key, values) {
    var english = catalogs.en || englishFallback;
    var active = catalogs[locale] || english;
    var text = Object.prototype.hasOwnProperty.call(active, key) ? active[key]
      : Object.prototype.hasOwnProperty.call(english, key) ? english[key]
        : englishFallback[key];
    return text == null ? key : interpolate(text, values);
  }

  function message(descriptor) {
    var english = catalogs.en || englishFallback;
    var known = descriptor && descriptor.key && (
      Object.prototype.hasOwnProperty.call(english, descriptor.key)
      || Object.prototype.hasOwnProperty.call(englishFallback, descriptor.key)
    );
    if (!known) {
      return descriptor && descriptor.fallback != null ? String(descriptor.fallback) : "";
    }
    return t(descriptor.key, descriptor.values || {});
  }

  function createMessageState() {
    var descriptor = null;
    return {
      set: function (next) { descriptor = next || null; },
      render: function () { return message(descriptor); }
    };
  }

  function apply(root) {
    if (!document) return;
    var scope = root || document;
    [
      ["[data-i18n]", "textContent", "data-i18n"],
      ["[data-i18n-placeholder]", "placeholder", "data-i18n-placeholder"],
      ["[data-i18n-aria-label]", "aria-label", "data-i18n-aria-label"],
      ["[data-i18n-content]", "content", "data-i18n-content"]
    ].forEach(function (rule) {
      scope.querySelectorAll(rule[0]).forEach(function (element) {
        var key = element.getAttribute(rule[2]);
        var translated = t(key);
        if (translated === key) return;
        if (rule[1] === "textContent") {
          element.textContent = translated;
        } else {
          element.setAttribute(rule[1], translated);
        }
      });
    });
  }

  function updateDocument() {
    if (!document) return;
    document.documentElement.lang = locale;
    apply(document);
    document.querySelectorAll("[data-locale]").forEach(function (button) {
      button.setAttribute("aria-pressed", button.getAttribute("data-locale") === locale ? "true" : "false");
    });
  }

  function notify() {
    listeners.slice().forEach(function (listener) { listener(locale); });
  }

  function init() {
    locale = "en";
    return load("en").then(function () {
      updateDocument();
      return locale;
    }).catch(function () {
      updateDocument();
      return locale;
    });
  }

  function setLocale(value) {
    var next = normalize(value);
    return load(next).then(function () {
      locale = next;
      updateDocument();
      notify();
      return locale;
    }).catch(function (error) {
      locale = "en";
      updateDocument();
      notify();
      throw error;
    });
  }

  function subscribe(listener) {
    listeners.push(listener);
    return function () {
      listeners = listeners.filter(function (candidate) { return candidate !== listener; });
    };
  }

  return {
    init: init,
    setLocale: setLocale,
    getLocale: function () { return locale; },
    t: t,
    message: message,
    createMessageState: createMessageState,
    apply: apply,
    subscribe: subscribe
  };
}));
