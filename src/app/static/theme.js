// Theme before first paint, so the page never flashes the wrong colours.
// Loaded as a blocking script in <head>: a file rather than inline so the
// Content-Security-Policy can refuse every inline script.
(function () {
  var theme = null;
  try { theme = localStorage.getItem("grain-theme"); } catch (e) {}
  if (theme !== "light" && theme !== "dark") {
    theme = window.matchMedia && matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }
  document.documentElement.setAttribute("data-theme", theme);
}());
