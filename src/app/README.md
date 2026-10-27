# Capture app

Photograph trays of maize from a phone, measure mechanical damage, rank lots.

## Running it

On Windows, double-click `run-app.bat` in `stored-grain/`, or from that folder:

    .venv\Scripts\python.exe -m uvicorn src.app.server:app --host 0.0.0.0 --port 8000

Start it from `stored-grain/`: the model checkpoints are loaded from
`models/` relative to the current folder — `localiser-real.pt` (the kernel
localiser fine-tuned on real phone photos of trays) and
`classifier.session.pt`. On macOS/Linux the interpreter is `.venv/bin/python`
instead.

`--host 0.0.0.0` is what makes the phone able to reach it. Find the laptop's
address on the network with `ipconfig` (the `IPv4 Address` line of the Wi-Fi
adapter), then open `http://<that address>:8000` in Chrome on the phone. A
personal hotspot from the phone works and needs no wifi. The first time the
server starts, Windows Defender Firewall asks whether Python may accept
connections: allow it on **private** networks, and set the hotspot or Wi-Fi
network's profile to Private, or the phone cannot connect.

Models load on the first photograph, not at startup, so the first measurement
takes about three seconds and the rest about two.

After at least one lot is measured, the ranking panel offers PDF and CSV
downloads. Both reports are scoped to the current browser session and include
the rank, entered storage conditions, measured damage, kernel count,
measurement resolution, degradation result, and days-to-threshold when
available. The CSV retains the model-warning fields for analysis; the
presentation-focused PDF omits notes. Empty sessions return no report.

Report language follows the browser's selected language (`en`, `hi`, or `te`)
through the `?lang=` query parameter. Downloads are session-scoped; empty
sessions return no report.

## Interface

The header offers English, Hindi, and Telugu. English is restored on refresh;
the selected language is held in memory and is not persisted. All visible and
accessibility text, validation/status messages, and generated report text are
curated and offline. PDF and CSV use the selected language; CSV is UTF-8 BOM
encoded. Lot names, numbers, units, model names, and citations are preserved.
Indic PDF fonts are bundled (`src/app/fonts/`); ReportLab shaping requires the
pinned `uharfbuzz==0.55.0` dependency. The PDF shaping path is experimental
across deployment devices; Hindi and Telugu Poppler renders were visually
checked in this environment.

The browser UI has two themes, "Silo at night" (dark) and "Husk paper"
(light), with maize gold as the accent, sage for sound kernels and ember for
damage. The sun/moon button in the header switches them; the first visit
follows the device setting and the choice is remembered under the
`grain-theme` key. Language is still never persisted. A three-step rail
(capture, measure, rank) lights up as the assay moves. Desktop keeps the
capture station beside the reading and ranking workspace; phone layouts
collapse into the same sequence. Static CSS and JavaScript URLs are versioned
in `index.html` so layout changes cannot pair with stale cached assets.

`static/ui.js` is presentation only: it observes what `app.js` renders and
never touches measurement, API or translation logic. It draws the "grain
field" background (kernel-shaped grains, faintly linked, pushed aside by the
cursor, rippled by a tap, faster while a tray is measured), the cursor
spotlight on cards, the count-up of the damage figure, and the slide of lots
into their new ranking positions. The background starts after first paint,
caps its grain count and frame rate on phones and low-memory devices, pauses
in hidden tabs, and stays still under `prefers-reduced-motion`. No fonts,
images, or scripts are fetched from third parties.

## Reaching it from outside the network

A tunnel publishes the local server on an HTTPS URL without deploying
anything. The models, the weights and the inference stay on this machine, which
is the point: a real deploy would have to ship 120 MB of checkpoints and a
536 MB PyTorch install, and would then run on a CPU with no Metal backend.

    winget install --id Cloudflare.cloudflared
    cloudflared tunnel --url http://localhost:8000

It prints a `https://<name>.trycloudflare.com` address that works from any
network. Free, and no account for a quick tunnel.

**Do not depend on a tunnel for the November review.** A lecture room demonstration
should not depend on Cloudflare and campus internet both working at the same
minute. Use the LAN address, or a personal hotspot with the laptop joined to it,
and test that path in the room beforehand.

Visitor data is isolated by an HTTP-only session cookie. Upload byte and decoded
pixel limits, rate limiting, and session expiry are enforced by the server.
This still is a local capstone demonstration, not a production deployment.

## What it does not do

No durable persistence: sessions live in the process and expire. No
per-session calibration mode; that becomes useful only once weighed mixtures
exist to photograph.

## The numbers it shows

Every reading is uncorrected and comes from models trained on synthetic trays
from one laboratory rig. This limitation remains documented in the locale
catalogs, project handoff, master plan, and analysis-oriented report content;
the compact browser interface no longer renders a separate warning banner.
