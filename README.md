# Hybrid AI-Based Maize Quality Assessment and Deterioration Risk Ranking

Measures mechanical damage in shelled maize from a photograph and uses it as the
input to a published deterioration model, ranking storage lots by how fast they
are degrading.

**Hybrid** because only the vision half learns anything. Temperature, moisture
and storage duration never touch a model — they go into Steele's published
equations. That split is deliberate and is what lets the results be defended
without a storage trial.

The primary output is a **ranking**, because it survives inputs outside the
published validity ranges and dried bagged maize often sits below the moisture
floor. Absolute days-to-threshold is secondary: it appears whenever the inputs
are in range, comes from Steele's own reference times (1967, p. 108), and never
appears without his stated standard error beside it.

`src/physics/steele_agreement.py` checks the whole model against numbers Steele
published — his worked multiplier values, his Table 9 adjustments, and nine
measured storage lives. Run it to print the table:

    .venv\Scripts\python.exe -m src.physics.steele_agreement

Damage readings ship **uncorrected**. The classifier errs in both directions,
which compresses every reading toward the middle of the range — a clean tray
reads about 1.4% damaged and a 40% tray about 2.4 points low. That bias is
affine and therefore invertible, and `src/calibrate_pipeline.py` fits the
inverse. It is off by default because the fit does not transfer between
acquisition sessions: three estimates of the same line, from the same corpus,
range from `0.96 + 0.842x` to `4.75 + 0.774x`. Correcting with the wrong
session's line is worse than not correcting. Fit it per session, against
calibration trays photographed on the same rig on the same day.

The classifier the pipeline loads must be one trained on **whole** images.
Splitting GrainSet's paired views is right for training and wrong here: the
split-view model scores 0.8893 macro-F1 on its own split, then reads 16.91%
damage off a tray with none. Checkpoints record their framing and the pipeline
refuses a mismatch, because macro-F1 does not show it.

## Layout

- `src/physics/` — deterioration model. Pure arithmetic over published
  constants, no learned parameters, no dependency on the vision pipeline.
- `tests/` — each test pins a specific way the model can be silently wrong.
- `data/raw/grainset/` — GrainSet maize subset and its data card (gitignored).

## The capture app

`src/app/` serves a phone-facing page that photographs a tray, ranks lots, and
downloads the current visitor's evidence as PDF or CSV. See `src/app/README.md`
for how to run it and reach it from a phone.

## Running the tests

    py -3.13 -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    .venv\Scripts\python.exe -m pytest tests/ -q
    node --test tests/test_i18n_js.cjs

Run these from `stored-grain/`. On macOS/Linux, create the environment with
`python3 -m venv .venv` and use `.venv/bin/python` in place of
`.venv\Scripts\python.exe`.

Planning documents live one level up in `../`.
