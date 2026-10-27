"""The HTTP surface: photograph in, reading out, lots ranked.

One process, one pipeline, one lock. There is a single set of weights and a
single MPS context, and a capstone demonstration has a single photographer, so
requests are serialised rather than made concurrent -- concurrency here would
add failure modes and buy nothing.

The pipeline is built lazily and once. Loading weights takes 1.2 seconds, which
is fine at startup and unusable per request, and deferring it means the tests
can inject a fake and never touch the models at all.
"""

from __future__ import annotations

import io
import math
import os
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.app.images import NotAnImage, TooManyPixels, decode, to_working_scale
from src.app.i18n import describe_message, normalize_locale
from src.app.limits import RateLimiter, TooLarge, read_capped
from src.app.overlay import draw
from src.app.reports import csv_report, pdf_report
from src.app.sessions import Session, SessionRegistry
from src.pipeline import LotReading

STATIC = Path(__file__).parent / "static"

#: Name of the cookie carrying the session id.
COOKIE = "session_id"

#: Whether to believe `X-Forwarded-For` when identifying a caller. Off by
#: default because any client can set that header; turn it on only when this
#: process sits behind a proxy that overwrites it, such as a Cloudflare tunnel.
TRUST_FORWARDED_FOR = os.environ.get("TRUST_FORWARDED_FOR") == "1"

#: Longest lot name accepted. A name is a label on a bag, and it is repeated in
#: every row of the CSV and PDF, so an unbounded one bloats both reports.
MAX_LOT_ID_CHARS = 80

#: Sent with every response. The page loads nothing from anywhere else, so the
#: policy can be strict: no inline script, no framing, no foreign origins.
#: `blob:` is for the local preview of the photograph being measured.
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' blob:; connect-src 'self'; font-src 'self'; "
        "object-src 'none'; base-uri 'none'; form-action 'self'; "
        "frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}

#: Only meaningful over HTTPS, where a browser will honour it.
HSTS = "max-age=31536000; includeSubDomains"


def serialise(reading: LotReading) -> dict:
    """One reading as JSON, carrying its own limits.

    The limits travel with the number deliberately. A damage percentage without
    its resolvable gap, its mode, and whether it was corrected is a figure
    someone will quote out of context -- most likely in a review.
    """
    assessment = reading.assessment
    return {
        "lot_id": reading.lot_id,
        "kernels_counted": reading.kernels_counted,
        "damage_mass_pct": reading.damage_mass_pct,
        "damage_count_pct": reading.damage_count_pct,
        "raw_damage_mass_pct": reading.raw_damage_mass_pct,
        "calibrated": reading.calibration is not None,
        "biological_pct": reading.biological_pct,
        "class_counts": reading.class_counts,
        "temperature_c": reading.temperature_c,
        "moisture_pct_wb": reading.moisture_pct_wb,
        "resolvable_gap_pct": reading.resolvable_gap_pct,
        "degradation_rate": assessment.degradation_rate,
        "mode": assessment.mode,
        "days_to_threshold": assessment.days_to_threshold,
        "days_to_threshold_error_pct": assessment.days_to_threshold_error_pct,
        "suppression_reasons": list(assessment.suppression_reasons),
        "suppression_messages": [
            describe_message(text) for text in assessment.suppression_reasons
        ],
        "model_notes": list(assessment.model_notes),
        "model_messages": [describe_message(text) for text in assessment.model_notes],
        "ranges_ok": assessment.ranges.all_ok,
        "range_notes": list(assessment.ranges.notes),
        "range_messages": [describe_message(text) for text in assessment.ranges.notes],
    }


def _default_pipeline():
    from src.pipeline import Pipeline

    return Pipeline()


def create_app(
    pipeline_factory=_default_pipeline,
    registry: SessionRegistry | None = None,
    limiter: RateLimiter | None = None,
):
    # The generated API docs are off: they are a map of every endpoint for
    # anyone who finds the URL, and the page is the only intended client.
    app = FastAPI(
        title="Maize damage capture", docs_url=None, redoc_url=None, openapi_url=None
    )
    registry = registry if registry is not None else SessionRegistry()
    limiter = limiter if limiter is not None else RateLimiter()

    state = {"pipeline": None}

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = HSTS
        return response

    @app.exception_handler(StarletteHTTPException)
    async def not_found_page(request: Request, exc: StarletteHTTPException):
        """A page for people, JSON for the API.

        A mistyped address in a browser should land somewhere with a way back,
        not on a bare JSON error. API callers keep the JSON they parse.
        """
        if exc.status_code == 404 and not request.url.path.startswith("/api/"):
            return FileResponse(STATIC / "404.html", status_code=404)
        return await http_exception_handler(request, exc)

    # One set of weights and one Metal context, so inference is serialised
    # across every visitor. Sessions isolate what people see, not what the
    # hardware can do at once.
    lock = threading.Lock()

    def pipeline():
        if state["pipeline"] is None:
            state["pipeline"] = pipeline_factory()
        return state["pipeline"]

    def visitor(request: Request, response: Response) -> Session:
        """The caller's session, re-issuing the cookie on every request.

        Re-issuing rather than setting it once keeps the cookie's lifetime in
        step with the session's own idle timeout, so a browser does not hold an
        id the server has already forgotten.
        """
        session_id, session = registry.get_or_create(request.cookies.get(COOKIE))
        response.set_cookie(
            COOKIE,
            session_id,
            httponly=True,
            samesite="lax",
            # Secure only over HTTPS: on a LAN address the app is plain HTTP,
            # and a Secure cookie would never be sent back.
            secure=request.url.scheme == "https",
            max_age=int(registry.idle_timeout_s),
            path="/",
        )
        return session

    def caller(request: Request) -> str:
        """Who to count requests against.

        The peer address, not the session id: a cookie is discarded for free,
        so counting sessions would limit nobody. Behind a tunnel or proxy every
        peer is the proxy, which collapses all callers into one bucket -- read
        the forwarded address there instead, and only there, since a client can
        set that header itself.
        """
        if TRUST_FORWARDED_FOR:
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    @app.post("/api/lots", status_code=201)
    async def add_lot(
        photo: UploadFile = File(...),
        lot_id: str = Form(...),
        temperature_c: float = Form(...),
        moisture_pct_wb: float = Form(...),
        request: Request = None,
        session: Session = Depends(visitor),
    ):
        wait = limiter.check(caller(request))
        if wait is not None:
            raise HTTPException(
                status_code=429,
                detail="Too many measurements. Wait a moment and try again.",
                headers={"Retry-After": str(max(1, int(wait) + 1))},
            )

        # Checked here rather than left to the page: a NaN reading is stored,
        # then cannot be serialised, and every later request for the session
        # fails with it.
        if not lot_id.strip():
            raise HTTPException(status_code=422, detail="the lot name is empty")
        if len(lot_id) > MAX_LOT_ID_CHARS:
            raise HTTPException(
                status_code=422,
                detail=f"the lot name is longer than {MAX_LOT_ID_CHARS} characters",
            )
        if not (math.isfinite(temperature_c) and math.isfinite(moisture_pct_wb)):
            raise HTTPException(
                status_code=422, detail="temperature and moisture must be numbers"
            )

        try:
            raw = await read_capped(photo)
        except TooLarge as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc

        try:
            image = to_working_scale(decode(raw))
        except TooManyPixels as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except NotAnImage as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc

        with lock, TemporaryDirectory() as tmp:
            path = Path(tmp) / "upload.png"
            image.save(path)
            try:
                reading, detections = pipeline().analyse(
                    path,
                    lot_id=lot_id,
                    temperature_c=temperature_c,
                    moisture_pct_wb=moisture_pct_wb,
                )
            except ValueError as exc:
                # A failed photograph never mutates the session.
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            buffer = io.BytesIO()
            draw(image, detections).save(buffer, format="PNG")
            session.overlays[lot_id] = buffer.getvalue()
            session.photographs[lot_id] = raw
            session.store.add(reading)

        return serialise(reading)

    @app.get("/api/lots")
    def list_lots(session: Session = Depends(visitor)):
        return {
            "lots": [serialise(r) for r in session.store.readings()],
            "ranking": [
                {
                    "lot_id": entry.reading.lot_id,
                    "rank": entry.rank,
                    "tied_with": list(entry.tied_with),
                    "degradation_rate": entry.reading.assessment.degradation_rate,
                    "damage_mass_pct": entry.reading.damage_mass_pct,
                    "resolvable_gap_pct": entry.reading.resolvable_gap_pct,
                }
                for entry in session.store.ranking()
            ],
        }

    def report_ranking(session: Session):
        ranking = session.store.ranking()
        if not ranking:
            raise HTTPException(status_code=409, detail="No lots to export.")
        return ranking

    @app.get("/api/report.csv")
    def download_csv(lang: str = "en", session: Session = Depends(visitor)):
        locale = normalize_locale(lang)
        suffix = "" if locale == "en" else f"-{locale}"
        return Response(
            content=csv_report(report_ranking(session), locale),
            media_type="text/csv",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="maize-lot-report{suffix}.csv"'
                )
            },
        )

    @app.get("/api/report.pdf")
    def download_pdf(lang: str = "en", session: Session = Depends(visitor)):
        locale = normalize_locale(lang)
        suffix = "" if locale == "en" else f"-{locale}"
        return Response(
            content=pdf_report(report_ranking(session), locale),
            media_type="application/pdf",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="maize-lot-report{suffix}.pdf"'
                )
            },
        )

    # `:path` so a lot named "A/2" still has an address: the page sends the
    # slash encoded, and the router sees it decoded.
    @app.get("/api/lots/{lot_id:path}/overlay.png")
    def overlay(lot_id: str, session: Session = Depends(visitor)):
        if lot_id not in session.overlays:
            raise HTTPException(status_code=404, detail=f"no lot {lot_id!r}")
        return Response(content=session.overlays[lot_id], media_type="image/png")

    @app.delete("/api/lots/{lot_id:path}", status_code=204)
    def delete_lot(lot_id: str, session: Session = Depends(visitor)):
        if not session.store.remove(lot_id):
            raise HTTPException(status_code=404, detail=f"no lot {lot_id!r}")
        session.overlays.pop(lot_id, None)
        session.photographs.pop(lot_id, None)
        return Response(status_code=204)

    @app.post("/api/session/reset", status_code=204)
    def reset(session: Session = Depends(visitor)):
        session.store.reset()
        session.overlays.clear()
        session.photographs.clear()
        return Response(status_code=204)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
