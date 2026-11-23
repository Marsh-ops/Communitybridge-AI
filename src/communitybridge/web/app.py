"""The FastAPI app: server-rendered pages for the interface layer.

Serves: the interface layer of the design diagram. Every screen the user sees (goal entry, plan, approvals,
trace, escalation) is a Jinja2 template rendered on the server and sent as plain HTML.

Why it is designed this way:
- Server-rendered semantic HTML works with a keyboard and a screen reader without any JavaScript,
  and it keeps the approval gates out of the browser: a gate is a POSTed form the server records,
  so a page can never "approve" something by itself.
- `create_app()` is a factory so each test gets a separate app, and later steps can pass in a
  database path without a global.

This module only wires up the web layer. It holds no eligibility, logistics or consent logic:
those stay in the deterministic tools and the graph.
"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

from communitybridge.web.journeys import (
    JourneyNotFoundError,
    JourneyService,
    SetupProblemError,
    WithdrawalRefusedError,
)
from communitybridge.web.routes import (
    BadFormError,
    GateMovedOnError,
    PlanNotReadyError,
    build_router,
)

_WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = _WEB_DIR / "templates"
STATIC_DIR = _WEB_DIR / "static"


def create_app(*, service: JourneyService | None = None) -> FastAPI:
    """Build a new app with its templates, static files and pages.

    `service` runs the journeys; tests pass one built on the FakeProvider and temporary files.
    """
    # FastAPI's built-in /docs, /redoc and /openapi.json pages are switched off: they are not
    # screens we audit for WCAG, and a user of this prototype has no use for them.
    app = FastAPI(
        title="CommunityBridge AI",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    # Jinja2Templates escapes every value in an .html template by default, so text from a user's
    # goal can never inject markup into a page.
    templates = Jinja2Templates(directory=TEMPLATES_DIR)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(build_router(service or JourneyService(), templates))

    def message(request: Request, status: int, heading: str, paragraphs: list[str]) -> HTMLResponse:
        """Every error is a normal page with a heading and a way back, never bare JSON."""
        return templates.TemplateResponse(
            request,
            "message.html",
            {"nav_current": "", "heading": heading, "paragraphs": paragraphs},
            status_code=status,
        )

    @app.exception_handler(StarletteHTTPException)
    def not_a_page(request: Request, exc: StarletteHTTPException) -> HTMLResponse:
        if exc.status_code == 404:
            return message(request, 404, "Page not found", ["There is nothing at this address."])
        return message(request, exc.status_code, "That did not work", ["Please start again."])

    @app.exception_handler(JourneyNotFoundError)
    def no_such_journey(request: Request, exc: JourneyNotFoundError) -> HTMLResponse:
        return message(
            request,
            404,
            "We cannot find that request",
            ["The address may be wrong, or the request may have been removed."],
        )

    @app.exception_handler(GateMovedOnError)
    def gate_moved_on(request: Request, exc: GateMovedOnError) -> HTMLResponse:
        return message(
            request,
            409,
            "This step has already been answered",
            ["Nothing was changed. Open the request again to see where it is up to."],
        )

    @app.exception_handler(PlanNotReadyError)
    def no_plan_yet(request: Request, exc: PlanNotReadyError) -> HTMLResponse:
        return message(
            request,
            409,
            "There is no plan to show yet",
            ["Your details need to be confirmed first. Nothing has been booked or shared."],
        )

    @app.exception_handler(BadFormError)
    def bad_form(request: Request, exc: BadFormError) -> HTMLResponse:
        return message(request, 422, "That could not be read", [str(exc)])

    @app.exception_handler(WithdrawalRefusedError)
    def withdrawal_refused(request: Request, exc: WithdrawalRefusedError) -> HTMLResponse:
        return message(
            request,
            409,
            "That consent cannot be withdrawn now",
            [
                "It may not have been approved, it may already be withdrawn, "
                "or an answer is still due."
            ],
        )

    @app.exception_handler(SetupProblemError)
    def not_set_up(request: Request, exc: SetupProblemError) -> HTMLResponse:
        return message(
            request,
            503,
            "This prototype is not set up yet",
            ["The person running it needs to fix a setting before it can read requests.", str(exc)],
        )

    return app


# Module-level app so the server starts with `uvicorn communitybridge.web.app:app`.
app = create_app()
