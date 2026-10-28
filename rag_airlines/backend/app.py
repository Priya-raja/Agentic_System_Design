"""Local demo API with CSRF, rate limiting, scoped records and verified SSE."""

import json
import os
import secrets
from datetime import date
from functools import lru_cache
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, StrictBool, field_validator
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from starlette.middleware.trustedhost import TrustedHostMiddleware
from agent.refund_agent import RefundAgent, RefundError
from agent.refund_graph import RefundWorkflow
from agent.streaming import sse_stream
from backend.documents import pdf_document
from backend.evaluations import dashboard
from config import ROOT_DIR
from observability.logging import configure_logging, enable_optional_tracing


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("Question required")
        return value


class ReviewDecision(BaseModel):
    approved: StrictBool


class Confirmation(BaseModel):
    confirmed: StrictBool

    @field_validator("confirmed")
    @classmethod
    def positive(cls, value):
        if value is not True:
            raise ValueError("Explicit confirmation required")
        return value


def create_app(*, demo_customer=None, ledger=None, session=None, rate_limit="10/minute"):
    configure_logging()
    enable_optional_tracing()
    app = FastAPI(title="AeroNova bounded agent")
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1,testserver").split(","),
    )
    limiter = Limiter(key_func=get_remote_address, application_limits=[rate_limit])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    token = secrets.token_urlsafe(32)
    refunds = (
        RefundAgent(
            demo=True, customer_id=demo_customer, ledger=ledger or ROOT_DIR / ".runtime/refunds.sqlite3"
        )
        if demo_customer
        else None
    )
    refund_workflow = RefundWorkflow(refunds) if refunds else None
    origin = os.getenv("PUBLIC_ORIGIN", "http://127.0.0.1:3000")

    @lru_cache(maxsize=1)
    def chat_session():
        if session is not None:
            return session
        from agent.session import ChatSession

        return ChatSession.build(refund_agent=refunds)

    def verify(request):
        if request.headers.get("origin") not in {
            origin,
            "http://localhost:3000",
        } or not secrets.compare_digest(request.headers.get("x-aeronova-token", ""), token):
            raise HTTPException(403, "Invalid request origin or token. Refresh the page.")

    def records(filename):
        if not refunds:
            raise HTTPException(403, "Enable an explicitly bound demo customer to access fictional records.")
        rows = [
            json.loads(line) for line in (refunds.crm_dir / filename).read_text().splitlines() if line.strip()
        ]
        if any(row.get("synthetic") is not True for row in rows):
            raise HTTPException(503, "Non-synthetic record rejected")
        return rows

    def booking(pnr):
        item = next(
            (
                row
                for row in records("bookings.jsonl")
                if row["pnr"] == pnr and row["customer_id"] == refunds.customer_id
            ),
            None,
        )
        if not item:
            raise HTTPException(404, "Booking not found for this demo customer")
        return item

    @app.exception_handler(RequestValidationError)
    async def invalid(request, error):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=422, content={"detail": "Invalid request. Check required fields and confirmation."}
        )

    @app.middleware("http")
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/api/health")
    def health():
        return {"status": "ok", "demo": refunds is not None}

    @app.get("/api/overview")
    def overview():
        return {
            "csrf_token": token,
            "demo": refunds is not None,
            "bookings": [
                row for row in records("bookings.jsonl") if row["customer_id"] == refunds.customer_id
            ]
            if refunds
            else [],
            "claims": refunds.list_claims() if refunds else [],
            "notice": "Fictional demo identity, not production authentication.",
        }

    @app.post("/api/chat")
    def chat(body: ChatRequest, request: Request):
        verify(request)
        return chat_session().ask(body.question)

    @app.post("/api/chat/stream")
    def stream(body: ChatRequest, request: Request):
        verify(request)
        return sse_stream(chat_session(), body.question)

    @app.get("/api/policies/route-menus.pdf")
    def route_menu_pdf():
        from fastapi.responses import FileResponse

        return FileResponse(
            ROOT_DIR / "data/policies/CAT-CABIN-ROUTES-2026.pdf",
            media_type="application/pdf",
            filename="aeronova-route-menus-2026.pdf",
        )

    @app.get("/api/evals")
    def evals():
        return dashboard()

    @app.get("/api/flights")
    def flights(flight_number: str, departure_date: date):
        rows = json.loads((ROOT_DIR / "data/operations/flight_status.json").read_text())
        item = next(
            (
                r
                for r in rows
                if r["flight_number"] == flight_number.upper()
                and r["departure_date"] == departure_date.isoformat()
            ),
            None,
        )
        if not item:
            raise HTTPException(404, "No archived demo snapshot for this flight and date.")
        return {**item, "live": False, "notice": "Archived fictional snapshot; not live operations."}

    @app.get("/api/bookings/{pnr}/pdf")
    def ticket(pnr: str):
        item = booking(pnr)
        fields = [
            ("Booking", item["pnr"]),
            ("Flight", item["flight_number"]),
            ("Route", item["origin"] + " to " + item["destination"]),
            ("Date", item["departure_date"]),
            ("Cabin", item["cabin"]),
            ("Fare", item["fare_family"]),
            ("Status", item["status"]),
        ]
        return Response(
            pdf_document("Booking confirmation", fields),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="booking-{pnr}.pdf"'},
        )

    @app.post("/api/refunds/{claim_id}/review")
    def start_review(claim_id: str, request: Request):
        verify(request)
        if not refund_workflow:
            raise HTTPException(403, "No bound demo customer")
        try:
            return refund_workflow.start(claim_id)
        except RefundError as error:
            raise HTTPException(409, str(error))

    @app.get("/api/refund-reviews/{workflow_id}")
    def get_review(workflow_id: str):
        if not refund_workflow:
            raise HTTPException(403, "No bound demo customer")
        try:
            return refund_workflow.get(workflow_id)
        except RefundError as error:
            raise HTTPException(404, str(error))

    @app.post("/api/refund-reviews/{workflow_id}/decision")
    def decide_review(workflow_id: str, body: ReviewDecision, request: Request):
        verify(request)
        if not refund_workflow:
            raise HTTPException(403, "No bound demo customer")
        try:
            return refund_workflow.decide(workflow_id, body.approved)
        except RefundError as error:
            raise HTTPException(409, str(error))

    @app.post("/api/refunds/{claim_id}/proceed")
    def proceed(claim_id: str, body: Confirmation, request: Request):
        verify(request)
        if not refunds:
            raise HTTPException(403, "No bound demo customer")
        try:
            # Compatibility endpoint: explicit human confirmation still traverses
            # the persisted graph. The UI uses separate review and decision calls.
            review = refund_workflow.start(claim_id)
            if review["status"] == "completed":
                return review["result"]
            return refund_workflow.decide(review["workflow_id"], body.confirmed)["result"]
        except RefundError as error:
            raise HTTPException(409, str(error))

    @app.get("/api/refunds/{claim_id}/receipt")
    def receipt(claim_id: str):
        if not refunds:
            raise HTTPException(403, "No bound demo customer")
        try:
            item = refunds.check(claim_id)
        except RefundError:
            raise HTTPException(404, "Receipt unavailable")
        if item["refund_status"] != "simulated_processed":
            raise HTTPException(409, "No simulated refund has been processed")
        return Response(
            pdf_document(
                "Refund simulation receipt",
                [
                    ("Claim", item["claim_id"]),
                    ("Receipt", item["receipt_id"]),
                    ("Amount", "AED " + item["approved_amount_aed"]),
                    ("Status", "Simulation recorded - no money transferred"),
                ],
            ),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="receipt-{claim_id}.pdf"'},
        )

    @app.get("/api/code/{name}")
    def code(name: str):
        files = {
            key: ROOT_DIR / "agent" / f"{key}.py"
            for key in ["graph", "cascade", "route_types", "refund_graph", "session", "executor", "verifier", "factory"]
        }
        if name not in files:
            raise HTTPException(404, "Unknown source")
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(files[name].read_text())

    return app


app = create_app(
    demo_customer=os.getenv("AERONOVA_DEMO_CUSTOMER"), rate_limit=os.getenv("API_RATE_LIMIT", "10/minute")
)
