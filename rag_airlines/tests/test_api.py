import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from backend.app import create_app
from agent.session import ChatSession


def client(limit="1000/minute", demo="CUST-1001"):
    tmp = tempfile.TemporaryDirectory()
    session = ChatSession()
    session.ask = lambda question: {
        "answer": "Verified fixture",
        "sources": [],
        "cache": "miss",
        "kind": "answer",
    }

    async def stream(question):
        yield {"event": "safety", "action": "OK"}
        yield {"event": "verify", "stage": "final", "passed": True}
        yield {"event": "token", "text": "Verified fixture"}
        yield {
            "event": "result",
            "result": {"answer": "Verified fixture", "sources": [], "cache": "miss", "kind": "answer"},
        }

    session.astream = stream
    c = TestClient(
        create_app(
            demo_customer=demo, ledger=Path(tmp.name) / "ledger.sqlite3", session=session, rate_limit=limit
        )
    )
    return c, tmp


def test_api_scoping_csrf_confirmation_receipts_and_stream():
    c, tmp = client()
    try:
        overview = c.get("/api/overview").json()
        headers = {"origin": "http://127.0.0.1:3000", "x-aeronova-token": overview["csrf_token"]}
        assert c.get("/api/bookings/AN0002K/pdf").status_code == 404
        pdf = c.get("/api/bookings/AN0001K/pdf")
        assert pdf.content.startswith(b"%PDF") and pdf.headers["cache-control"] == "no-store"
        assert c.post("/api/chat", json={"question": "hello"}).status_code == 403
        assert c.post("/api/chat", json={"question": " "}, headers=headers).status_code == 422
        assert c.post("/api/chat/stream", json={"question": "test"}, headers=headers).text.endswith(
            'data: {"event":"done"}\n\n'
        )
        assert c.get("/api/refunds/REF-DEMO-001/receipt").status_code == 409
        url = "/api/refunds/REF-DEMO-001/proceed"
        for value in [False, 1, "true"]:
            assert c.post(url, json={"confirmed": value}, headers=headers).status_code == 422
        one = c.post(url, json={"confirmed": True}, headers=headers).json()
        two = c.post(url, json={"confirmed": True}, headers=headers).json()
        assert one["receipt_id"] == two["receipt_id"] and two["already_processed"]
        assert c.get("/api/refunds/REF-DEMO-001/receipt").content.startswith(b"%PDF")
        assert c.get("/api/flights?flight_number=AN201&departure_date=2026-10-02").json()["live"] is False
        assert c.get("/api/code/../../.env").status_code in [404, 307]
        assert c.get("/api/evals").json()["summary"]["total_cases"] >= 1
    finally:
        tmp.cleanup()


def test_public_mode_hides_records_and_rate_limit_is_enforced():
    c, tmp = client(limit="10/minute", demo=None)
    try:
        assert c.get("/api/overview").json()["bookings"] == []
        assert c.get("/api/bookings/AN0001K/pdf").status_code == 403
        statuses = [c.get("/api/health").status_code for _ in range(12)]
        assert 429 in statuses
    finally:
        tmp.cleanup()


def test_graph_review_api_pauses_reloads_and_rejects_without_receipt():
    c, tmp = client()
    try:
        token = c.get('/api/overview').json()['csrf_token']
        headers = {'origin':'http://127.0.0.1:3000', 'x-aeronova-token':token}
        url = '/api/refunds/REF-DEMO-001/review'
        assert c.post(url).status_code == 403
        review = c.post(url, headers=headers).json()
        assert review['status'] == 'pending'
        workflow_url = '/api/refund-reviews/'+review['workflow_id']
        assert c.get(workflow_url).json()['status'] == 'pending'
        assert c.get('/api/refunds/REF-DEMO-001/receipt').status_code == 409
        for value in [1, 'true', None]:
            assert c.post(workflow_url+'/decision', json={'approved':value}, headers=headers).status_code == 422
        assert c.post(workflow_url+'/decision', json={'approved':True}).status_code == 403
        assert c.post(workflow_url+'/decision', json={'approved':False}, headers=headers).json()['status'] == 'rejected'
        assert c.get('/api/refunds/REF-DEMO-001/receipt').status_code == 409
        assert c.get('/api/policies/route-menus.pdf').content.startswith(b'%PDF')
    finally:
        tmp.cleanup()
