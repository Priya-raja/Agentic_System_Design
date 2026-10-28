"""Deterministic refund simulation for synthetic CRM claims; never sends money.

Customer IDs supplied to this CLI are DEMO identities, not authentication.
A production adapter must provide verified identity, approval, and payment services.
"""

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from config import ROOT_DIR


class RefundError(ValueError):
    pass


def _records(path: Path, key: str) -> dict:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("synthetic") is not True:
            raise RefundError("This agent accepts synthetic demo records only.")
        identity = record[key]
        if identity in result:
            raise RefundError("Duplicate record identifier; review required.")
        result[identity] = record
    return result


class RefundAgent:
    """Check reviewed claims and record an idempotent simulated refund."""

    def __init__(
        self,
        *,
        demo: bool,
        customer_id: str,
        crm_dir: Path = ROOT_DIR / "data" / "crm",
        ledger: Path = ROOT_DIR / ".runtime" / "refunds.sqlite3",
    ):
        if demo is not True or not customer_id:
            raise RefundError("Explicit demo mode and a synthetic customer identity are required.")
        self.customer_id = customer_id
        self.crm_dir = Path(crm_dir)
        self.ledger = Path(ledger)

    def _claim(self, claim_id: str):
        claims = _records(self.crm_dir / "refund_claims.jsonl", "claim_id")
        cases = _records(self.crm_dir / "baggage_cases.jsonl", "case_id")
        bookings = _records(self.crm_dir / "bookings.jsonl", "pnr")
        claim = claims.get(claim_id)
        # Return the same message for missing and another customer's claim.
        if not claim or claim.get("customer_id") != self.customer_id:
            raise RefundError("Claim not found for this demo customer.")
        case = cases.get(claim.get("case_id"))
        booking = bookings.get(claim.get("pnr"))
        if (
            not case
            or not booking
            or case.get("customer_id") != self.customer_id
            or booking.get("customer_id") != self.customer_id
            or case.get("pnr") != claim.get("pnr")
        ):
            raise RefundError("Claim, case and booking ownership do not match; review required.")
        return claim, case, booking

    def list_claims(self) -> list[dict]:
        claims = _records(self.crm_dir / "refund_claims.jsonl", "claim_id")
        return [
            self.check(claim_id)
            for claim_id, claim in claims.items()
            if claim.get("customer_id") == self.customer_id
        ]

    def _eligibility(self, claim, case):
        if claim.get("status") != "approved":
            return None, "Claim needs review or has been rejected; no refund can proceed."
        if not claim.get("approval_reference") or claim.get("evidence_verified") is not True:
            return None, "A recorded approval and verified evidence are required."
        if claim.get("payment_destination") != "original_payment_method":
            return None, "Only the original payment method is supported."
        if case.get("case_type") != "delayed_baggage":
            return None, "This demo supports approved delayed-baggage expense claims only."
        try:
            amount = Decimal(str(claim.get("approved_amount_aed")))
            receipts = [Decimal(str(value)) for value in case.get("expense_receipts_aed", [])]
            if (
                not amount.is_finite()
                or amount <= 0
                or amount != amount.quantize(Decimal(".01"))
                or not receipts
                or any(not value.is_finite() or value < 0 for value in receipts)
                or amount > sum(receipts)
            ):
                raise ValueError()
        except (InvalidOperation, ValueError, TypeError):
            return None, "The approved amount or receipt evidence is invalid; review required."
        return amount, "Reviewed claim is ready for a simulated refund. No money will move."

    def check(self, claim_id: str) -> dict:
        claim, case, _ = self._claim(claim_id)
        amount, reason = self._eligibility(claim, case)
        previous = None
        if self.ledger.exists():
            with sqlite3.connect(self.ledger) as db:
                previous = db.execute(
                    "SELECT receipt_id FROM refunds WHERE claim_id = ? AND customer_id = ?",
                    (claim_id, self.customer_id),
                ).fetchone()
        return {
            "claim_id": claim_id,
            "claim_status": claim["status"],
            "refund_status": "simulated_processed" if previous else "ready" if amount else "review_required",
            "approved_amount_aed": str(amount) if amount else None,
            "can_proceed": amount is not None and not previous,
            "receipt_id": previous[0] if previous else None,
            "reason": reason,
            "simulation": True,
        }

    def review_fingerprint(self, claim_id: str) -> str:
        """Bind human approval to the claim, receipts and booking reviewed."""
        return hashlib.sha256(json.dumps(self._claim(claim_id), sort_keys=True).encode()).hexdigest()

    def proceed(
        self, claim_id: str, *, confirmed: bool = False, expected_fingerprint: str | None = None
    ) -> dict:
        if confirmed is not True:
            raise RefundError("Explicit confirmation is required before simulating a refund.")
        records = self._claim(claim_id)
        if (
            expected_fingerprint is not None
            and hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
            != expected_fingerprint
        ):
            raise RefundError("Claim evidence changed after review. Start a new review.")
        claim, case, _ = records
        amount, reason = self._eligibility(claim, case)
        if amount is None:
            raise RefundError(reason)
        fingerprint = hashlib.sha256(json.dumps(claim, sort_keys=True).encode()).hexdigest()
        receipt = "SIM-" + hashlib.sha256(claim_id.encode()).hexdigest()[:16]
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.ledger, isolation_level=None) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS refunds (
                claim_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL,
                amount_aed TEXT NOT NULL, receipt_id TEXT NOT NULL,
                fingerprint TEXT NOT NULL, processed_at TEXT NOT NULL)""")
            db.execute("BEGIN IMMEDIATE")
            try:
                old = db.execute(
                    "SELECT customer_id, amount_aed, receipt_id, fingerprint FROM refunds WHERE claim_id = ?",
                    (claim_id,),
                ).fetchone()
                if old and (old[0] != self.customer_id or old[1] != str(amount) or old[3] != fingerprint):
                    raise RefundError("Previously processed claim has changed; manual review required.")
                if not old:
                    db.execute(
                        "INSERT INTO refunds VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            claim_id,
                            self.customer_id,
                            str(amount),
                            receipt,
                            fingerprint,
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                db.commit()
            except Exception:
                db.rollback()
                raise
        return {
            "claim_id": claim_id,
            "refund_status": "simulated_processed",
            "amount_aed": str(amount),
            "receipt_id": receipt,
            "already_processed": old is not None,
            "simulation": True,
            "message": "Simulation recorded. No payment was sent.",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", required=True)
    parser.add_argument("--customer-id", required=True, help="Synthetic demo identity, not authentication")
    parser.add_argument("--ledger", type=Path, default=ROOT_DIR / ".runtime" / "refunds.sqlite3")
    parser.add_argument("action", choices=["list", "check", "proceed"])
    parser.add_argument("claim_id", nargs="?")
    parser.add_argument("--confirm", action="store_true", help="Confirm a simulated refund only")
    args = parser.parse_args()
    if args.action != "list" and not args.claim_id:
        parser.error("claim_id is required for check and proceed")
    agent = RefundAgent(demo=args.demo, customer_id=args.customer_id, ledger=args.ledger)
    try:
        result = (
            agent.list_claims()
            if args.action == "list"
            else agent.check(args.claim_id)
            if args.action == "check"
            else agent.proceed(args.claim_id, confirmed=args.confirm)
        )
        print(json.dumps(result, indent=2))
    except RefundError as error:
        parser.exit(1, f"Refund not processed: {error}\n")


if __name__ == "__main__":
    main()
