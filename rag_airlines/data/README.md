# AeroNova Airways RAG corpus v1

This is fictional test material. AeroNova Airways, its fares, prices, customers, booking references, and claim rules are invented. Never present them as real travel advice or airline terms.

## Contents
- `policies/`: five searchable PDFs, including superseded and active baggage rules.
- `policies/` also includes a two-page pictured seat-category and menu guide with four JPEG image assets.
- `web/`: six Markdown help/blog articles, including an intentionally stale blog.
- `crm/`: synthetic customer, booking, claim and disruption JSONL records.
- `voice/`: four timestamped scripted call transcripts. These contain no recorded audio.
- `evaluation/`: twelve gold seed questions with source section identifiers.

## Retrieval behavior
1. Parse PDFs by section, retaining policy ID, page, heading, version and effective dates. Keep conditions with the associated rule.
2. Index public policy and web content with hybrid lexical/vector retrieval; rerank the eligible results.
3. Set an `as_of` date and filter out superseded policies for current questions. Let historical questions select the policy effective on the requested travel date.
4. Treat policy as higher authority than blogs. A booking-specific printed allowance may override the generic policy.
5. Fetch CRM rows through authenticated, customer-scoped structured queries after verifying identity; never put all customer rows into the public vector index.
6. Store `doc_id`, `section`, `page`, `effective_from`, `effective_to`, `status`, `source_type` and `authority` on every chunk.
7. For RAGAS, capture `user_input`, `retrieved_contexts`, `response` and `reference`. Use source IDs for deterministic retrieval and policy-version checks in addition to model-judged metrics.

## Voice limitation
The voice files are scripted transcripts, not WAV audio. Record actors with consent or generate synthetic speech for speech-to-text tests, and retain the transcript as ground truth.

## Provenance
Entirely created for this project. No real airline text or personal customer data copied.

## Added visual and tabular sources
- `policies/IMG-BAG-UPDATE-2026.png`: policy notice with important text embedded in an image. OCR ground truth: Flex 25 kg; eligible Gold +5 kg; 32 kg per single bag; ticket allowance takes precedence.
- `policies/TAB-BAG-ALLOWANCE-2026.pdf`: allowance matrix and explanatory notes.
- `policies/TAB-BAG-ALLOWANCE-2026.csv`: same table in a machine-readable format.

## Rewards and refund simulation additions

- `policies/POL-REWARDS-2026.md`: fictional earning, expiry, status and account rules.
- `policies/POL-REDEEM-2026.md`: fictional redemption steps, inventory, points plus cash and cancellation rules.
- `web/WEB-008.md`: secondary rewards/redemption guide.
- `crm/refund_claims.jsonl`: three synthetic review outcomes tied to existing cases and bookings. REF-DEMO-001 is explicitly approved for AED 154; the other samples are pending and rejected. These are invented review fixtures, not inferred approval from policy limits.

All additions are fictional. CRM claims remain private to the local demo workflow.

## EU air passenger rights source

- `policies/POL-EU261-2004.md` is the English text of Regulation (EC) No 261/2004
  downloaded from the `Hamses/EU_Regulation_261_2004` Hugging Face dataset. The
  dataset identifies EUR-Lex as the source and declares CC0-1.0. Chunks are
  indexed as primary `disruptions` evidence. This is the original 2004 act text,
  not a consolidated edition; its metadata limits it to 2027-10-22 because
  Regulation (EU) 2026/2202 applies from 2027-10-23.

## Directional menu supplement

`policies/CAT-CABIN-ROUTES-2026.pdf` contains six fictional sample menus:
DXB-LHR, LHR-DXB, DXB-BOM, BOM-DXB, DXB-SIN and SIN-DXB. Each page is one
direction, with meals, vegetarian options, light bites, desserts and drinks.
Prices and availability are illustrative, not real JetBlue policies. The DXB-LHR
page preserves the existing menu choices and adds a dessert. Regenerate with
`python scripts/build_route_menus.py`, then rebuild the public index.
Page-level route metadata is inherited by every chunk; menu queries with a
recognized route filter both dense and BM25 candidates by direction.
