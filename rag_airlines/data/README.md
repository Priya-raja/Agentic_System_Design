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
