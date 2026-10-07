# AeroNova RAG

## Code structure

```text
agent/
  factory.py                 # create_agent(), grounded answers, validation, CLI
  jev_classifier.py          # Department and request-scope routing
  prompt_manager.py          # Versioned prompt loading
context/
  ingest/
    load_documents.py        # Corpus loading and OCR
    prepare_documents.py     # Public document selection and chunking
  indexers/
    build_index.py           # Qdrant index construction
    retrieve.py              # Hybrid retrieval and freshness checks
  memory/
    answer_cache.py           # Redis answer cache
config.py                    # Shared configuration and project paths
observability/               # Tracing
prompts/                     # Versioned prompts
```

From the project root, use the virtual environment with dependencies from
`requirements.txt` and configure API keys in `.env`. Start Qdrant and Redis
with `docker compose up -d`, then run:

```sh
python -m context.indexers.build_index
python -m agent.factory
```

For retrieval alone, run `python -m context.indexers.retrieve`.
The original root scripts remain as compatibility entry points, so
`python build_index.py` and `python answer.py` still work.

For programmatic initialization, import `create_agent` from `agent.factory`.
It returns the loaded prompt, prepared chunks, and vector store after checking
that the index matches the current corpus. It does not rebuild the index.
Corpus data, prompts, and `index_state.json` remain at the project root.

## Prompt caching

The agent keeps its loaded system prompt in memory. Model requests also use a
stable OpenAI `prompt_cache_key` derived from the prompt content hash, and print
the number of cached input tokens returned by the API. OpenAI manages this cache;
hits depend on matching prefixes, eligible prefix length, and cache availability.
The full system prompt is sent on every model call. Changing the prompt changes
the routing key automatically. This is separate from the Redis answer cache,
which skips generation entirely for a cached answer.

## Semantic answer caching

`context/memory/semantic_answer_cache.py` adds a second cache after an exact Redis
miss and local metadata topic selection. It reuses only validated, answerable public
policy answers, preserving the response and source chunks. Lookup requires the
same topic, policy year, prompt version and hash, corpus fingerprint, pipeline
version, embedding model, and calendar day. Meaningful question terms must also
match in order, with a small set of equivalent words allowed; similarity alone
does not establish equivalence. Personal, conditional, comparative, negated, and
relative-date questions bypass semantic caching.

Redis stores up to 100 entries per scope. Entries expire after 24 hours. The
cosine similarity threshold is 0.97, configured in `config.py`. Lookups embed
only when a candidate passes the metadata and wording guards; storing eligible
answers also calls the embedding API. Cache errors fall back to normal retrieval
and generation. A semantic hit populates the exact cache for the new question.
Existing Qdrant vectors do not need rebuilding.

Run offline cache tests with `python -m unittest discover -s tests -v`.

## Department and request-scope routing

JEV classifies department, request scope, and model tier in one request. The
answer model remains `ANSWER_MODEL`; model tier is recorded for future use.
Public-information requests in baggage, disruptions, refunds_claims,
cabin_onboard, general_policies, and upgrades use RAG with department metadata filters.
General support, inactive departments, and uncertain decisions receive a support
handoff message. Customer-specific requests require a secure workflow and do not
query public RAG or use semantic caching. No booking, claims, CRM, authentication,
or live operations integration has been implemented; no actual handoff is sent.

Exact-cache hits bypass routing because this cache stores only previously routed
public RAG answers. Semantic caching runs after routing and includes the department
in its scope (the internal cache field is named `topic` for compatibility).

Department metadata changes the index schema. Rebuild existing indexes with:

```sh
python -m context.indexers.build_index
```

The freshness check rejects indexes without `departments-v2` metadata. The cache
pipeline version has also changed, invalidating previous cache entries.

# AeroNova upgrade corpus extension

This fictional corpus adds cabin-upgrade rules for AeroNova Airways. It is intended only for RAG development and evaluation.

## Core rule

- Economy Saver and Economy Flex cannot upgrade directly to Business.
- Premium Economy Standard may upgrade to Business when eligible inventory is available.
- An originally Economy ticket cannot reach Business through sequential upgrades on the same journey.

## Files

- `policies/POL-UPG-2026.pdf`: authoritative upgrade policy.
- `policies/TAB-UPG-ELIGIBILITY-2026.pdf`: compact eligibility table.
- `web/WEB-007.md`: customer-facing secondary explanation.
- `crm/upgrade_offers.jsonl`: synthetic private booking and offer records. Do not index this in the public knowledge collection.
- `evals/upgrade_golden.jsonl`: starter retrieval and answer-evaluation cases.

## Suggested public metadata

Use `topic=upgrades`, `policy_year=2026`, `valid_from=2026-01-01`, and `valid_to=2026-12-31` for the three public documents. Treat both PDFs as `authority=primary` and `WEB-007.md` as `authority=secondary`.

Keep `crm/upgrade_offers.jsonl` outside the public Qdrant collection. Access it only through an authenticated customer workflow.

## Upgrade policies

Public upgrade-policy requests route to `public_rag`, including first-person
questions such as “Can I upgrade Economy Flex directly to Business?” Only
`POL-UPG-2026`, `TAB-UPG-ELIGIBILITY-2026`, and `WEB-007` receive both the
`upgrades` topic and department before chunking. Requests requiring booking
records, a personalized price, or live inventory route to `upgrade_workflow`
with authentication required. The secure workflow is a destination label and
handoff message; no live booking/inventory integration is connected.

Department and request-scope confidence use separate thresholds. Scope requires
0.65 confidence (`JEV_SCOPE_MIN_CONFIDENCE`); model complexity still uses 0.70.
Uncertain scope preserves the department and requires review, rather than
relabeling a confidently identified department as general support.

## No-context caching

Empty retrieval results for public RAG questions are cached in Redis for
`NO_CONTEXT_CACHE_TTL_SECONDS = 300`. Repeating the same normalized question
returns the original no-policy message without routing, retrieval, or generation.
The usual prompt, corpus, and pipeline versions remain part of the key. These
results never enter the semantic cache. Normal answers retain their 24-hour TTL.
