import pytest
from agent.cascade import route_public_intent, needs_customer_workflow
from context.indexers.retrieve import legal_bm25_tokens


@pytest.mark.parametrize(('question', 'department'), [
    ('What is my baggage allowance for premium?', 'baggage'),
    ('When might extraordinary circumstances affect compensation?', 'disruptions'),
    ('What expenses can I claim for delayed baggage?', 'refunds_claims'),
    ('How do I redeem rewards points?', 'rewards_loyalty'),
    ('Can I upgrade Economy Flex directly to Business?', 'upgrades'),
    ('What is on the onboard menu?', 'cabin_onboard'),
    ('What are the check-in rules?', 'general_policies'),
])
def test_intent_router_selects_one_policy_department(question, department):
    assert route_public_intent(question) == department


def test_unsupported_and_multi_intent_requests_stop_before_retrieval():
    assert route_public_intent('Tell me a joke') is None
    assert route_public_intent('Can I upgrade and bring extra baggage?') is None
    assert needs_customer_workflow('How much is my upgrade offer?')


def test_legal_bm25_expands_extraordinary_circumstances_terms():
    tokens = legal_bm25_tokens("When might extraordinary circumstances affect compensation?")
    assert "extraordinary" in tokens
    assert "circumstances" in tokens
    assert "reasonable" in tokens
    assert "measures" in tokens
    assert "meteorological" in tokens


def test_legal_bm25_surfaces_extraordinary_circumstances_passage():
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document

    docs = [
        Document(page_content="Extraordinary circumstances can release the carrier from compensation if reasonable measures were taken; examples include meteorological conditions.", metadata={"chunk_id": "EU-LAW"}),
        Document(page_content="Compensation is paid for denied boarding and some cancellations.", metadata={"chunk_id": "GENERIC"}),
    ]
    retriever = BM25Retriever.from_documents(docs, preprocess_func=legal_bm25_tokens)
    retriever.k = 1
    assert retriever.invoke("When might extraordinary circumstances affect compensation?")[0].metadata["chunk_id"] == "EU-LAW"
