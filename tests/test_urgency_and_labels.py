from ticket_router.data import clean_text
from ticket_router.labels import BITEXT_TO_INTENT, INTENT_TO_TEAM, INTENTS
from ticket_router.urgency import score_urgency


def test_all_27_bitext_intents_mapped():
    assert len(BITEXT_TO_INTENT) == 27
    assert set(BITEXT_TO_INTENT.values()) == set(INTENTS)
    assert set(INTENT_TO_TEAM) == set(INTENTS)


def test_urgency_levels():
    assert score_urgency("how do I receive your newsletter?", "general") == "low"
    assert score_urgency("where is my invoice", "billing") == "normal"
    assert score_urgency("cancel my plan", "cancellation") == "normal"
    assert score_urgency("cancel my damn plan right now", "cancellation") == "high"
    assert score_urgency("I was charged twice, fix this ASAP", "billing") == "high"
    assert score_urgency("I think my account was hacked", "account") == "high"  # risk alone
    assert score_urgency("I want my bloody refund", "refund") == "high"  # anger + money
    assert score_urgency("help with a fucking address change", "order_fulfillment") == "normal"


def test_clean_text_fills_placeholders():
    assert clean_text("cancel  order {{Order Number}} pls") == "cancel order order number pls"
