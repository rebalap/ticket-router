"""Intent taxonomy, team routing table, and misroute severities.

Bitext is an e-commerce-flavoured dataset, so its 11 coarse categories don't map
cleanly onto SaaS support queues (e.g. its CANCEL category only covers "check
cancellation fee", while real cancellations live under ORDER and ACCOUNT). We
regroup its 27 fine-grained intents into 8 SaaS-style intents instead.
"""

from __future__ import annotations

# Bitext fine-grained intent -> our intent label.
BITEXT_TO_INTENT: dict[str, str] = {
    # billing
    "check_invoice": "billing",
    "get_invoice": "billing",
    "check_payment_methods": "billing",
    "payment_issue": "billing",
    # refund
    "check_refund_policy": "refund",
    "get_refund": "refund",
    "track_refund": "refund",
    # cancellation
    "cancel_order": "cancellation",
    "delete_account": "cancellation",
    "check_cancellation_fee": "cancellation",
    # account management
    "create_account": "account",
    "edit_account": "account",
    "switch_account": "account",
    "recover_password": "account",
    # technical issue (closest Bitext proxy for "bug")
    "registration_problems": "technical_issue",
    # orders / provisioning / delivery
    "place_order": "order_fulfillment",
    "change_order": "order_fulfillment",
    "track_order": "order_fulfillment",
    "delivery_options": "order_fulfillment",
    "delivery_period": "order_fulfillment",
    "change_shipping_address": "order_fulfillment",
    "set_up_shipping_address": "order_fulfillment",
    # complaint
    "complaint": "complaint",
    # general / low-stakes
    "contact_customer_service": "general",
    "contact_human_agent": "general",
    "review": "general",
    "newsletter_subscription": "general",
}

INTENTS: list[str] = [
    "billing",
    "refund",
    "cancellation",
    "account",
    "technical_issue",
    "order_fulfillment",
    "complaint",
    "general",
]
INTENT_TO_ID = {name: i for i, name in enumerate(INTENTS)}

# Which team owns each intent.
INTENT_TO_TEAM: dict[str, str] = {
    "billing": "billing",
    "refund": "billing",
    "cancellation": "retention",
    "complaint": "retention",
    "account": "tech_support",
    "technical_issue": "tech_support",
    "order_fulfillment": "operations",
    "general": "frontline",
}
HUMAN_TRIAGE_TEAM = "human_triage"

# Cost of sending a ticket with this TRUE intent to the wrong team, in units of
# "one low-stakes misroute". An angry cancellation landing in billing is the
# canonical expensive mistake; a newsletter question landing in ops is cheap.
MISROUTE_SEVERITY: dict[str, float] = {
    "cancellation": 5.0,
    "complaint": 4.0,
    "refund": 3.0,
    "billing": 3.0,
    "technical_issue": 3.0,
    "account": 2.0,
    "order_fulfillment": 1.0,
    "general": 1.0,
}

# Natural-language descriptions used as zero-shot hypotheses.
ZERO_SHOT_DESCRIPTIONS: dict[str, str] = {
    "billing": "billing, invoices, charges, or payment methods",
    "refund": "getting a refund or reimbursement",
    "cancellation": "cancelling an order or subscription, or deleting an account",
    "account": "managing an account, profile, or password",
    "technical_issue": "a technical error or problem signing up",
    "order_fulfillment": "placing, changing, tracking, or delivering an order",
    "complaint": "a complaint about bad service",
    "general": "contacting customer service, leaving a review, or the newsletter",
}
ZERO_SHOT_TEMPLATE = "This customer support ticket is about {}."

assert set(INTENTS) == set(BITEXT_TO_INTENT.values()) == set(INTENT_TO_TEAM)
assert set(INTENTS) == set(MISROUTE_SEVERITY) == set(ZERO_SHOT_DESCRIPTIONS)
