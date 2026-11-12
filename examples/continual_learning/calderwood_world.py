"""Calderwood Insurance: the one world all five notebooks share.

Calderwood Insurance is FICTIONAL. Every system, code, team, person and policy
here is invented, so no model knows any of it before it is told.

A fact has one spelling, here. A notebook builds its own small fixtures (the
claims on the desk that day, the requests a handler makes) and reads names,
codes and limits from this module, so the Grade 1 payout limit in notebook 01
is the same number notebook 03 later supersedes.

Nothing here imports MemoRizz. It is plain data and a few helpers.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

# --------------------------------------------------------------------- time
#: Calderwood's knowledge arrives in rounds. A later round supersedes an
#: earlier one; a value with no entry for a round carries over.
ROUNDS = {1: "2026-09", 2: "2027-01", 3: "2027-04"}
ROUND_NAMES = {1: "September 2026", 2: "January 2027", 3: "April 2027"}

# ------------------------------------------------------------------- people
#: name -> grade. A handler's payout limit is read from POLICIES, never copied.
HANDLERS = {
    "Priya Shah": "grade_2",
    "Tom Okafor": "grade_1",
    "Marcus Webb": "grade_1",
    "Ellie Burns": "grade_2",
    "Hannah Doyle": "grade_1",
    "Gwen": "senior",
    "Ruth": "senior",
}
GRADE_NAMES = {"grade_1": "Grade 1", "grade_2": "Grade 2", "senior": "Senior Handler"}
SENIOR_HANDLERS = ["Gwen", "Ruth"]
DESK_LEADS = ["Dev", "Callum"]
#: user ids the notebooks use for the people who talk to the assistant
USER_IDS = {
    "Priya Shah": "priya.shah",
    "Tom Okafor": "tom.okafor",
    "Marcus Webb": "marcus.webb",
    "Ellie Burns": "ellie.burns",
    "Hannah Doyle": "hannah.doyle",
    "Gwen": "gwen",
    "Ruth": "ruth",
}
#: Customers only ever appear on a claim. They are episodic detail: a skill,
#: a persona sentence or an entity record must never carry one.
CUSTOMERS = [
    "Mr D. Hollis", "Mrs P. Achebe", "Brightwater Bakeries Ltd", "Ms J. Kowalski",
    "Fenmoor Haulage", "Mr R. Iqbal", "Linden Dental Group", "Mrs S. O'Neill",
    "Mr T. Varga", "Oakham Lettings", "Ms C. Duarte", "Penrose Joinery",
]  # fmt: skip

# -------------------------------------------------------------------- teams
TEAMS = {
    "Counter Fraud Unit": {
        "aliases": ["CFU", "Counter Fraud", "the fraud team"],
        "lead": "Nadia Hussain",
        "does": "investigates suspected fraudulent claims; receives Amber Packs",
    },
    "Financial Crime": {
        "aliases": ["FinCrime"],
        "does": "receives RFN-110 sanctions referrals",
    },
    "Large Loss Unit": {"aliases": ["LLU"], "does": "handles any claim over £50,000"},
    "Care Team": {
        "aliases": [],
        "does": "calls customers with an RVF within 1 working day",
    },
    "Payments Control Review": {
        "aliases": ["PCR"],
        "does": "gives second approval on blocked payments",
    },
    "Settlement Review Team": {
        "aliases": ["SRT"],
        "does": "clears payments over £25,000 before Tally releases them",
        "from_round": 2,
    },
    "Motor Hub": {"aliases": [], "does": "takes motor total losses", "from_round": 3},
}

# ------------------------------------------------------------------ systems
SYSTEMS = {
    "Beacon": "claims management system; claim references are CW- plus six digits",
    "Atlas": "policy administration system; migrated Keystone policies carry a K- prefix",
    "Tally": "payments system; claim payments are released through Tally",
    "The Ringfence": "sanctions screening; every new payee is screened before any payment",
    "Drop": "customer upload portal for large files, retired in January 2027",
    "Compass": "customer upload portal that replaces Drop from January 2027",
}

#: Beacon status codes
STATUS_CODES = {
    "Q3": "awaiting documents from the customer",
    "Q7": "awaiting a surveyor or engineer report",
    "Q9": "referred to the CFU; internal only, the customer is told 'under review'",
    "Q12": "payment released",
}

ERRORS = {
    "BCN-217": "Beacon rejected an attachment over 25MB",
    "BCN-502": "the claim is locked by another handler in Beacon",
    "ATL-031": "Atlas can't find the policy; migrated Keystone policies need the K- prefix",
    "TLY-409": "Tally blocked a possible duplicate payment; it needs PCR approval",
    "TLY-512": "Tally is holding the payment for Settlement Review Team (SRT) clearance",
    "RFN-110": "the Ringfence found a potential sanctions match; the payment is frozen",
}

#: House vocabulary. Five of these collide with what the wider industry means.
GLOSSARY = {
    "TPA": "Total Payout Authority: the most a handler can pay on a claim without referring it up",
    "Amber Pack": "the fraud referral pack a handler sends to the CFU",
    "HSR": "Hold for Surveyor Review: the claim is paused until the surveyor's report is back",
    "RVF": "Rapid Vulnerability Flag: set in Beacon, routes the case to the Care Team",
    "IBNR": "Internal Broker Notification Request: tells a broker about a mid-term policy change",
    "CAT": "Claims Audit Trail: Beacon's log of every action on a claim",
    "LAE": "Large Account Escalation: an escalation for a PCL customer",
    "PML": "Policy Mid-term Lapse: cover lapsed because a premium was missed",
    "PCL": "Premier Commercial Lines: the top commercial customer tier",
    "GDR": "Goodwill Discretion Reserve: the budget goodwill payments come from",
    "FNOL": "first notification of loss",
    "Green Lane": "fast track for simple claims under £1,500 with no injury",
}
COLLISIONS = ["TPA", "IBNR", "CAT", "LAE", "PML"]

# ----------------------------------------------------------------- policies
#: value per round. Read with policy(); a later round supersedes an earlier one.
POLICIES: Dict[str, Any] = {
    "tpa": {
        "grade_1": {1: 2500, 2: 3000},
        "grade_2": {1: 10000},
        "senior": {1: 50000},
    },
    "amber_pack_days": {1: 2, 2: 3},
    "srt_threshold": {2: 25000},  # payments above this need an SRT clearance
    "upload_portal": {1: "Drop", 2: "Compass"},
    "attachment_limit_mb": {1: 25},
    "llu_threshold": {1: 50000},
    "motor_tl_days": {1: 10, 3: 7},
    "pcl_response_hours": {1: 2, 3: 1},
    "ibnr_hours": {1: 48},
    "fnol_hours": {1: 4},
}

CURRENCY = "GBP"

#: One sentence every assistant's instruction carries.
HOUSE_INSTRUCTION = (
    "You are Ada, the internal claims support assistant at Calderwood Insurance, a UK "
    "insurer. Amounts are in pounds sterling: write them with a £ sign."
)


def policy(name: str, key: Optional[str] = None, round: int = 1) -> Any:
    """The value of a policy in force in a round (None before it exists)."""
    values = POLICIES[name]
    if key is not None:
        values = values[key]
    in_force = [r for r in sorted(values) if r <= round]
    return values[in_force[-1]] if in_force else None


def grade_of(handler: str) -> str:
    return HANDLERS[handler]


def tpa(handler: str, round: int = 1) -> int:
    """A handler's Total Payout Authority in pounds, in a round."""
    return policy("tpa", grade_of(handler), round)


def pounds(amount: float) -> str:
    """£1,840.00 as people at Calderwood write it."""
    return f"£{amount:,.2f}".replace(".00", "")


_CLAIM_REF = re.compile(r"^CW-\d{6}$")
_KEYSTONE_REF = re.compile(r"^K-\d+$")


def is_claim_ref(value: Any) -> bool:
    """A Beacon claim reference: CW- followed by six digits."""
    return bool(_CLAIM_REF.match(str(value)))


def is_keystone_ref(value: Any) -> bool:
    return bool(_KEYSTONE_REF.match(str(value)))


class CalderwoodError(ValueError):
    """A system refusal, carrying its code: str(error) starts 'BCN-217: ...'."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        text = f"{code}: {ERRORS[code]}" if code in ERRORS else code
        super().__init__(f"{text}. {detail}".strip() if detail else text)


def money(**fields: Any) -> Dict[str, Any]:
    """A tool result that mentions money: every one says which currency."""
    return {**fields, "currency": CURRENCY}


def team_named(name: str) -> Optional[str]:
    """The team a name or alias refers to ('the fraud team' -> 'Counter Fraud Unit')."""
    wanted = str(name).casefold()
    for team, row in TEAMS.items():
        if wanted == team.casefold() or wanted in {
            a.casefold() for a in row["aliases"]
        }:
            return team
    return None
