from typing import List, Dict, Set

CHUNK_SIZE: int = 1024

GLINER_MODEL: str = "urchade/gliner_small-v2.1"

LANCEDB_URI = "data/lancedb"

NER_LABELS: List[str] = [
    "Behavior",
    "City",
    "Company",
    "Condition",
    "Conference",
    "Country",
    "Food",
    "Food Additive",
    "Hospital",
    "Organ",
    "Organization",
    "People Group",
    "Person",
    "Publication",
    "Research",
    "Science",
    "University",
]

RE_LABELS: dict = {
    "glirel_labels": {
        "co_founder": {"allowed_head": ["PERSON"], "allowed_tail": ["ORG"]},
        "country_of_origin": {
            "allowed_head": ["PERSON", "ORG"],
            "allowed_tail": ["LOC", "GPE"],
        },
        "no_relation": {},
        "parent": {"allowed_head": ["PERSON"], "allowed_tail": ["PERSON"]},
        "followed_by": {
            "allowed_head": ["PERSON", "ORG"],
            "allowed_tail": ["PERSON", "ORG"],
        },
        "spouse": {"allowed_head": ["PERSON"], "allowed_tail": ["PERSON"]},
        "child": {"allowed_head": ["PERSON"], "allowed_tail": ["PERSON"]},
        "founder": {"allowed_head": ["PERSON"], "allowed_tail": ["ORG"]},
        "headquartered_in": {
            "allowed_head": ["ORG"],
            "allowed_tail": ["LOC", "GPE", "FAC"],
        },
        "acquired_by": {"allowed_head": ["ORG"], "allowed_tail": ["ORG", "PERSON"]},
        "subsidiary_of": {"allowed_head": ["ORG"], "allowed_tail": ["ORG", "PERSON"]},
    }
}

SCRAPE_HEADERS: Dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
}

SPACY_MODEL: str = "en_core_web_md"

STOP_WORDS: Set[str] = set(
    [
        "PRON.it",
        "PRON.that",
        "PRON.they",
        "PRON.those",
        "PRON.we",
        "PRON.which",
        "PRON.who",
    ]
)

TR_ALPHA: float = 0.85
TR_LOOKBACK: int = 3
