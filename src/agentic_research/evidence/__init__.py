from agentic_research.evidence.dedup import (
    Candidate,
    DedupStats,
    dedupe_by_content,
    dedupe_search_results,
)
from agentic_research.evidence.quality import (
    classify_source,
    domain_concentration,
    score_source,
)
from agentic_research.evidence.store import EvidencePackage, EvidenceStore, verify_quote

__all__ = [
    "Candidate",
    "DedupStats",
    "EvidencePackage",
    "EvidenceStore",
    "classify_source",
    "dedupe_by_content",
    "dedupe_search_results",
    "domain_concentration",
    "score_source",
    "verify_quote",
]
