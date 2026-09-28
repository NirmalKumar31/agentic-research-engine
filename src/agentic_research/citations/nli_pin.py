"""The pinned verifier identity, defined once.

Kept in its own module so both :mod:`agentic_research.config` and
:mod:`agentic_research.citations.nli` can import it without a cycle.

The revision is not decoration. The 0.98 support threshold was
calibrated against this exact commit, and a checkpoint that moves under
a fixed threshold is no longer the verifier that was measured. There is
deliberately no unpinned default: an earlier version defaulted to
"main", which would have silently served whatever the repository held
that day.
"""

from __future__ import annotations

NLI_DEFAULT_MODEL_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
NLI_DEFAULT_REVISION = "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"
