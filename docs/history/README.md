# History

Preserved historical evidence, not current documentation.

Every file under this directory describes the project *as it was at
the time it was written*: deployed commit SHAs, branch names, Render
service states, and other "the deployed service runs X" or "this
branch is Y" statements are snapshots, not live facts. They are kept
unedited on purpose -- a record changed to match today's state would no
longer be a record of what was actually checked and when. For anything
you need to be true *right now*, check `docs/LIMITATIONS.md`,
`CHANGELOG.md`'s latest entries, or query the running deployment
directly (`/api/health`, `/api/config`, `/api/readiness`).

## Contents

- **[`releases/`](releases/)** -- release-specific acceptance records:
  `v1.2.0-acceptance.md`, `RELEASE-EVIDENCE-v1.9.0.md`,
  `RELEASE-MATRIX-v1.9.0.md`.
- **[`vnext-baseline/`](vnext-baseline/)** -- the pre-v1.9.0 baseline
  captures (paid-run acceptance criteria, phase-by-phase budget and
  retrieval audits) referenced by `RELEASE-EVIDENCE-v1.9.0.md`.
