# Manual review of the published findings

Every claim this run published, checked by hand against the quote it
cites and the page that quote came from. The engine's own verdict is
not evidence here — the point of the exercise is to find out whether
the verdict was right.

Reviewer: project author, assisted by Claude. Reviewed 2026-09-27.

**Verdict: 5 of 5 supported by their cited quotes. No claim was
published that its evidence does not carry.** Two carry provenance
caveats that are recorded below rather than smoothed over.

## Scope of this review

This checks one thing: does the cited quote support the claim as
written? It does **not** establish that the underlying research is
sound, that the source reported it accurately, or that the claim is
true of the world. A claim can pass here and still be wrong, if the
page it came from is wrong.

---

## 1. PASS — transcript-based measurement

**Claim.** One transcript-based study compared coding-agent transcripts
with hypothetical baseline task times estimated using LLM-as-a-judge
methods.

**Quote (S5).** "Their workaround was to analyze coding-agent
transcripts and compare them against a hypothetical baseline estimated
with LLM-as-a-judge methods."

**Assessment.** A direct restatement. The claim says "one
transcript-based study" where the quote says "their", which loses the
attribution but does not overstate it.

---

## 2. PASS, with a provenance caveat — the METR slowdown

**Claim.** A METR study reported that AI assistants slowed experienced
open-source developers on large, complex projects by as much as 20%.

**Quote (S5).** "A METR study from early 2025 found that AI assistants
actually slowed experienced open-source developers working on large,
complex projects by as much as 20% (METR, 2025)."

**Assessment.** The claim restates the quote faithfully.

**Caveat.** S5 is a Medium post with a heuristic source-quality score of
0.56, not the METR paper. The claim is therefore supported by a
secondary account. The engine cited what it read, which is the correct
behaviour, and the reader should treat "METR reported" here as "a blog
reports that METR reported". The primary result is widely cited as 19%;
"as much as 20%" is the blog's phrasing, and the engine reproduced the
blog rather than correcting it. Correcting it would have been the
defect.

---

## 3. PASS — technical debt and future velocity

**Claim.** The study's panel GMM models found that accumulated technical
debt subsequently reduced future velocity.

**Quote (S8).** "Panel GMM models reveal that accumulated technical debt
subsequently reduces future velocity, creating a self-reinforcing
cycle."

**Assessment.** Direct restatement, tense shifted to reported speech.
S8 is an arXiv paper, the highest-scoring source in the run.

---

## 4. PASS — what one study measured

**Claim.** One study focused on task-completion time rather than code
quality.

**Quote (S10).** "The study focused on task completion time rather than
code quality."

**Assessment.** Near-verbatim. Narrow and uninformative on its own, but
accurate, and it is a scope limitation rather than a result.

---

## 5. PASS, with an attribution caveat — slowdown mechanisms

**Claim.** That source identified developer over-optimism, low AI
reliability, and high task complexity as potential slowdown mechanisms.

**Quote (S8).** "Becker et al. (2025) show through controlled
experiments that early-2025 AI tools, including Cursor, do not help
experienced open-source developers solve real day-to-day tasks faster,
pointing to potential slowdown mechanisms such as developer
over-optimism, low AI reliability, and high task complexity."

**Assessment.** The three mechanisms appear in the quote and the claim
does not add to them.

**Caveat.** "That source identified" is loose. S8 is relaying Becker et
al.'s findings, so S8 reports the mechanisms rather than identifying
them. The attribution guard passed because the claim names no party the
evidence does not, but a stricter reading would want "a study S8 cites".
Not a false publication; worth noting as the weakest of the five.

---

## What the withheld claims say about the gate

35 of the 40 checked claims were withheld. Spot-checking the recorded
reasons, the gate refused several claims that read as the most
quotable:

- "In one controlled experiment, developers using Copilot completed a
  standardized HTTP-server task roughly 56% faster" — withheld on the
  atomicity guard.
- "In METR's randomized trial, developers allowed to use AI took 19%
  longer to complete tasks than the control group" — withheld on the
  atomicity guard.
- "The evidence includes controlled experiments reporting faster
  completion, slower completion, and no statistically significant
  difference" — withheld at entailment 0.000 against its best evidence.

The 19% figure is, as far as the reviewer knows, the correct primary
result, and the gate withheld it anyway because the claim bundled
several propositions into one sentence. That is the gate behaving as
designed and it is also its cost: correct statements are withheld when
they are not atomic. This is a trade-off, not a triumph, and the
published set is narrower and duller than the evidence would support.
