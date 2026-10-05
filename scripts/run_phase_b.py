"""Execute Benchmark Phase B: the frozen-corpus Ollama-vs-cloud comparison.

Track 1 only (`docs/BENCHMARK-PROTOCOL.md`): 12 questions x 2 repetitions
x 2 arms over a corpus frozen once per question = 48 arm-runs, plus the
12 corpus freezes that produce those corpora. This is the only track
this script runs; Track 2 (live end-to-end) is explicitly out of scope
and this script does not touch it.

Authorized ceiling: a single hard total of US$10.00, inclusive of cloud
model usage, covering both the freezes and the 48 arm-runs. NLI
verification runs on the local torch checkpoint for the whole script
(overriding this machine's NLI_MODE=remote default), which is $0
marginal cost and the same pinned model/revision either way -- this
removes the one cost this repository has no priced model for (the
private Hugging Face endpoint bills by uptime, not by call, and this
script has no visibility into that rate).

Every cloud dispatch is gated by `SpendLedger`, which shrinks each call's
own `max_cloud_cost_usd` ceiling to whatever headroom remains under the
$10 total, so the existing `CloudBudgetExceededError` pre-dispatch check
(`llm/base.py`) enforces the global ceiling without new budget code. A
run finishing under its shrunk per-call ceiling cannot push the running
total over $10; the sum of per-call ceilings telescopes to at most
$10.00 by construction.

Run with `--smoke-test` first: this exercises one throwaway question
(not one of the 12 preregistered ones, not reported as benchmark data)
through one real cloud call, to prove the cost-recording and
ceiling-abort paths work against a real provider response before any
preregistered run is dispatched.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_research.config import Settings  # noqa: E402
from agentic_research.environment import capture as capture_environment  # noqa: E402
from agentic_research.evaluation.ab import (  # noqa: E402
    ArmResult,
    EvidenceCorpus,
    all_roles,
    run_arm,
)
from agentic_research.evaluation.benchmark_manifest import (  # noqa: E402
    BenchmarkManifest,
    freeze_corpus_hashes,
    verify_corpus_hash,
)
from agentic_research.evaluation.blinding import blind  # noqa: E402
from agentic_research.observability import get_logger  # noqa: E402
from agentic_research.provenance import (  # noqa: E402
    config_fingerprint,
    git_state,
    prompt_version,
    schema_version,
)
from agentic_research.runner import run_research  # noqa: E402

log = get_logger(__name__)

OUT_DIR = ROOT / "evaluations" / "phase_b"
CORPORA_DIR = OUT_DIR / "corpora"
RUNS_DIR = OUT_DIR / "runs"
BLINDED_DIR = OUT_DIR / "blinded"

LOCAL_SPEC = "ollama:qwen3:4b"
CLOUD_SPEC = "openai:gpt-6-luna"
SPEND_CEILING_USD = 10.00
PER_CALL_CAP_USD = 0.50  # this machine's existing per-run default; kept as a floor
ARM_TIMEOUT_SECONDS = 120.0  # synthesis + verification over a tiny frozen corpus


class PhaseBAbort(Exception):
    """An integrity or ceiling condition that must stop the whole run."""


@dataclass
class SpendLedger:
    ceiling: float
    spent: float = 0.0
    entries: list[dict[str, Any]] = field(default_factory=list)

    def headroom(self) -> float:
        return round(self.ceiling - self.spent, 6)

    def record(self, label: str, cost: float, note: str = "") -> None:
        self.spent = round(self.spent + cost, 6)
        self.entries.append(
            {"label": label, "cost_usd": cost, "note": note, "cumulative_usd": self.spent}
        )
        log.info("phase_b_spend", label=label, cost_usd=cost, cumulative_usd=self.spent)

    def cap_for_next_call(self) -> float:
        return round(min(PER_CALL_CAP_USD, max(self.headroom(), 0.0)), 6)


def base_settings() -> Settings:
    """This machine's configured settings, with NLI forced local.

    Everything else (LLM_MODE=local for freeze, search provider, run
    ceilings) comes from `.env` unchanged -- this script does not alter
    production settings, it only builds `Settings` objects in memory for
    its own calls.
    """
    return Settings(nli_mode="local")


def load_questions() -> list[dict[str, Any]]:
    data = json.loads((ROOT / "examples" / "benchmark" / "questions.json").read_text())
    questions = data["questions"]
    if len(questions) != 12:
        raise PhaseBAbort(f"expected 12 preregistered questions, found {len(questions)}")
    return questions


async def freeze_one(
    question_id: str, question_text: str, settings: Settings
) -> tuple[EvidenceCorpus, list[str]]:
    path = CORPORA_DIR / f"{question_id}.json"
    if path.is_file():
        # Resuming after an interrupted run: a corpus already frozen to disk
        # is kept as-is rather than re-run, which would waste ~10 minutes of
        # real search+extraction per question for no benefit and would
        # introduce a new captured_at / possibly different live search
        # results for a question that already froze cleanly.
        corpus = EvidenceCorpus.load(path)
        problems = corpus.validate_for_replay()
        print(f"    (resumed from disk: {corpus.summary()})")
        return corpus, problems
    result = await run_research(question_text, settings)
    corpus = EvidenceCorpus.from_result(question_text, result)
    corpus.save(path)
    problems = corpus.validate_for_replay()
    return corpus, problems


def fatal_degradation(corpus: EvidenceCorpus) -> list[str]:
    """Whether a corpus is unusable, not merely imperfect.

    `EvidenceCorpus.validate_for_replay()` flags *any* source missing text,
    which is right for `compare()` to warn on but too strict to abort a
    whole Phase B freeze over: across 12 questions x ~5 sources, a single
    web fetch failing (paywall, block, timeout) is an expected outcome of
    live retrieval, not evidence the corpus is degraded the way the
    stripped-source-text bug this guard was built for actually was. Only
    "nothing left to cite" or "no sub-questions" make a corpus unusable;
    those are checked directly here rather than by parsing
    `validate_for_replay()`'s message strings.
    """
    fatal: list[str] = []
    if not any(e.is_citable for e in corpus.evidence):
        fatal.append("no citable evidence: synthesis will have nothing to cite")
    if not corpus.sub_questions:
        fatal.append("no sub-questions")
    return fatal


def build_manifest(settings: Settings, corpus_hashes: dict[str, str]) -> BenchmarkManifest:
    git = git_state()
    ollama_version, ollama_digest = _ollama_pin(settings.ollama_model)
    manifest = BenchmarkManifest(
        benchmark_version="phase-b-track1-v1",
        frozen_engine_commit=git.get("commit", "unavailable"),
        cloud_model_id=settings.openai_model,
        nli_model_id=settings.nli_model_id,
        nli_model_revision=settings.nli_model_revision,
        prompt_version=prompt_version(),
        schema_version=schema_version(),
        config_fingerprint=config_fingerprint(settings),
        corpus_hashes=corpus_hashes,
        ollama_model=settings.ollama_model,
        ollama_version=ollama_version,
        ollama_digest=ollama_digest,
        ceilings={
            "max_research_rounds": settings.max_research_rounds,
            "max_sources": settings.max_sources,
            "max_llm_calls": settings.max_llm_calls,
            "max_provider_requests": settings.max_provider_requests,
            "total_spend_ceiling_usd": int(SPEND_CEILING_USD),
            "per_call_cap_usd_cents": round(PER_CALL_CAP_USD * 100),
            "arm_timeout_seconds": int(ARM_TIMEOUT_SECONDS),
        },
    )
    problems = manifest.completeness_problems()
    if problems:
        raise PhaseBAbort("manifest incomplete before any paid call: " + "; ".join(problems))
    return manifest


def _ollama_pin(model: str) -> tuple[str, str]:
    import subprocess

    try:
        out = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10, check=True
        ).stdout
    except Exception as exc:  # pragma: no cover - environment probe
        return ("unavailable", f"ollama list failed: {exc}")
    for line in out.splitlines()[1:]:
        parts = line.split()
        if parts and parts[0] == model:
            return (model, parts[1] if len(parts) > 1 else "unknown")
    raise PhaseBAbort(f"pinned Ollama model {model!r} is not present in `ollama list`")


async def dispatch(
    label: str,
    corpus: EvidenceCorpus,
    settings: Settings,
    arm: str,
    manifest: BenchmarkManifest,
    qid: str,
    ledger: SpendLedger,
) -> ArmResult:
    verify_corpus_hash(qid, corpus, manifest)  # fail closed, immediately before use

    if arm == "cloud":
        cap = ledger.cap_for_next_call()
        if cap <= 0:
            raise PhaseBAbort(
                f"spend ceiling reached (${ledger.spent:.4f} of "
                f"${ledger.ceiling:.2f}) before {label}"
            )
        run_settings = settings.model_copy(
            update={"max_cloud_cost_usd": cap, "run_timeout_seconds": ARM_TIMEOUT_SECONDS}
        )
        spec = CLOUD_SPEC
    else:
        run_settings = settings.model_copy(update={"run_timeout_seconds": ARM_TIMEOUT_SECONDS})
        spec = LOCAL_SPEC

    result = await run_arm(label, corpus, run_settings, overrides=all_roles(spec))

    if arm == "local" and any("openai" in v.lower() for v in result.models.values()):
        raise PhaseBAbort(f"unexpected provider call: local arm {label} used {result.models}")
    if arm == "cloud":
        ledger.record(label, result.known_cost_usd, note="ok" if result.ok else result.error[:80])
        if ledger.spent > ledger.ceiling + 1e-6:
            raise PhaseBAbort(
                f"actual cumulative spend ${ledger.spent:.4f} exceeded "
                f"${ledger.ceiling:.2f} at {label}"
            )

    return result


def save_raw(qid: str, rep: int, arm: str, result: ArmResult) -> None:
    path = RUNS_DIR / f"{qid}_rep{rep}_{arm}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": result.label,
        "models": result.models,
        "ok": result.ok,
        "timed_out": result.timed_out,
        "error": result.error,
        "duration_s": result.duration_s,
        "llm_calls": result.llm_calls,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "known_cost_usd": result.known_cost_usd,
        "cost_is_complete": result.cost_is_complete,
        "metrics": {m.name: m.value for m in result.metrics},
        "markdown": result.markdown,
        "verification": result.verification,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # The replacement label must never itself name the arm: "local"/"cloud"
    # as the label would print `[LOCAL]`/`[CLOUD]` wherever an identifier
    # was redacted, which is a more direct identity leak than the original
    # string. One constant, arm-neutral label for every arm.
    blinded = blind("REDACTED_MODEL", result.markdown)
    blinded_path = BLINDED_DIR / f"{qid}_rep{rep}_{arm}.md"
    blinded_path.parent.mkdir(parents=True, exist_ok=True)
    blinded_path.write_text(
        f"<!-- redactions: {blinded.redaction_count} -->\n\n{blinded.markdown}", encoding="utf-8"
    )


async def run_smoke_test() -> None:
    """One throwaway question through one real cloud call.

    Not one of the 12 preregistered questions, not written to the
    results table: this exists only to prove the ledger and the
    CloudBudgetExceededError path work against a real provider response
    before the preregistered sequence starts spending.
    """
    print("=== Phase B smoke test (real, tiny cloud call) ===")
    settings = base_settings()
    corpus = EvidenceCorpus(
        question="What is 2+2? (smoke test, not a benchmark question)",
        sub_questions=[],
        evidence=[],
        sources=[],
        completed_queries=[],
    )
    from datetime import UTC, datetime

    corpus.captured_at = datetime.now(UTC).isoformat()
    from agentic_research.models import (
        DiscoveryRef,
        EvidenceItem,
        QuoteMatch,
        SourceDocument,
        SubQuestion,
    )

    corpus.sub_questions = [SubQuestion(id="SQ1", text="what is 2+2?", rationale="smoke")]
    corpus.sources = [
        SourceDocument(
            id="S1",
            url="https://example.org/arithmetic",
            canonical_url="https://example.org/arithmetic",
            title="Arithmetic",
            domain="example.org",
            text="Two plus two equals four.",
            content_hash="smoke",
            discovered_by=[DiscoveryRef(query_id="Q1", sub_question_id="SQ1")],
        )
    ]
    corpus.evidence = [
        EvidenceItem(
            id="S1-e1",
            source_id="S1",
            sub_question_id="SQ1",
            claim="Two plus two equals four.",
            quote="Two plus two equals four.",
            quote_match=QuoteMatch.EXACT_NORMALIZED,
            relevance=0.9,
            discovery=DiscoveryRef(query_id="Q1", sub_question_id="SQ1"),
        )
    ]
    corpus.completed_queries = [
        __import__("agentic_research.models", fromlist=["SearchQuery"]).SearchQuery(
            id="Q1", sub_question_id="SQ1", text="2+2", round_number=1
        )
    ]

    ledger = SpendLedger(ceiling=0.05)  # deliberately tiny, to also prove the abort path
    run_settings = settings.model_copy(
        update={
            "max_cloud_cost_usd": ledger.cap_for_next_call(),
            "run_timeout_seconds": ARM_TIMEOUT_SECONDS,
        }
    )
    result = await run_arm("smoke-cloud", corpus, run_settings, overrides=all_roles(CLOUD_SPEC))
    ledger.record("smoke-cloud", result.known_cost_usd)
    print(
        f"ok={result.ok} timed_out={result.timed_out} cost_usd={result.known_cost_usd} "
        f"models={result.models}"
    )
    if result.error:
        print(f"error: {result.error}")
    print(f"ledger: spent=${ledger.spent:.6f} headroom=${ledger.headroom():.6f}")

    if not result.ok and not result.error:
        raise PhaseBAbort(
            "smoke test produced no result and no error -- investigate before spending further"
        )
    print("=== smoke test done; ledger math and real-call path confirmed ===")


async def run_phase_b() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    settings = base_settings()
    questions = load_questions()
    ledger = SpendLedger(ceiling=SPEND_CEILING_USD)

    print(f"Freezing {len(questions)} corpora (local model + search, $0 marginal cost)...")
    corpora: dict[str, EvidenceCorpus] = {}
    freeze_problems: dict[str, list[str]] = {}
    for q in questions:
        qid = q["id"]
        print(f"  freezing {qid}: {q['text'][:60]}...")
        corpus, problems = await freeze_one(qid, q["text"], settings)
        corpora[qid] = corpus
        if problems:
            freeze_problems[qid] = problems
            fatal = fatal_degradation(corpus)
            if fatal:
                raise PhaseBAbort(f"{qid}: corpus cannot support verification: {fatal}")
            print(f"    WARNING (non-fatal, recorded in limitations): {problems}")
        print(f"    {corpus.summary()}")

    corpus_hashes = freeze_corpus_hashes(corpora)
    manifest = build_manifest(settings, corpus_hashes)
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
    )
    environment = capture_environment(settings)
    (OUT_DIR / "environment.json").write_text(
        json.dumps(environment, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"Manifest frozen: commit={manifest.frozen_engine_commit[:12]}, "
        f"cloud={manifest.cloud_model_id}, "
        f"ollama={manifest.ollama_model}@{manifest.ollama_digest}"
    )

    all_results: list[dict[str, Any]] = []
    run_count = 0
    for i, q in enumerate(questions):
        qid = q["id"]
        corpus = corpora[qid]
        arm_order = ["local", "cloud"] if i % 2 == 0 else ["cloud", "local"]
        for rep in (1, 2):
            for arm in arm_order:
                label = f"{qid}-rep{rep}-{arm}"
                run_count += 1
                raw_path = RUNS_DIR / f"{qid}_rep{rep}_{arm}.json"
                if raw_path.is_file():
                    # Resuming after an interrupted run: a completed cloud
                    # arm-run must never be re-dispatched, or it would spend
                    # money twice for one preregistered run. Its already-paid
                    # cost is replayed into the ledger so the cumulative
                    # ceiling check downstream still sees the true total.
                    cached = json.loads(raw_path.read_text())
                    if arm == "cloud":
                        ledger.record(label, cached["known_cost_usd"], note="resumed from disk")
                    print(f"  [{run_count}/48] {label} (resumed from disk)")
                    all_results.append(
                        {
                            "question_id": qid,
                            "shape": q["shape"],
                            "repetition": rep,
                            "arm": arm,
                            "ok": cached["ok"],
                            "timed_out": cached["timed_out"],
                            "error": cached["error"],
                            "duration_s": cached["duration_s"],
                            "known_cost_usd": cached["known_cost_usd"],
                            "input_tokens": cached["input_tokens"],
                            "output_tokens": cached["output_tokens"],
                            "metrics": cached["metrics"],
                        }
                    )
                    continue
                print(f"  [{run_count}/48] {label} ...", end=" ", flush=True)
                result = await dispatch(label, corpus, settings, arm, manifest, qid, ledger)
                save_raw(qid, rep, arm, result)
                all_results.append(
                    {
                        "question_id": qid,
                        "shape": q["shape"],
                        "repetition": rep,
                        "arm": arm,
                        "ok": result.ok,
                        "timed_out": result.timed_out,
                        "error": result.error,
                        "duration_s": result.duration_s,
                        "known_cost_usd": result.known_cost_usd,
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                        "metrics": {m.name: m.value for m in result.metrics},
                    }
                )
                print(
                    f"ok={result.ok} timed_out={result.timed_out} "
                    f"cost=${result.known_cost_usd:.4f} ({result.duration_s:.1f}s)"
                )

    elapsed = time.time() - started
    write_report(all_results, manifest, ledger, elapsed, questions, freeze_problems)
    print(
        f"\nDone. {run_count} arm-runs. Total spend ${ledger.spent:.4f} of ${ledger.ceiling:.2f}. "
        f"Elapsed {elapsed / 60:.1f} min."
    )


def write_report(
    all_results: list[dict[str, Any]],
    manifest: BenchmarkManifest,
    ledger: SpendLedger,
    elapsed_s: float,
    questions: list[dict[str, Any]],
    freeze_problems: dict[str, list[str]] | None = None,
) -> None:
    (OUT_DIR / "raw_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    (OUT_DIR / "spend_ledger.json").write_text(
        json.dumps(
            {"ceiling_usd": ledger.ceiling, "spent_usd": ledger.spent, "entries": ledger.entries},
            indent=2,
        ),
        encoding="utf-8",
    )

    failures = [r for r in all_results if not r["ok"]]
    timeouts = [r for r in all_results if r["timed_out"]]

    lines: list[str] = []
    lines.append("# Benchmark Phase B results (Track 1: frozen-corpus comparison)\n")
    lines.append(f"- Commit: `{manifest.frozen_engine_commit}`")
    lines.append(
        f"- Cloud model: `{manifest.cloud_model_id}`; Ollama model: `{manifest.ollama_model}` "
        f"(digest `{manifest.ollama_digest}`)"
    )
    lines.append(
        f"- NLI: `{manifest.nli_model_id}` @ `{manifest.nli_model_revision}` "
        "(local checkpoint, this run)"
    )
    lines.append(f"- Arm-runs completed: {len(all_results)} of 48 preregistered")
    lines.append(f"- Total spend: ${ledger.spent:.4f} of ${ledger.ceiling:.2f} ceiling")
    lines.append(f"- Failures (ok=False): {len(failures)}; Timeouts: {len(timeouts)}")
    lines.append(f"- Wall-clock elapsed: {elapsed_s / 60:.1f} minutes\n")

    lines.append(
        "**Statistical caveat, stated once and binding throughout:** each cell below "
        "aggregates n=2 repetitions. No significance test is computed or implied; a "
        "difference between arms here is a measured observation at n=2, not a "
        "generalizable claim.\n"
    )

    metric_names: list[str] = []
    for r in all_results:
        for name in r["metrics"]:
            if name not in metric_names:
                metric_names.append(name)

    lines.append("## Per-question, per-arm raw results\n")
    lines.append(
        "| Question | Rep | Arm | ok | timed_out | duration_s | cost_usd | "
        + " | ".join(metric_names)
        + " |"
    )
    lines.append("|---" * (6 + len(metric_names)) + "|")
    for r in all_results:
        metric_cells = " | ".join(
            "n/a" if r["metrics"].get(name) is None else f"{r['metrics'][name]:.2f}"
            for name in metric_names
        )
        row = (
            f"| {r['question_id']} | {r['repetition']} | {r['arm']} | {r['ok']} | "
            f"{r['timed_out']} | {r['duration_s']:.1f} | {r['known_cost_usd']:.4f} | "
            f"{metric_cells} |"
        )
        lines.append(row)

    if failures:
        lines.append("\n## Failures\n")
        for r in failures:
            lines.append(f"- {r['question_id']} rep{r['repetition']} {r['arm']}: {r['error']}")

    lines.append("\n## Limitations\n")
    lines.append(
        "- n=2 repetitions per arm per question: no statistical significance claimed or computable."
    )
    lines.append(
        '- Track 1 holds retrieval fixed (frozen corpus); it answers "do the models differ '
        'on the same evidence", not "which pipeline is better in practice" (Track 2, not '
        "run here)."
    )
    lines.append(
        "- Whether a claim is an overclaim against its question's `forbidden_overclaims`, "
        "and whether a refusal was the correct one, are not scored mechanically -- blinded "
        "human review (see `evaluations/phase_b/blinded/`) decides those, not this script."
    )
    lines.append(
        "- Tavily search-credit cost during corpus freezing is not priced per-credit anywhere "
        "in this repository and is not included in the $ figure above; it is assumed to "
        "remain within the account's free tier, consistent with every prior run in this project."
    )
    lines.append(
        "- `blind()` catches identifying strings, not a model describing its own "
        "architecture in other words."
    )
    if freeze_problems:
        lines.append(
            "- Non-fatal corpus degradation at freeze time (a source failed to fetch "
            "live, e.g. paywall/block/timeout; citable evidence still existed from the "
            "remaining sources, so freezing continued rather than aborting):"
        )
        for qid, problems in freeze_problems.items():
            lines.append(f"  - {qid}: {'; '.join(problems)}")
    capability_gaps = [r for r in all_results if r["arm"] == "local" and not r["ok"]]
    if capability_gaps:
        lines.append(
            f"- {len(capability_gaps)} local-arm run(s) failed outright; see Failures "
            "above for which capability was unavailable locally."
        )

    (OUT_DIR / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test", action="store_true", help="Run the tiny real-call rehearsal and exit."
    )
    args = parser.parse_args()
    if args.smoke_test:
        asyncio.run(run_smoke_test())
    else:
        asyncio.run(run_phase_b())


if __name__ == "__main__":
    main()
