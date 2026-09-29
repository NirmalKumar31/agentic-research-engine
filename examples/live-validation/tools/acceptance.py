"""Hosted acceptance capture: the credential-free checks, then one run.

Written as a committed tool rather than performed by hand because the
first hosted capture was done by hand and lost its own evidence. It
read the stream line by line and truncated each line at 150
characters, which destroyed the result payload -- and the service runs
with PERSIST_RUNS=false, so the stream was the only copy. There was
nothing to go back to.

So the rule here is: bytes to disk, unchanged, as they arrive. Timing
is recorded in a separate index rather than interleaved, because a
capture that annotates the stream is no longer the stream. Everything
else -- the contract, the propositions, the relevance decisions, the
repairs, the citations, the usage -- is *derived* from that file
afterwards by ``build``, never from a second live run. One authorised
run means one, including when the derivation turns out to be wrong.

Three subcommands, in the order acceptance uses them:

``checks``  Credential-free probes of a deployed service. Spends
            nothing. Refuses to probe the research route unless the
            service reports live research disabled.
``capture`` The one authorised live run. Writes only the raw stream
            and its index.
``build``   Turns a capture into the committed artifact. Offline, and
            repeatable against the same bytes.

The tool holds no credentials and reads none from the environment. The
public demo route needs none, which is what makes an acceptance run
reproducible by a reader.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

# Longer than DEMO_MAX_RUNTIME_SECONDS plus the cold start of a free
# instance. A client timeout shorter than the server's wall-clock
# ceiling turns a slow run into a lost one, which is the same data loss
# the truncation caused.
READ_TIMEOUT_SECONDS = 420.0
CONNECT_TIMEOUT_SECONDS = 30.0

# Applied to every file before it is written into the artifact. The
# same patterns the artifact test enforces, checked here so a violation
# is caught while the capture can still be redone rather than at commit
# time when the run is gone.
FORBIDDEN: dict[str, re.Pattern[str]] = {
    "hugging face endpoint url": re.compile(r"[a-z0-9]{16,}\.endpoints\.huggingface\.cloud"),
    "local path": re.compile(r"(/Users/|/home/[a-z])"),
    "bearer token": re.compile(r"Bearer\s+[A-Za-z0-9_\-.]{12,}"),
    "hf token": re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    "openai key": re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
    "tavily key": re.compile(r"\btvly-[A-Za-z0-9]{20,}"),
    "redis url": re.compile(r"redis://\S*:\S*@"),
}


def scan(name: str, text: str) -> list[str]:
    return [f"{name}: {label}" for label, pattern in FORBIDDEN.items() if pattern.search(text)]


# --------------------------------------------------------------------
# checks -- credential-free, spends nothing
# --------------------------------------------------------------------


# Three states, not two. The research probe is skipped when live
# research is enabled, and calling that a failure made a clean run
# against production report 15/16 and exit non-zero -- which trains a
# reader to ignore the exit code, the one thing it is for.
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


def verdict(ok: bool) -> str:
    return PASS if ok else FAIL


def run_checks(base: str, expect_namespace: str | None) -> int:
    """Everything that can be established without spending a run.

    Deliberately includes a POST to the research route, but only after
    readiness has reported live research disabled. That probe is the
    only way to demonstrate the staging interlock actually holds --
    asserting it from the blueprint proves what was configured, not
    what the process did with it.
    """
    findings: list[tuple[str, str, str]] = []
    with httpx.Client(base_url=base, timeout=httpx.Timeout(60.0, connect=30.0)) as client:

        def get(path: str) -> tuple[int, Any]:
            response = client.get(path)
            try:
                return response.status_code, response.json()
            except ValueError:
                return response.status_code, response.text

        code, health = get("/api/health")
        findings.append(("GET /api/health is 200", verdict(code == 200), str(code)))
        if isinstance(health, dict):
            findings.append(
                (
                    "health reports a version and commit",
                    verdict(bool(health.get("version")) and health.get("commit") != "unavailable"),
                    f"{health.get('version')} / {health.get('commit')}",
                )
            )
            findings.append(
                (
                    "demo_mode is on",
                    verdict(health.get("demo_mode") is True),
                    str(health.get("demo_mode")),
                )
            )

        code, readiness = get("/api/readiness")
        live_enabled = (
            bool(readiness.get("live_research_enabled")) if isinstance(readiness, dict) else True
        )
        findings.append(("GET /api/readiness answers", verdict(code in (200, 503)), str(code)))
        if isinstance(readiness, dict):
            findings.append(
                ("ready", verdict(readiness.get("ready") is True), json.dumps(readiness))
            )
            findings.append(
                (
                    "replay works independently of live research",
                    verdict(readiness.get("replay_available") is True),
                    str(readiness.get("replay_available")),
                )
            )

        # The namespace is the only thing keeping two deployments that
        # share a Key Value store from sharing one daily allowance, so
        # it is asserted against the running process rather than read
        # off a dashboard. Checked only when the caller says what to
        # expect: the tool cannot know which deployment it is pointed
        # at, and guessing would either pass vacuously or fail the
        # public demo, whose namespace is correctly empty.
        if expect_namespace is not None and isinstance(readiness, dict):
            actual = readiness.get("quota_namespace")
            findings.append(
                (
                    f"quota namespace is {expect_namespace!r}",
                    verdict(actual == expect_namespace),
                    f"reported {actual!r}",
                )
            )
        elif isinstance(readiness, dict):
            findings.append(
                (
                    "quota namespace reported",
                    SKIP,
                    f"{readiness.get('quota_namespace')!r} (pass --expect-namespace to assert)",
                )
            )

        code, config = get("/api/config")
        findings.append(("GET /api/config is 200", verdict(code == 200), str(code)))
        if isinstance(config, dict):
            findings.append(
                (
                    "config exposes no secret-shaped value",
                    verdict(not scan("config", json.dumps(config))),
                    "clean" if not scan("config", json.dumps(config)) else "LEAK",
                )
            )

        code, examples = get("/api/examples")
        listed = examples.get("examples", []) if isinstance(examples, dict) else []
        findings.append(
            ("recorded examples are served", verdict(bool(listed)), f"{len(listed)} listed")
        )
        for summary in listed:
            example_id = summary["id"]
            code, body = get(f"/api/examples/{example_id}")
            ok = code == 200 and isinstance(body, dict) and "result" in body
            findings.append((f"example {example_id} replays", verdict(ok), str(code)))

        # The interactive schema is removed in demo mode on purpose: it
        # is a machine-readable description of an endpoint that spends
        # money.
        for path in ("/openapi.json", "/docs", "/redoc"):
            code, _ = get(path)
            findings.append((f"{path} is not served", verdict(code == 404), str(code)))

        if live_enabled:
            findings.append(
                (
                    "research route probed",
                    SKIP,
                    "live research is enabled here, and a probe would spend a run",
                )
            )
        else:
            response = client.post(
                "/api/research",
                json={"query": "What did the study measure about developer productivity?"},
            )
            findings.append(
                (
                    "research is refused while live research is disabled",
                    verdict(response.status_code in (403, 503)),
                    f"{response.status_code} {response.text[:200]}",
                )
            )

    width = max(len(name) for name, _, _ in findings)
    failed = sum(1 for _, state, _ in findings if state == FAIL)
    skipped = sum(1 for _, state, _ in findings if state == SKIP)
    for name, state, detail in findings:
        print(f"{state}  {name:<{width}}  {detail}")
    checked = len(findings) - skipped
    print(f"\n{checked - failed}/{checked} passed, {skipped} skipped")
    return 1 if failed else 0


# --------------------------------------------------------------------
# capture -- the one authorised run
# --------------------------------------------------------------------


def run_capture(base: str, question: str, out: Path) -> int:
    """Stream one run to disk, byte for byte.

    Bytes are flushed as they arrive so a process killed mid-run still
    leaves everything that reached it. The index records the offset,
    length and arrival time of each chunk, which is enough to replay
    the timing without putting a single byte of annotation into the
    stream itself.
    """
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "stream.raw.sse"
    index_path = out / "stream.index.jsonl"

    started = time.time()
    started_iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    offset = 0
    chunks = 0

    timeout = httpx.Timeout(READ_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS)
    with (
        raw_path.open("wb") as raw,
        index_path.open("w", encoding="utf-8") as index,
        httpx.Client(base_url=base, timeout=timeout) as client,
        client.stream(
            "POST",
            "/api/research",
            json={"query": question},
            headers={"Accept": "text/event-stream"},
        ) as response,
    ):
        print(f"HTTP {response.status_code}", file=sys.stderr)
        if response.status_code != 200:
            body = response.read()
            raw.write(body)
            print(body.decode("utf-8", "replace")[:2000], file=sys.stderr)
            return 1
        for chunk in response.iter_bytes():
            if not chunk:
                continue
            raw.write(chunk)
            raw.flush()
            index.write(
                json.dumps(
                    {
                        "offset": offset,
                        "length": len(chunk),
                        "elapsed_s": round(time.time() - started, 3),
                    }
                )
                + "\n"
            )
            index.flush()
            offset += len(chunk)
            chunks += 1
            # Progress on stderr, so stdout stays free and nothing the
            # service sent is reformatted on its way to the file.
            print(".", end="", file=sys.stderr, flush=True)

    elapsed = round(time.time() - started, 2)
    print(f"\n{offset} bytes in {chunks} chunks over {elapsed}s -> {raw_path}", file=sys.stderr)

    # Say whether the stream finished, rather than reporting a byte
    # count and letting a reader assume it did.
    #
    # Two captures were reported this way and neither was a run: one
    # ended with an `error` event after the coverage critique, the
    # other simply stopped mid-wake with no terminal event at all. The
    # tool printed a byte count both times and said nothing, which is
    # the same defect it exists to prevent -- a number that looks like
    # success.
    text = raw_path.read_text(encoding="utf-8", errors="replace")
    complete = "event: done" in text
    has_result = "event: result" in text
    has_error = "event: error" in text
    if not complete or not has_result:
        print(
            "\nINCOMPLETE CAPTURE — this is not a run artifact:"
            f"\n  result event: {'yes' if has_result else 'NO'}"
            f"\n  done event  : {'yes' if complete else 'NO'}"
            f"\n  error event : {'yes' if has_error else 'no'}"
            "\n  The bytes are kept. `build` will refuse them.",
            file=sys.stderr,
        )
    (out / "capture.json").write_text(
        json.dumps(
            {
                "complete": complete and has_result,
                "terminal_event": ("result" if has_result else ("error" if has_error else "none")),
                "started_utc": started_iso,
                "wall_clock_seconds": elapsed,
                "raw_bytes": offset,
                "chunks": chunks,
                "service_url": base,
                "question": question,
                "method": "raw bytes streamed to disk, timing recorded in a separate index",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


# --------------------------------------------------------------------
# build -- derive the artifact, offline
# --------------------------------------------------------------------


def parse_events(raw: str) -> list[tuple[str, Any]]:
    """Every SSE event, in order, with its data parsed where it is JSON."""
    events: list[tuple[str, Any]] = []
    for block in raw.split("\n\n"):
        name = None
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data_lines.append(line[len("data: ") :])
        if name is None:
            continue
        payload: Any = "\n".join(data_lines)
        # A non-JSON data line is kept as text rather than dropped: the
        # point of the capture is that nothing the service sent is lost
        # because the parser did not recognise it.
        with contextlib.suppress(ValueError):
            payload = json.loads(payload)
        events.append((name, payload))
    return events


def verification_records(result: dict[str, Any]) -> dict[str, Any]:
    """The v1.2.0 decisions, pulled apart into their own files.

    One judgment object carries support, decomposition, relevance and
    repair together, which is right for the payload and unreadable as
    evidence. Split so a reviewer can answer one question at a time.
    """
    verification = result.get("verification") or {}
    judgments = verification.get("judgments") or []

    propositions = [
        {
            "claim": j["claim_text"],
            "parts": j["propositions"],
            "verdict": j.get("verdict"),
        }
        for j in judgments
        if j.get("propositions")
    ]
    relevance = [
        {
            "claim": j["claim_text"],
            "stage": j["relevance"]["stage"],
            "relevant": j["relevance"]["relevant"],
            "reason": j["relevance"]["reason"],
            "published": j.get("publishable"),
        }
        for j in judgments
        if j.get("relevance")
    ]
    repairs = [
        {
            "original": j["repair"]["original_text"],
            "rewritten": j["repair"]["repaired_text"],
            "guard": j["repair"]["guard"],
            "accepted": j["repair"]["accepted"],
            "reason": j["repair"]["reason"],
        }
        for j in judgments
        if j.get("repair")
    ]
    withheld = [
        {
            "claim": j["claim_text"],
            "verdict": j.get("verdict"),
            "reason": j.get("reason"),
            "best_entailment": j.get("best_entailment"),
            "best_evidence_id": j.get("best_evidence_id"),
            "guards_passed": j.get("guards_passed"),
        }
        for j in judgments
        if not j.get("publishable")
    ]
    return {
        "propositions": propositions,
        "relevance-decisions": relevance,
        "repairs": repairs,
        "withheld-reasons": withheld,
    }


def reconciliation(result: dict[str, Any]) -> dict[str, Any]:
    """The derived sections a reader checks the artifact against.

    Computed from the stream rather than written alongside it. The
    first summary of a run got this wrong by counting only
    ``unsupported`` and forgetting that a partially supported claim is
    withheld too, and a hand-written number cannot be caught by
    arithmetic that was also written by hand.

    ``withheld`` is deliberately the sum of its parts rather than
    ``checked - published``. The subtraction would satisfy the
    reconciliation identity by construction and check nothing; adding
    the categories makes a miscount fail.
    """
    metrics = result.get("metrics") or {}
    v = result.get("verification") or {}

    exact = metrics.get("exact_quotes", 0)
    fuzzy = metrics.get("fuzzy_quotes", 0)
    unmatched = metrics.get("unmatched_quotes", 0)
    extracted = exact + fuzzy + unmatched

    unsupported = v.get("unsupported_claims", 0)
    partial = v.get("partially_supported_claims", 0)
    published = v.get("final_published_claims", 0)
    checked = v.get("checked_claims", 0)

    reserved = metrics.get("reserved_worst_case_usd")
    known = metrics.get("known_cost_usd")
    ratio = (
        f"{reserved / known:.1f}x actual here."
        if known and reserved
        else "no priced call to compare against."
    )

    return {
        "claims_reconciliation": {
            "generated": v.get("generated_substantive_claims", 0),
            "checked": checked,
            "not_checked": v.get("not_checked_claims", 0),
            "published": published,
            "withheld": unsupported + partial,
            "withheld_unsupported": unsupported,
            "withheld_partially_supported": partial,
            "note": (
                "Only a supported and relevant claim publishes. A claim demoted "
                "to irrelevant is counted under unsupported, because it is "
                "withheld for the same reason from the reader's point of view: "
                "it is not in the report."
            ),
        },
        "quote_fidelity": {
            "definition": (
                "Exact normalised substring match against the fetched source "
                "text. Fuzzy matches are counted separately and are not citable."
            ),
            "denominator_extracted_quotes": extracted,
            "numerator_exact_quotes": exact,
            "fuzzy_quotes": fuzzy,
            "unmatched_quotes": unmatched,
            "rate": round(exact / extracted, 4) if extracted else None,
        },
        "cost_field_definitions": {
            "known_cost_usd": (
                "OpenAI token cost only, the engine's arithmetic over "
                "provider-reported counts, priced from pricing.toml."
            ),
            "reserved_worst_case_usd": (
                "Sum of the pre-dispatch upper bounds. Bytes, not tokens -- "
                "see llm/token_bound.py. " + ratio
            ),
            "cost_is_complete": metrics.get("cost_is_complete"),
            "unpriced_calls": metrics.get("unpriced_calls"),
            "total_cross_provider_cost_complete": False,
            "why_not_complete": (
                "Hugging Face endpoint uptime and Tavily credits are billed by "
                "their own providers and are not visible to this process."
            ),
        },
    }


def citations(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Every published claim with the quote and URL behind it.

    Resolved here rather than trusted from the report: the point of the
    artifact is that a reader can follow a claim to a page, and an id
    that resolves to nothing is exactly what that has to expose.
    """
    evidence = {e["id"]: e for e in result.get("evidence") or []}
    sources = {s["id"]: s for s in result.get("sources") or []}
    report = result.get("report") or {}

    claims: list[dict[str, Any]] = []
    groups = [report.get("summary_claims") or [], report.get("key_findings") or []]
    groups += [section.get("claims") or [] for section in report.get("sections") or []]
    for group in groups:
        for claim in group:
            resolved = []
            for evidence_id in claim.get("evidence_ids") or []:
                item = evidence.get(evidence_id)
                source = sources.get(item["source_id"]) if item else None
                resolved.append(
                    {
                        "evidence_id": evidence_id,
                        "resolved": item is not None,
                        "quote": (item or {}).get("quote"),
                        "quote_match": (item or {}).get("quote_match"),
                        "source_url": (source or {}).get("url"),
                        "source_title": (source or {}).get("title"),
                    }
                )
            claims.append({"claim": claim["text"], "kind": claim["kind"], "evidence": resolved})
    return claims


def run_build(run_dir: Path) -> int:
    raw = (run_dir / "stream.raw.sse").read_text(encoding="utf-8")
    events = parse_events(raw)
    names = [name for name, _ in events]
    if "result" not in names:
        print(
            "no result event in this capture, so there is no run to describe.\n"
            "A failed run has no report, metrics or reconciliation and cannot be\n"
            "made into a run artifact. Keep the bytes under\n"
            "examples/live-validation/failures/ instead.",
            file=sys.stderr,
        )
        return 1
    result = next(payload for name, payload in events if name == "result")

    written: dict[str, str] = {}

    def write(name: str, content: str) -> None:
        offenders = scan(name, content)
        if offenders:
            raise SystemExit(f"refusing to write: {offenders}")
        (run_dir / name).write_text(content, encoding="utf-8")
        written[name] = content

    write("report.md", result.get("markdown") or "")
    # The engine's own metrics, plus the derived sections a reader
    # checks them against. Both, rather than a curated subset: the
    # subset is what a reviewer reads, and the raw numbers are what
    # makes the subset checkable.
    metrics_out: dict[str, Any] = dict(result.get("metrics") or {})
    metrics_out.update(reconciliation(result))
    metrics_out["verification"] = {
        key: value
        for key, value in (result.get("verification") or {}).items()
        if not isinstance(value, list) or all(not isinstance(v, dict) for v in value)
    }
    write("metrics.json", json.dumps(metrics_out, indent=2, sort_keys=True) + "\n")
    write("contract.json", json.dumps(result.get("contract"), indent=2) + "\n")
    write("sources.json", json.dumps(result.get("sources") or [], indent=2) + "\n")
    write("citations.json", json.dumps(citations(result), indent=2) + "\n")
    for name, payload in verification_records(result).items():
        write(f"{name}.json", json.dumps(payload, indent=2) + "\n")

    # Provider usage, kept as its own file: it is the number that says
    # what the run cost, and burying it in metrics.json is how a
    # reviewer ends up quoting an estimate instead.
    metrics = result.get("metrics") or {}
    write(
        "provider-usage.json",
        json.dumps(
            {
                key: metrics.get(key)
                for key in (
                    "llm_calls",
                    "provider_requests",
                    "input_tokens",
                    "output_tokens",
                    "known_cost_usd",
                    "cost_display",
                    "search_calls",
                    "search_credits",
                    "model_assignments",
                    "duration_s",
                )
                if key in metrics
            },
            indent=2,
        )
        + "\n",
    )

    write(
        "events.json",
        json.dumps(
            [
                {"event": name, "data": payload}
                for name, payload in events
                # The result is committed in full elsewhere; repeating
                # 45KB of it here would make the event log unreadable.
                if name != "result"
            ],
            indent=2,
        )
        + "\n",
    )

    lines = []
    for path in sorted(run_dir.iterdir()):
        if path.name == "checksums.sha256" or path.is_dir():
            continue
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (run_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"built {len(written) + 1} files in {run_dir}")
    print("still to write by hand: README.md, manual-review.md, environment.json")
    print("then re-run `build` so the checksums cover them")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    checks = sub.add_parser("checks", help="credential-free probes; spends nothing")
    checks.add_argument("--base", required=True, help="https://<service>.onrender.com")
    checks.add_argument(
        "--expect-namespace",
        default=None,
        help=(
            "assert the deployment reports this DEMO_QUOTA_NAMESPACE "
            "(use '' for the public demo, 'rc' for a release candidate)"
        ),
    )

    capture = sub.add_parser("capture", help="one live run, streamed to disk")
    capture.add_argument("--base", required=True)
    capture.add_argument("--question", required=True)
    capture.add_argument("--out", required=True, type=Path)

    build = sub.add_parser("build", help="derive the artifact from a capture, offline")
    build.add_argument("--run-dir", required=True, type=Path)

    args = parser.parse_args()
    if args.command == "checks":
        return run_checks(args.base.rstrip("/"), args.expect_namespace)
    if args.command == "capture":
        return run_capture(args.base.rstrip("/"), args.question, args.out)
    return run_build(args.run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
