# Captures of runs that failed

Not run artifacts. A run that produced no result has no report, no
metrics and no reconciliation, so it cannot satisfy the contract the
directories beside this one are held to — and a directory that
pretends otherwise is worse than no directory.

They are kept because a failure is evidence. The raw stream shows how
far the pipeline got and what it emitted instead of a report, which is
what a fix has to be checked against.

## v140-20260929-161315

The first hosted run on v1.4.0. It reached `assessing_coverage` with 6
sources and 28 extracted quotes, then emitted `error` and `done`
instead of a report.

The coverage critique is advisory — every number that routes the run
is computed from evidence already in hand before the model is called,
and its own comment said "the counted half still stands". But it
caught only `LLMError`, so any other failure ended a run that had
already paid for four searches and twenty-eight extractions in order
to lose an opinion.

That narrow catch was the defect, and it was in six other places with
the same shape: a defined fallback that fires only for the failure
type someone anticipated. All seven now degrade on any exception, log
the exception type into the run's error list, and cannot swallow the
wall-clock deadline — `asyncio.timeout` cancels with `CancelledError`,
which is a `BaseException` and passes straight through `except
Exception`.

**The underlying exception is not recorded here.** The public error
message is deliberately generic, because an exception string can carry
a URL or a provider detail, and the server-side log was not retrieved
before it rotated. So this artifact shows what failed and where, and
does not claim to show why.
