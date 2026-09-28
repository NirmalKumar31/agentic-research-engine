# Hosted deployment evidence

What the live deployment is configured as, and what was verified against
it without spending anything. Recorded so the configuration behind any
hosted claim can be checked rather than taken on trust.

**This is deployment acceptance. It is not a research-quality
evaluation, and nothing here says the engine answers questions well.**

## Configuration verified 2026-09-28

### Web service

| | |
| --- | --- |
| Service | `agentic-research-engine-live` (separate from the replay service, which is unchanged) |
| Deployed commit | `d0e0f992` |
| Version | `1.1.0` |
| Auto-deploy | off (`autoDeployTrigger: "off"`) |
| Health check path | `/api/readiness` |
| Web plan | free |

### Shared quota

| | |
| --- | --- |
| Store | Render Key Value, **free plan**, private (`ipAllowList: []`) |
| Wiring | `DEMO_QUOTA_URL` from `fromService` / `connectionString`, never pasted |
| Daily admissions | **5** = 150 // 30 |
| Per client | 2/hour, 1 concurrent run |

The free plan has no persistence. The counter is **atomic and shared
across web replicas and survives a web cold start**; it does **not**
survive a restart of the Key Value service itself, which would return
the day's five admissions. The hard spend limit on the OpenAI project
is the financial backstop, not this counter.

### Verifier endpoint (Hugging Face)

| | |
| --- | --- |
| Visibility | private |
| Replicas | min **0**, max **1** |
| Automatic scale-to-zero | **after 15 minutes idle** (changed from 1 hour on 2026-09-28) |
| Hardware | Intel Sapphire Rapids, 2 vCPU, CPU only |
| Repository | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| Revision | `b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7` (pinned; checked against the control plane at preflight) |
| Task | `text-classification` |
| Measured cold start | ~57s hosted, 49.2s from the CLI |
| Configured warm-up budget | 90s, inside a 240s run deadline |

Why 15 minutes rather than an hour: the endpoint bills per hour of
uptime whether or not anyone uses it. At a 60-minute idle timeout five
separated runs can mean five billed hours; at 15 minutes the same five
runs bill about 1.25 hours.

## Cost domains — three, bounded separately

None of these bounds any other.

| Domain | Bound | Worst case at 5 runs/day |
| --- | --- | --- |
| OpenAI | project hard spend limit ($5) | ~$0.25/day → about **20 worst-case days**, not a month |
| Tavily | plan credits | 5 x 6 = 30 queries/day, ~900/month |
| Hugging Face | endpoint uptime | 5 x 15 min ≈ 1.25h ≈ $0.084/day |

The per-run `MAX_CLOUD_COST_USD` is an **estimate**, not a billing cap:
input tokens are approximated from character length, which undercounts
CJK, emoji, dense punctuation, code and JSON schemas. Provider-side
enforcement is the backstop, and it is neither instantaneous nor exact
per request.

## Credential-free acceptance, 2026-09-28

All verified against the deployed service. None of it spent anything.

| Check | Result |
| --- | --- |
| Deployed SHA matches the reviewed commit | `d0e0f992` |
| `/api/readiness` | 200, all states green, `shared_quota_available: true` |
| `/api/health` | 200, `global_runs_per_day: 5`, no `runs_today` |
| `/api/config` | public limits active: 1 round, 6 sources, 240s |
| Replay | 3 recordings served |
| Security headers on the SPA | CSP, nosniff, Referrer-Policy, Permissions-Policy, X-Frame-Options, HSTS, COOP |
| Same headers on an API error | 6/6 present |
| `/docs`, `/redoc`, `/openapi.json` | **404** with JSON, not the app shell |
| Oversized request body | **413** |
| Declared-but-false `Content-Length` | **400** |
| Frontend under the new CSP | 200, zero inline scripts, all references same-origin |
| Secrets in any response body | none |

### Earlier hosted run, commit `7e565492`

One live run on 2026-09-28 at the public limits, before the fixes above:

- verifier cold start from `scaledToZero`: **~57s**
- complete run: **142s** against the 240s ceiling
- 6 sources, 32 evidence items, **2 claims checked, 0 key findings**
- the shared cap refused the next request, and refused it again for
  three forged `X-Forwarded-For` values
- replay kept working with the allowance spent

**Its final payload was not captured** — the capture truncated the
result event — so exact provider usage and the exact charge for that
run are unknown. A thin two-claim report is also not evidence of
research quality. Both gaps are open.

## Known limitations

- The free Key Value counter resets if the Key Value service restarts.
- `MAX_CLOUD_COST_USD` is an estimate; exact pre-dispatch token counting
  is not implemented.
- Cached-input, cache-write and reasoning token categories are not
  tracked or priced.
- Hugging Face and Tavily spending are independent of the OpenAI cap.
- Client disconnect and application-timeout cleanup have unit coverage
  but no hosted measurement.
- One hosted run is not an evaluation. Replay remains the fuller
  demonstration of what the engine produces.
