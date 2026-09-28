# Remote NLI endpoint

Sequence-pair entailment scoring, and nothing else. This is the server
side of the contract in
`agentic_research.citations.nli.RemoteNLIVerifier`.

## Why it exists

Claim support is decided by a pinned NLI classifier. Its measured peak
is about **1.3 GB**; the hosted web plan has **512 MB**. Putting torch
into `Dockerfile.web` would not make it fit — it would only make the
image too large to deploy. So the verifier runs here, and the web
service calls it.

It cannot do anything else. No search, no generation, no synthesis, no
retrieval, no publication decision. It turns `(premise, hypothesis)`
pairs into three probabilities. The threshold, the deterministic guards
and the publish/withhold decision stay in the research engine, where
they are tested.

## The pin

| | |
|---|---|
| Model | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| Revision | `b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7` |
| Threshold (applied by the caller) | `0.98` |

The revision is not decoration. The threshold was calibrated against
this exact commit; a checkpoint that moves under a fixed threshold is
no longer the verifier that was measured. The service refuses a request
naming a different model or revision, and the client refuses a response
that reports one.

Weights are **not** baked into the image. Hugging Face mounts the
pinned repository at `/repository` and the service loads from there
with `local_files_only`, so the served checkpoint is whatever the
endpoint was configured with and cannot drift between restarts.

## Deploying

Create a **private, dedicated** Hugging Face Inference Endpoint:

- Repository: the model above
- **Revision: the exact commit above** — not `main`
- Container: custom, built from this directory
- Accelerator: CPU with room for ~1.4 GB resident
- **Replicas: min 1, max 1** for initial acceptance

Prove correctness while warm. Only afterwards enable scale-to-zero and
test the bounded 502/503 warm-up path, because a cold endpoint returns
502 with no request queue.

Store the token as a secret in the web service. Never commit it.

## Cost

This endpoint bills **hourly, independently of OpenAI and Tavily**. An
OpenAI spend limit does not bound it. A warm endpoint accrues cost
whether or not anybody runs research.

## Contract

`GET /health` → readiness, model id, expected revision, contract
version, resolved label order. Nothing secret.

`POST /score`:

```json
{ "contract_version": 1,
  "model": "...", "revision": "...",
  "pairs": [{"pair_id": "0", "premise": "...", "hypothesis": "..."}] }
```

```json
{ "contract_version": 1, "model_id": "...", "model_revision": "...",
  "results": [{"pair_id": "0", "entailment": 0.0, "neutral": 0.0,
               "contradiction": 0.0, "truncated": false}] }
```

Every field is required. The client withholds the claim if the contract
version is unsupported, the model or revision differs from what it
configured, `truncated` is absent, the three probabilities are not a
finite distribution, or the `pair_id`s do not match what it sent —
missing, duplicated, unknown or short. Results are matched by id, never
by position: a reordered response would otherwise attach one claim's
score to another claim, which is the worst failure available because
nothing downstream could detect it.

`truncated` is computed, never assumed. A premise whose qualifying half
was cut is not the premise, and the caller refuses to publish on it.

Class order is read from `model.config.id2label`. This checkpoint puts
entailment at index 0 and the cross-encoder family puts contradiction
there; hardcoding would invert the gate while still returning plausible
numbers.

## Local check

```bash
docker build -t nli-service .
docker run --rm -p 8099:80 \
  -e NLI_MODEL_DIR=/repository \
  -e NLI_MODEL_REVISION=b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7 \
  -v /path/to/checkpoint:/repository nli-service
curl localhost:8099/health
```
