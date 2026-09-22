# Integration tests

These are excluded from a default `pytest` run and from CI, because they
either spend API credits or need a local Ollama server.

```bash
# Local models only — no API keys, no cost. Needs `ollama serve`.
pytest -m ollama

# Live search and cloud models. Spends Tavily credits and OpenAI tokens.
pytest -m integration
```

Each test skips itself with a clear reason when its dependency is absent, so
running the whole suite on a machine with neither is silent rather than red.
