# Multi-stage: the build stage carries the toolchain, the runtime image does not.
FROM python:3.12-slim AS build

WORKDIR /build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --upgrade pip build && python -m build --wheel


FROM python:3.12-slim AS runtime

# trafilatura parses HTML through lxml, which needs these at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libxml2 libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# Never run as root in a container that fetches arbitrary web pages.
RUN useradd --create-home --uid 1000 researcher
WORKDIR /home/researcher

COPY --from=build /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

COPY pricing.toml ./
USER researcher

ENV PYTHONUNBUFFERED=1 \
    OUTPUT_DIR=/home/researcher/outputs \
    CHECKPOINT_PATH=/home/researcher/checkpoints/research.sqlite \
    LOG_FORMAT=json

# From inside a container, localhost is the container. Point Ollama at the
# host explicitly when running in local or hybrid mode:
#   docker run -e OLLAMA_BASE_URL=http://host.docker.internal:11434 ...
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434

ENTRYPOINT ["agentic-research"]
CMD ["--help"]
