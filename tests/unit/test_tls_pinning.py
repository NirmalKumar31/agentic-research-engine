"""TLS certificate verification under IP pinning.

The pinning design connects to a validated address while still verifying
the certificate against the *original hostname*, carried in the TLS SNI.
That only works if httpx's ``sni_hostname`` extension genuinely drives
certificate validation. If it does not, one of two things is true and both
are serious:

* verification happens against the address in the URL, so every HTTPS
  fetch breaks; or
* verification is skipped, which is a silent security downgrade and
  strictly worse than not pinning at all.

Neither can be established by reading the code, so these tests run a real
TLS server with a real certificate and assert both directions.

Hermetic: the certificate authority is generated in-process by ``trustme``
and trusted explicitly via ``verify=<ca file>``. That is not the same as
disabling verification -- ``verify=False`` would make every assertion here
vacuous and must never appear in this file.
"""

from __future__ import annotations

import ast
import http.server
import ssl
import threading
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

trustme = pytest.importorskip(
    "trustme",
    reason=(
        "trustme generates the in-process CA these tests need. It is declared "
        "in the dev extra; install it with `pip install -e '.[dev]'`."
    ),
)

SERVER_HOSTNAME = "pinned.test"
WRONG_HOSTNAME = "attacker.test"
BODY = b"<html><body><article><p>served over TLS</p></article></body></html>"


def _client_trusting(ca_path: Path) -> httpx.AsyncClient:
    """A client that trusts the in-process CA and nothing else.

    Not a relaxation: the chain is still verified, against a CA we control
    instead of the system store. ``verify=<path>`` would do the same thing
    but is deprecated in httpx.
    """
    context = ssl.create_default_context(cafile=str(ca_path))
    return httpx.AsyncClient(verify=context, timeout=10)


class _QuietHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/html")
        self.send_header("content-length", str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY)

    def log_message(self, *args: object) -> None:
        """Silence the default stderr access log."""


@pytest.fixture(scope="module")
def tls_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[int, Path]]:
    """A real HTTPS server on loopback holding a cert for SERVER_HOSTNAME.

    Yields ``(port, ca_pem_path)``. The client trusts only this CA, so a
    hostname mismatch is a genuine verification failure rather than an
    artefact of an untrusted issuer.
    """
    directory = tmp_path_factory.mktemp("tls")
    authority = trustme.CA()
    certificate = authority.issue_cert(SERVER_HOSTNAME)

    ca_path = directory / "ca.pem"
    authority.cert_pem.write_to_path(str(ca_path))
    cert_path = directory / "server.pem"
    certificate.private_key_and_cert_chain_pem.write_to_path(str(cert_path))

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(cert_path))

    server = http.server.HTTPServer(("127.0.0.1", 0), _QuietHandler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], ca_path
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class TestCertificateVerificationSurvivesPinning:
    """The property the whole pinning design rests on."""

    async def test_correct_sni_connects_to_the_pinned_address(
        self, tls_server: tuple[int, Path]
    ) -> None:
        """Connecting by IP with the right SNI must succeed.

        If this fails, pinning has broken ordinary HTTPS and the design is
        unusable regardless of its security properties.
        """
        port, ca_path = tls_server
        async with _client_trusting(ca_path) as client:
            response = await client.get(
                f"https://127.0.0.1:{port}/",
                headers={"Host": SERVER_HOSTNAME},
                extensions={"sni_hostname": SERVER_HOSTNAME},
            )
        assert response.status_code == 200
        assert b"served over TLS" in response.content

    async def test_wrong_sni_is_refused(self, tls_server: tuple[int, Path]) -> None:
        """The critical one.

        Same address, same trusted CA, but the SNI names a host the
        certificate does not cover. If this *succeeds*, ``sni_hostname``
        is not driving verification and pinning has silently disabled
        hostname checking.
        """
        port, ca_path = tls_server
        async with _client_trusting(ca_path) as client:
            with pytest.raises((httpx.ConnectError, ssl.SSLCertVerificationError)) as info:
                await client.get(
                    f"https://127.0.0.1:{port}/",
                    headers={"Host": WRONG_HOSTNAME},
                    extensions={"sni_hostname": WRONG_HOSTNAME},
                )
        message = str(info.value).lower()
        assert "certificate" in message or "ssl" in message or "hostname" in message

    async def test_absent_sni_falls_back_to_the_address_and_is_refused(
        self, tls_server: tuple[int, Path]
    ) -> None:
        """Without the extension, verification targets the IP in the URL,
        which the certificate does not cover. This is why the extension is
        required rather than optional."""
        port, ca_path = tls_server
        async with _client_trusting(ca_path) as client:
            with pytest.raises((httpx.ConnectError, ssl.SSLCertVerificationError)):
                await client.get(f"https://127.0.0.1:{port}/", headers={"Host": SERVER_HOSTNAME})

    async def test_an_untrusted_issuer_is_still_refused(self, tls_server: tuple[int, Path]) -> None:
        """Pinning must not weaken chain validation either. With the system
        trust store instead of the test CA, the self-issued certificate has
        no valid chain and must be rejected even with correct SNI."""
        port, _ = tls_server
        async with httpx.AsyncClient(timeout=10) as client:
            with pytest.raises((httpx.ConnectError, ssl.SSLCertVerificationError)):
                await client.get(
                    f"https://127.0.0.1:{port}/",
                    headers={"Host": SERVER_HOSTNAME},
                    extensions={"sni_hostname": SERVER_HOSTNAME},
                )


class TestPinnedRequestProducesWhatTheServerNeeds:
    """The values SafeTarget hands to httpx are the ones proven above."""

    def test_pinned_request_supplies_address_host_and_sni(self) -> None:
        from agentic_research.retrieval.safety import SafeTarget

        target = SafeTarget(
            url=f"https://{SERVER_HOSTNAME}/doc",
            host=SERVER_HOSTNAME,
            port=443,
            addresses=("93.184.216.34",),
            scheme="https",
        )
        url, headers, extensions = target.pinned_request()

        assert url == "https://93.184.216.34/doc"
        assert headers["Host"] == SERVER_HOSTNAME
        assert extensions["sni_hostname"] == SERVER_HOSTNAME

    def test_plain_http_carries_no_sni(self) -> None:
        """SNI is a TLS concept; sending it for http would be meaningless."""
        from agentic_research.retrieval.safety import SafeTarget

        target = SafeTarget(
            url="http://plain.test/doc",
            host="plain.test",
            port=80,
            addresses=("93.184.216.34",),
            scheme="http",
        )
        _, headers, extensions = target.pinned_request()
        assert extensions == {}
        assert headers["Host"] == "plain.test"

    def test_this_file_never_disables_verification(self) -> None:
        """A guard against a future edit making every assertion vacuous.

        Parsed rather than grepped. The first version searched the source
        text for ``verify=False`` and matched this module's own docstring
        explaining why that is banned -- a false positive whose obvious
        remedy is deleting the guard. An AST walk sees only real keyword
        arguments, so prose about the rule cannot trip the rule.
        """
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        disabled = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.keyword)
            and node.arg == "verify"
            and isinstance(node.value, ast.Constant)
            and node.value.value is False
        ]
        assert not disabled, f"certificate verification disabled at line(s) {disabled}"
