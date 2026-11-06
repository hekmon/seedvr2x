"""The server's HTTP (DESIGN.md, The image server): the standard library's ThreadingHTTPServer, a
thread a connection, no new dependency. Python's documentation calls http.server not hardened for
production, which the reverse proxy in front covers (TLS, further authentication, the limits on
connections); the server still holds its own: the API key before any body is read, the body's
size before it is read, a deadline on a request's head and another on its body, a cap on the
connections served at once, and a cap on the pictures taken in at once (a body read and decoded
is host memory the plan counts, reservation.host_process).

POST and OPTIONS /upscale, GET /health and GET /metrics; every other path 404, every other
method 501 (OPTIONS * included)."""

import hmac
import logging
import select
import socket
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import admission
import answers
import metrics
import reservation
from service import Failed, Service, State, Unavailable, Unprocessable

logger = logging.getLogger("imgserver.web")

# The seconds a socket waits on one send or receive of the answer's: a client that stops reading
# its answer is cut after it (BaseHTTPRequestHandler's `timeout`, set on the socket). A choice,
# not measured: long enough for a slow link between two chunks, short enough that a client that
# has stopped reading does not hold a thread. The request's own reading has deadlines below.
TIMEOUT = 60.0
# The seconds a client has to send a request's line and headers, whole: a connection still short
# of them is shut down by a timer (a socket timeout restarts at each byte, so a byte every 59 s
# would hold a thread for as long as the client liked). A choice: a client on a LAN or behind a
# proxy sends a head in milliseconds, and 20 s still lets a slow mobile link through.
HEAD_SECONDS = 20.0
# The seconds a body has, beyond its length at BODY_RATE: BODY_SECONDS + length / BODY_RATE. A
# body that misses it is closed unanswered. Choices: 64 KiB/s is far under any LAN's or proxy's
# rate (a 50 MP picture's 416 MB limit then gets about 1 h 50 min, the upload of a very slow link
# and no more), and 20 s lets a small picture start.
BODY_SECONDS = 20.0
BODY_RATE = 64 * 1024
# What Allow says of /upscale (RFC 9110, 10.2.1): OPTIONS answers the formats in Accept-Post.
UPSCALE_ALLOW = "POST, OPTIONS"
# What Retry-After says for the queue full or the server starting or stopping, in seconds: a 4K
# picture's upscale on the RTX 5090, 14 s (DESIGN.md, Still images), rounded up.
RETRY_AFTER = 15
# What of a body left unread is read and dropped before the connection closes, at most, in bytes
# and seconds: closed with bytes unread, a socket sends a reset that can reach the client before
# the refusal it was sent (RFC 9112, 9.6), which a client sending a body too large must read.
# Choices: 4 MiB is a picture of a phone's size, which a client that sent its body at once sends
# before it reads, and 2 s bounds what a client that sends none can hold.
DRAIN_BYTES = 4 * 1024**2
DRAIN_SECONDS = 2.0
# The connections served at once beyond the pictures admitted and waiting: those reading a body,
# being refused, or asking /health and /metrics. Past them a connection is answered 503 at once,
# so that a flood of idle connections can't hold a thread each without bound. A choice, not
# measured: four times the default queue of 8, which leaves a probe a slot while every picture's
# connection is busy, and holds the threads (and their stacks) to a few dozen.
EXTRA_CONNECTIONS = 32
# The most digits a Content-Length may have: 2**64 has 20, and a body is far smaller than that,
# so a longer one is refused as no number before int() is asked to read it.
LENGTH_DIGITS = 20


class Server(ThreadingHTTPServer):
    """The HTTP server, its handler threads joined on close (ThreadingMixIn's block_on_close), so
    that the pictures running finish before the process exits."""

    daemon_threads = False
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        service: Service,
        api_key: str | None,
        connections: int,
    ) -> None:
        super().__init__(address, Handler)
        self.service = service
        self.api_key = None if api_key is None else api_key.encode()
        self.connections = threading.BoundedSemaphore(connections)
        # The pictures taken in at once: count running and queue waiting, the figure the host's
        # plan counts (reservation.host_process). A body is read and decoded only holding one.
        self.intake = threading.BoundedSemaphore(service.options.count + service.options.queue)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self.connections.acquire(blocking=False):
            try:
                body = b"the server is serving as many connections as it takes\n"
                request.sendall(
                    b"HTTP/1.0 503 Service Unavailable\r\nContent-Type: text/plain;"
                    b" charset=utf-8\r\nRetry-After: %d\r\nContent-Length: %d\r\n"
                    b"Connection: close\r\n\r\n%s" % (RETRY_AFTER, len(body), body)
                )
            except OSError:
                pass
            self.service.metrics.answered(HTTPStatus.SERVICE_UNAVAILABLE)
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.connections.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()


class Handler(BaseHTTPRequestHandler):
    """One request on one connection, HTTP/1.0: the connection closes after the answer."""

    server: Server  # pyright: ignore[reportIncompatibleVariableOverride]: the server it is made by
    server_version = "seedvr2x-imgserver"
    sys_version = ""
    timeout = TIMEOUT
    error_content_type = "text/plain; charset=utf-8"
    error_message_format = "%(code)d %(message)s\n"

    def send_response(self, code: int, message: str | None = None) -> None:
        super().send_response(code, message)
        self.server.service.metrics.answered(code)

    def setup(self) -> None:
        super().setup()
        self._read = False
        self._head_late = False
        self._head_timer = threading.Timer(HEAD_SECONDS, self._head_missed)
        self._head_timer.daemon = True
        self._head_timer.start()

    def _head_missed(self) -> None:
        """The head's timer: the connection is shut down, which ends the read it is blocked in."""
        self._head_late = True
        logger.info(
            "%s: the request's head missed its %.0f s deadline: connection closed",
            self.address_string(),
            HEAD_SECONDS,
        )
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def finish(self) -> None:
        self._head_timer.cancel()
        try:
            super().finish()
        finally:
            if not self._read:
                self._drain()

    def _drain(self) -> None:
        """A body left unread, read and dropped up to DRAIN_BYTES within DRAIN_SECONDS."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except (AttributeError, ValueError):
            return
        left, deadline = min(length, DRAIN_BYTES), time.monotonic() + DRAIN_SECONDS
        connection: socket.socket = self.connection
        try:
            while left > 0 and time.monotonic() < deadline:
                connection.settimeout(max(deadline - time.monotonic(), 0.01))
                chunk = connection.recv(min(left, 1 << 16))
                if not chunk:
                    break
                left -= len(chunk)
        except OSError:
            pass

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("%s %s", self.address_string(), format % args)

    # ---- Answers ----------------------------------------------------------------------------

    def _text(self, status: int, text: str, headers: dict[str, str] | None = None) -> None:
        body = (text.rstrip("\n") + "\n").encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self._write(body)

    def _write(self, body: bytes) -> None:
        try:
            self.wfile.write(body)
        except OSError as error:
            logger.debug("%s: the answer couldn't be sent: %s", self.address_string(), error)

    def _unauthorised(self) -> None:
        self._text(
            HTTPStatus.UNAUTHORIZED,
            "an API key is needed: Authorization: Bearer KEY",
            {"WWW-Authenticate": "Bearer"},
        )

    def _unavailable(self, error: Unavailable) -> None:
        headers = {"Retry-After": str(RETRY_AFTER)} if error.retry else {}
        self._text(HTTPStatus.SERVICE_UNAVAILABLE, str(error), headers)

    # ---- GET ------------------------------------------------------------------------------

    def do_GET(self) -> None:
        self._head_timer.cancel()
        path = self.path.split("?", 1)[0]
        service = self.server.service
        if path == "/health":
            state = service.health()
            status = HTTPStatus.OK if state is State.READY else HTTPStatus.SERVICE_UNAVAILABLE
            self._text(status, state.value)
        elif path == "/metrics":
            body = service.metrics.text(service.gauges).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", metrics.CONTENT_TYPE)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self._write(body)
        elif path == "/upscale":
            self._text(
                HTTPStatus.METHOD_NOT_ALLOWED,
                "/upscale takes POST and OPTIONS",
                {"Allow": UPSCALE_ALLOW},
            )
        else:
            self._text(
                HTTPStatus.NOT_FOUND,
                f"{path}: the server answers POST and OPTIONS /upscale, GET /health and"
                " GET /metrics",
            )

    # ---- OPTIONS /upscale ------------------------------------------------------------------

    def do_OPTIONS(self) -> None:
        """What /upscale takes: 204 with Allow and Accept-Post (the formats of the pictures, the
        W3C LDP 1.0's header, section 7.1), behind the API key as POST is. No Access-Control-*
        header: a browser's cross-origin calls are the reverse proxy's to allow, never the
        server's. Another path is a 404 as for GET; `OPTIONS *` stays unsupported (501), as it
        was before this method existed."""
        self._head_timer.cancel()
        target = self.path.split("?", 1)[0]
        if target == "*":
            self.send_error(HTTPStatus.NOT_IMPLEMENTED, "Unsupported method ('OPTIONS')")
            return
        if target != "/upscale":
            self._text(HTTPStatus.NOT_FOUND, f"{target}: the server answers /upscale")
            return
        if not self._authorised():
            self._unauthorised()
            return
        # No Content-Length and no body: a 204 carries none (RFC 9110, 8.6).
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Allow", UPSCALE_ALLOW)
        self.send_header("Accept-Post", answers.ACCEPT_POST)
        self.end_headers()

    # ---- POST /upscale ---------------------------------------------------------------------

    def do_POST(self) -> None:
        self._head_timer.cancel()
        if self.path.split("?", 1)[0] != "/upscale":
            self._text(HTTPStatus.NOT_FOUND, f"{self.path}: the server takes POST /upscale")
            return
        service = self.server.service
        # The key first, then the server's state, the headers and the body's size: nothing of a
        # body is read for a request that would be refused anyway.
        if not self._authorised():
            self._unauthorised()
            return
        try:
            service.taking()
        except Unavailable as error:
            self._unavailable(error)
            return
        try:
            settings = answers.settings(self.headers.items())
            formats = answers.formats(self.headers.get("Accept"))
        except answers.BadRequest as error:
            self._text(HTTPStatus.BAD_REQUEST, str(error))
            return
        except answers.NotAcceptable as error:
            self._text(HTTPStatus.NOT_ACCEPTABLE, str(error))
            return
        # A picture is read and decoded only holding a place among the count + queue the host's
        # plan counts; none free is the queue full, the body left unread (finish drains it).
        if not self.server.intake.acquire(blocking=False):
            self._text(
                HTTPStatus.SERVICE_UNAVAILABLE,
                f"the server's queue is full: {service.options.queue} pictures wait already",
                {"Retry-After": str(RETRY_AFTER)},
            )
            return
        try:
            self._upscale(service, settings, formats)
        finally:
            self.server.intake.release()

    def _upscale(
        self, service: Service, settings: answers.Settings, formats: answers.Formats
    ) -> None:
        data = self._body(reservation.body_limit(service.options.input_pixels))
        if data is None:
            return
        started = time.monotonic()
        try:
            prepared = service.prepare(data, settings, formats)
            del data
            made = service.run(prepared, self._gone)
        except Unprocessable as error:
            # A body that is no picture the server takes says which formats it does.
            headers = {"Accept-Post": answers.ACCEPT_POST} if error.kind == "not-a-picture" else {}
            self._text(HTTPStatus.UNPROCESSABLE_ENTITY, str(error), headers)
            return
        except answers.NotAcceptable as error:
            self._text(HTTPStatus.NOT_ACCEPTABLE, str(error))
            return
        except Unavailable as error:
            self._unavailable(error)
            return
        except admission.Gone:
            logger.info("%s: the client left while its picture waited", self.address_string())
            self.close_connection = True
            return
        except Failed as error:
            self._text(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
            return
        except Exception:
            logger.exception("a picture failed")
            self._text(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "The picture failed: the server's log says why; the server goes on.",
            )
            return
        logger.info(
            "%s: %dx%d to %dx%d (%s), seed %d, waited %.1f s, upscaled in %.1f s, %.1f s in all",
            self.address_string(),
            *made.source,
            *made.output,
            made.content_type,
            made.seed,
            made.waited,
            made.seconds,
            time.monotonic() - started,
        )
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", made.content_type)
        self.send_header("Content-Length", str(len(made.data)))
        self.send_header("Seedvr2x-Seed", str(made.seed))
        self.send_header("Seedvr2x-Source-Size", f"{made.source[0]}x{made.source[1]}")
        self.send_header("Seedvr2x-Output-Size", f"{made.output[0]}x{made.output[1]}")
        self.send_header(
            "Server-Timing",
            f'queue;dur={made.waited * 1000:.1f};desc="the wait in the queue",'
            f' upscale;dur={made.seconds * 1000:.1f};desc="the upscale"',
        )
        self.end_headers()
        self._write(made.data)

    def _authorised(self) -> bool:
        """Whether the request carries the server's key, `Authorization: Bearer KEY`, compared in
        constant time (hmac.compare_digest); always where the server has none."""
        key = self.server.api_key
        if key is None:
            return True
        given = self.headers.get_all("Authorization") or []
        if len(given) != 1:
            return False
        scheme, _, token = given[0].strip().partition(" ")
        if scheme.lower() != "bearer":
            return False
        return hmac.compare_digest(token.strip().encode(), key)

    def _body(self, limit: int) -> bytes | None:
        """The request's body, read whole, at most `limit` bytes: None once a refusal is answered
        (411 without a Content-Length, a chunked body included; 400 for one that isn't a number;
        413 over the limit, the body left unread and the connection closed) or once the client
        is gone with its body short."""
        if self.headers.get("Transfer-Encoding") is not None:
            self.close_connection = True
            self._text(
                HTTPStatus.LENGTH_REQUIRED,
                "a body with a Content-Length is needed, not a Transfer-Encoding",
            )
            return None
        lengths = self.headers.get_all("Content-Length") or []
        if not lengths:
            self._text(
                HTTPStatus.LENGTH_REQUIRED,
                "the picture's bytes are needed as the body, with a Content-Length",
            )
            return None
        text = lengths[0].strip()
        if (
            len(lengths) != 1
            or len(text) > LENGTH_DIGITS
            or not text.isascii()
            or not text.isdigit()
        ):
            self.close_connection = True
            self._text(HTTPStatus.BAD_REQUEST, "Content-Length: not one whole number")
            return None
        length = int(text)
        if length > limit:
            self.close_connection = True
            self._text(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"a body of {length:,} bytes, over the server's limit of {limit:,}: the largest"
                " picture it takes, --max-input pixels at 16 bits in four channels, and room"
                " for its metadata",
            )
            return None
        if length == 0:
            self._text(
                HTTPStatus.UNPROCESSABLE_ENTITY, "The body is empty: send the picture's bytes."
            )
            return None
        self._read = True
        data = self._read_body(length)
        if data is None:
            self.close_connection = True
        return data

    def _read_body(self, length: int) -> bytes | None:
        """`length` bytes of the body within BODY_SECONDS + length / BODY_RATE, in chunks of one
        receive each, the socket's timeout set to what is left of the deadline before each; None,
        logged, for a client gone or too slow: the connection closes unanswered."""
        deadline = time.monotonic() + BODY_SECONDS + length / BODY_RATE
        connection: socket.socket = self.connection
        chunks: list[bytes] = []
        got = 0
        try:
            while got < length:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError
                connection.settimeout(left)
                # read1: one receive at most, so that the timeout bounds the chunk, not the body.
                chunk = self.rfile.read1(min(length - got, 1 << 20))
                if not chunk:
                    logger.info(
                        "%s: the body ended after %d of its %d bytes",
                        self.address_string(),
                        got,
                        length,
                    )
                    return None
                chunks.append(chunk)
                got += len(chunk)
        except TimeoutError:
            logger.info(
                "%s: the body missed its deadline, %d of its %d bytes after %.0f s: connection"
                " closed",
                self.address_string(),
                got,
                length,
                BODY_SECONDS + length / BODY_RATE,
            )
            return None
        except OSError as error:
            logger.info("%s: the body couldn't be read: %s", self.address_string(), error)
            return None
        finally:
            try:
                connection.settimeout(TIMEOUT)
            except OSError:
                pass
        # One copy more than the bytes read, for as long as the join takes.
        return chunks[0] if len(chunks) == 1 else b"".join(chunks)

    def _gone(self) -> bool:
        """Whether the client has closed its connection while its picture waits: its socket
        readable with nothing to read."""
        connection: socket.socket = self.connection
        try:
            readable, _, _ = select.select([connection], [], [], 0)
            if not readable:
                return False
            return connection.recv(1, socket.MSG_PEEK) == b""
        except (OSError, ValueError):
            return True
