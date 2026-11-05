#!/usr/bin/env python3
"""Time a ComfyUI picture workflow node by node, over several prompts in a row.

  comfy_nodes.py --python PY --comfy-dir DIR --run-dir RUN --template T.json --input PIC \
      --sub dit=NAME --sub vae=NAME --sub scale=2 --seed N [--prompts 4] [--pidfile FILE] \
      [-- EXTRA COMFYUI ARGS]

Starts ComfyUI (PY DIR/main.py) on 127.0.0.1 with its output, input, temp and user directories
under RUN, waits until it answers, posts the prompts one after the other (the first with the seed,
prompt i with seed + i - 1, so that what ComfyUI keeps between prompts, its loaded models and the
outputs it may cache, shows in the later ones) and stops ComfyUI (SIGINT, SIGTERM after 20 s).
With --phase-peaks ComfyUI is started through comfy_wrap.py, which times the sampler and the
VAE's calls and logs their memory peaks (RUN/phases.jsonl, read into the result).

Why a websocket: the HTTP history gives a prompt's total only. ComfyUI sends an `executing`
message when a node starts and `executing` with a null node when the prompt ends, none of them
timestamped, so each is stamped on arrival here (monotonic clock): a node ends where the next one
starts, the last where the prompt ends. A node ComfyUI takes from its cache sends no message and
shows as cached. The socket is written with the standard library alone (RFC 6455: a text or binary
frame is read, a ping answered, nothing else sent), so that nothing is installed in ComfyUI's
environment.

The template is an API-format prompt in which a string that is exactly "{{name}}" is replaced by
the value of --sub name=value (a number stays a number), and "{{name}}" inside a longer string as
text; `input` (the picture's file name in RUN/inputs, copied there: ComfyUI refuses a link that
leaves its input directory), `seed` and `prefix` (p1, p2, ...) are set here. A placeholder left
without a value is an error. The substitution follows promptlib.py of comfy-study (this
repository's own script), copied; nothing of ComfyUI's code is copied or imported.

Result: RUN/nodes.json (the nodes of each prompt with their start and end, ComfyUI's "Prompt
executed in" lines of its log, the wall time of each prompt, the output files and sizes) and one
table per prompt on the standard output, then the median of the warm prompts per node. ComfyUI's
log is RUN/comfy.log.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import re
import shutil
import signal
import socket
import statistics
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

PLACE = re.compile(r"\{\{(\w+)\}\}")
CLIENT_ID = "comfy-nodes"


def substitute(value, subs: dict):
    """The template with its placeholders replaced (promptlib.py of comfy-study, copied)."""
    if isinstance(value, dict):
        return {k: substitute(v, subs) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute(v, subs) for v in value]
    if isinstance(value, str):
        whole = PLACE.fullmatch(value)
        if whole:
            if whole.group(1) not in subs:
                raise KeyError(f"placeholder {{{{{whole.group(1)}}}}} has no substitution")
            return subs[whole.group(1)]

        def one(m):
            if m.group(1) not in subs:
                raise KeyError(f"placeholder {{{{{m.group(1)}}}}} has no substitution")
            return str(subs[m.group(1)])

        return PLACE.sub(one, value)
    return value


def number_or_text(text: str):
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def http(method: str, url: str, body: dict | None = None, timeout: float = 30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            return response.status, (json.loads(raw) if raw[:1] in (b"{", b"[") else raw)
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            return error.code, json.loads(raw)
        except ValueError:
            return error.code, raw.decode(errors="replace")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Failure(Exception):
    """A run that cannot go on: its text is the status the result records."""


class Socket:
    """A WebSocket client reading ComfyUI's messages, each stamped when it arrives."""

    def __init__(self, port: int, client_id: str):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall(
            (
                f"GET /ws?clientId={client_id} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                f"Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                "Sec-WebSocket-Version: 13\r\n\r\n"
            ).encode()
        )
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise OSError("the websocket handshake was closed")
            head += chunk
        head, self.buffer = head.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            raise OSError(f"the websocket handshake was refused: {head[:200]!r}")
        self.sock.settimeout(None)

    def _read(self, count: int) -> bytes:
        while len(self.buffer) < count:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise EOFError
            self.buffer += chunk
        out, self.buffer = self.buffer[:count], self.buffer[count:]
        return out

    def _pong(self, payload: bytes) -> None:
        mask = os.urandom(4)  # a client's frames are masked (RFC 6455, 5.3)
        body = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(bytes([0x8A, 0x80 | len(payload)]) + mask + body)

    def message(self) -> tuple[int, bytes]:
        """The next text (1) or binary (2) message, fragments joined; EOFError at the end."""
        kind, parts = 0, []
        while True:
            first, second = self._read(2)
            opcode, final = first & 0x0F, bool(first & 0x80)
            length = second & 0x7F
            if length == 126:
                (length,) = struct.unpack(">H", self._read(2))
            elif length == 127:
                (length,) = struct.unpack(">Q", self._read(8))
            payload = self._read(length)  # the server's frames are not masked
            if opcode == 0x9:
                self._pong(payload[:125])
            elif opcode == 0x8:
                raise EOFError
            elif opcode in (1, 2):
                kind, parts = opcode, [payload]
                if final:
                    return kind, payload
            elif opcode == 0 and parts:
                parts.append(payload)
                if final:
                    return kind, b"".join(parts)


class Events:
    """Everything the socket says, in order: (monotonic seconds, wall seconds, message)."""

    def __init__(self, port: int):
        self.items: list[tuple[float, float, dict]] = []
        self.lock = threading.Lock()
        self.socket = Socket(port, CLIENT_ID)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            while True:
                kind, payload = self.socket.message()
                if kind != 1:
                    continue  # binary: previews, not used here
                stamp = (time.monotonic(), time.time())
                try:
                    message = json.loads(payload)
                except ValueError:
                    continue
                with self.lock:
                    self.items.append((*stamp, message))
        except (EOFError, OSError):
            pass

    def snapshot(self) -> list[tuple[float, float, dict]]:
        with self.lock:
            return list(self.items)


def prompt_events(events: list, prompt_id: str) -> list:
    return [e for e in events if e[2].get("data", {}).get("prompt_id") == prompt_id]


def finished(events: list, prompt_id: str) -> str | None:
    for _, _, message in prompt_events(events, prompt_id):
        data = message["data"]
        if message["type"] == "executing" and data.get("node") is None:
            return "done"
        if message["type"] in ("execution_error", "execution_interrupted"):
            return message["type"]
    return None


def nodes_of(events: list, prompt_id: str, prompt: dict, posted: float) -> dict:
    """The prompt's nodes from the stamps: start, end, seconds, class; cached ones apart."""
    mine = prompt_events(events, prompt_id)
    starts: list[tuple[str, float]] = []
    end = None
    cached: list[str] = []
    for stamp, _, message in mine:
        data = message["data"]
        if message["type"] == "execution_cached":
            cached = [str(n) for n in data.get("nodes", [])]
        elif message["type"] == "executing":
            if data.get("node") is None:
                end = stamp
            else:
                starts.append((str(data["node"]), stamp))
    nodes = []
    for index, (node, start) in enumerate(starts):
        stop = starts[index + 1][1] if index + 1 < len(starts) else end
        nodes.append(
            {
                "node": node,
                "class": prompt[node]["class_type"],
                "start_s": start - posted,
                "end_s": None if stop is None else stop - posted,
                "seconds": None if stop is None else stop - start,
            }
        )
    return {"nodes": nodes, "cached": cached, "end_s": None if end is None else end - posted}


def stop_process(proc: subprocess.Popen, grace: float = 20.0) -> str:
    if proc.poll() is not None:
        return "already gone"
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(grace)
        return "SIGINT"
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(grace)
            return f"SIGTERM after {grace:.0f} s"
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return f"SIGKILL after {2 * grace:.0f} s"


def outputs_of(entry: dict, out_dir: str) -> list[dict]:
    files = []
    for outs in entry.get("outputs", {}).values():
        for items in outs.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and "filename" in item:
                    path = os.path.join(out_dir, item.get("subfolder", ""), item["filename"])
                    size = os.path.getsize(path) if os.path.exists(path) else None
                    files.append({"path": path, "bytes": size})
    return files


def print_table(title: str, record: dict) -> None:
    print(f"\n{title}")
    for node in record["nodes"]:
        seconds = "-" if node["seconds"] is None else f"{node['seconds']:8.3f}"
        print(f"  {node['node']:>3}  {node['class']:<24} {seconds}")
    if record["cached"]:
        print(f"  cached nodes: {', '.join(record['cached'])}")
    print(f"  wall {record['wall_s']:.3f} s; ComfyUI says {record.get('executed_s')}")


def main() -> int:
    argv = sys.argv[1:]
    extra: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1 :]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--python", required=True, help="the interpreter of ComfyUI's environment")
    parser.add_argument("--comfy-dir", required=True, help="ComfyUI's directory (main.py)")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--template", required=True, help="API-format prompt with {{name}} places")
    parser.add_argument("--input", required=True, help="the picture")
    parser.add_argument("--sub", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--pidfile", help="where to write ComfyUI's pid (a watcher may stop it)")
    parser.add_argument(
        "--phase-peaks",
        action="store_true",
        help="start ComfyUI through comfy_wrap.py: the sampler and VAE calls timed, peaks logged",
    )
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--prompt-timeout", type=float, default=1800)
    args = parser.parse_args(argv)

    run = os.path.abspath(args.run_dir)
    for sub in ("inputs", "output", "temp", "user"):
        os.makedirs(f"{run}/{sub}", exist_ok=True)
    name = os.path.basename(args.input)
    shutil.copyfile(args.input, f"{run}/inputs/{name}")
    subs = {}
    for item in args.sub:
        key, _, value = item.partition("=")
        subs[key] = number_or_text(value)
    with open(args.template) as handle:
        template = json.load(handle)
    prompts = []
    for number in range(1, args.prompts + 1):
        values = dict(subs, seed=args.seed + number - 1, input=name, prefix=f"p{number}")
        prompt = substitute(copy.deepcopy(template), values)
        with open(f"{run}/prompt{number}.json", "w") as handle:
            json.dump(prompt, handle, indent=1)
        prompts.append(prompt)

    port = args.port or free_port()
    base = f"http://127.0.0.1:{port}"
    command = [
        args.python,
        *(
            [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "comfy_wrap.py"),
                args.comfy_dir,
            ]
            if args.phase_peaks
            else [f"{args.comfy_dir}/main.py"]
        ),
        "--listen",
        "127.0.0.1",
        "--port",
        str(port),
        "--disable-auto-launch",
        "--output-directory",
        f"{run}/output",
        "--input-directory",
        f"{run}/inputs",
        "--temp-directory",
        f"{run}/temp",
        "--user-directory",
        f"{run}/user",
        *extra,
    ]
    env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1")
    phases = f"{run}/phases.jsonl"
    if args.phase_peaks:
        if os.path.exists(phases):
            os.remove(phases)
        env["COMFY_PHASE_LOG"] = phases
    result: dict = {"command": command, "port": port, "prompts": []}
    log = open(f"{run}/comfy.log", "wb")
    t0 = time.monotonic()
    proc = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=args.comfy_dir,
        start_new_session=True,
    )
    if args.pidfile:
        with open(args.pidfile, "w") as handle:
            handle.write(str(proc.pid))
    status = "ran"
    try:
        ready = False
        while time.monotonic() - t0 < args.startup_timeout and proc.poll() is None:
            try:
                code, _ = http("GET", f"{base}/system_stats", timeout=5)
                if code == 200:
                    ready = True
                    break
            except OSError:
                pass
            time.sleep(0.5)
        result["start_to_ready_s"] = time.monotonic() - t0
        if not ready:
            raise Failure(f"ComfyUI was not ready (exit status {proc.poll()})")
        events = Events(port)
        for number, prompt in enumerate(prompts, 1):
            posted = time.monotonic()
            code, body = http("POST", f"{base}/prompt", {"prompt": prompt, "client_id": CLIENT_ID})
            if code != 200:
                raise Failure(f"POST /prompt {code}: {json.dumps(body)[:1500]}")
            pid = body["prompt_id"]
            while True:
                ended = finished(events.snapshot(), pid)
                if ended:
                    break
                if proc.poll() is not None:
                    raise Failure(f"ComfyUI died during prompt {number}")
                if time.monotonic() - posted > args.prompt_timeout:
                    http("POST", f"{base}/interrupt", {})
                    raise Failure(f"prompt {number} timed out")
                time.sleep(0.05)
            wall = time.monotonic() - posted
            time.sleep(0.5)  # the last messages and the log line may trail the end
            snapshot = events.snapshot()
            record = {"n": number, "prompt_id": pid, "seed": args.seed + number - 1, "wall_s": wall}
            record["state"] = ended
            record.update(nodes_of(snapshot, pid, prompt, posted))
            _, history = http("GET", f"{base}/history/{pid}")
            entry = history.get(pid, {}) if isinstance(history, dict) else {}
            record["outputs"] = outputs_of(entry, f"{run}/output")
            record["messages"] = [
                {"type": m["type"], "at_s": s - posted, "data": m["data"]}
                for s, _, m in prompt_events(snapshot, pid)
                if m["type"] != "executing"
            ]
            result["prompts"].append(record)
            if ended != "done":
                status = ended
                break
    except Failure as failure:
        status = str(failure)
        print(f"FAILED: {status}", file=sys.stderr)
    finally:
        result["stop"] = stop_process(proc)
        log.close()
    text = open(f"{run}/comfy.log", errors="replace").read()
    executed = [float(x) for x in re.findall(r"Prompt executed in ([0-9.]+) seconds", text)]
    for record, seconds in zip(result["prompts"], executed, strict=False):
        record["executed_s"] = seconds
    result["status"] = status
    result["prompt_executed_s"] = executed
    result["dynamic_vram_enabled"] = "DynamicVRAM support detected and enabled" in text
    result["attention_lines"] = [
        line for line in text.splitlines() if "ttention" in line and "sage" not in line.lower()
    ][:6]
    if args.phase_peaks and os.path.exists(phases):
        with open(phases) as handle:
            result["phases"] = [json.loads(line) for line in handle if line.strip()]
    with open(f"{run}/nodes.json", "w") as handle:
        json.dump(result, handle, indent=1)

    print(f"ready after {result['start_to_ready_s']:.1f} s; stop: {result['stop']}")
    for record in result["prompts"]:
        title = f"prompt {record['n']} ({'cold' if record['n'] == 1 else 'warm'})"
        print_table(title, record)
    for phase in result.get("phases", []):
        print(
            f"  {phase['function']:<18} depth {phase['depth']} {phase['seconds']:8.3f} s  "
            f"peak alloc {phase['peak_allocated_gib']:.2f} "
            f"reserved {phase['peak_reserved_gib']:.2f} "
            f"GiB; alloc before {phase['allocated_before_gib']:.2f}; card used "
            f"{phase['device_used_before_gib']:.2f} -> max {phase['device_used_max_gib']:.2f} GiB"
        )
    warm = result["prompts"][1:]
    if len(warm) > 1:
        print("\nwarm prompts, median seconds per node")
        for index, node in enumerate(warm[0]["nodes"]):
            values = [
                p["nodes"][index]["seconds"]
                for p in warm
                if len(p["nodes"]) > index and p["nodes"][index]["node"] == node["node"]
                if p["nodes"][index]["seconds"] is not None
            ]
            if values:
                print(f"  {node['node']:>3}  {node['class']:<24} {statistics.median(values):8.3f}")
        walls = [p["wall_s"] for p in warm]
        print(f"  wall median {statistics.median(walls):.3f} s")
    return 0 if status == "ran" else 1


if __name__ == "__main__":
    sys.exit(main())
