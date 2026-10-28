# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Private, same-host shared KV pool control plane. No tensor payloads over RPC."""

from __future__ import annotations

import argparse
import collections
import json
import os
import socket
import socketserver
import threading
import uuid
from pathlib import Path

MAX_FRAME = 16 * 1024 * 1024


class PoolState:
    def __init__(self, root: str, backing_file: str | None = None):
        self.root = Path(root)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.epoch = uuid.uuid4().hex
        self.namespace = None
        self.capacity = 0
        self.free = collections.deque()
        self.records = {}
        self.index = {}
        self.owners = collections.defaultdict(set)
        self.lru = collections.OrderedDict()
        self.generations = {}
        self.layout = None
        self.contracts = {}
        self.file = (
            Path(backing_file) if backing_file else self.root / (self.epoch + ".kv")
        )
        self.version = 0
        self.stats = collections.Counter()
        self.lock = threading.RLock()

    def _drop(self, slot):
        r = self.records.pop(slot)
        assert not r["refs"]
        for key in r["aliases"]:
            if self.index.get(key) == slot:
                del self.index[key]
        self.lru.pop(slot, None)
        self.free.append(slot)

    def disconnect(self, owner):
        # A disconnected scheduler does NOT prove its CUDA worker stopped DMA.
        # Never reclaim its held slots based on a timeout or connection closure.
        with self.lock:
            for slot in self.owners.pop(owner, ()):
                r = self.records[slot]
                if r["refs"].get(owner):
                    self.lru.pop(slot, None)
                    for key in r["aliases"]:
                        if self.index.get(key) == slot:
                            del self.index[key]
                    r["state"] = "quarantined"
                    self.stats["quarantined"] += 1

    def dispatch(self, owner, op, a):
        with self.lock:
            if op == "hello":
                ns, capacity = a
                if capacity < 2:
                    raise ValueError("Pool must have at least two slots")
                if self.namespace is None:
                    self.namespace, self.capacity = ns, capacity
                    self.free = collections.deque(range(1, capacity))
                if (ns, capacity) != (self.namespace, self.capacity):
                    raise ValueError(
                        f"Shared KV namespace/capacity mismatch: "
                        f"expected {self.namespace[:12]}/{self.capacity}, "
                        f"got {ns[:12]}/{capacity}"
                    )
                return {"epoch": self.epoch, "capacity": self.capacity}
            if self.namespace is None:
                raise ValueError("hello required")
            if op == "contract":
                role, contract = a
                if role not in ("scheduler", "worker"):
                    raise ValueError("Unknown cache client role")
                if role in self.contracts and self.contracts[role] != contract:
                    raise ValueError(f"Incompatible {role} KV contract")
                self.contracts[role] = contract
                (self.root / (self.epoch + ".contracts.json")).write_text(
                    json.dumps(self.contracts, indent=2)
                )
                return True
            if op == "attach":
                layout = a[0]
                if self.layout is None:
                    size = sum(n for _, n in layout) * self.capacity
                    # New epoch uses a fresh inode. Old mappings are never truncated.
                    if self.file.exists():
                        if self.file.stat().st_size != size:
                            raise ValueError("Preallocated backing size mismatch")
                    else:
                        with self.file.open("xb") as f:
                            f.truncate(size)
                    os.chmod(self.file, 0o600)
                    self.layout = layout
                if self.layout != layout:
                    raise ValueError("Shared KV tensor layout mismatch")
                return {"file": str(self.file), "layout": self.layout}
            if op == "reserve":
                n = min(int(a[0]), 4096)
                out = []
                for _ in range(n):
                    if not self.free:
                        if not self.lru:
                            break
                        victim = next(iter(self.lru))
                        self._drop(victim)
                        self.stats["evicted"] += 1
                    slot = self.free.popleft()
                    generation = self.generations.get(slot, 0) + 1
                    self.generations[slot] = generation
                    self.records[slot] = {
                        "generation": generation,
                        "state": "writing",
                        "refs": {owner: 1},
                        "aliases": {},
                        "version": self.version,
                    }
                    self.owners[owner].add(slot)
                    out.append([slot, generation])
                self.stats["reserved"] += len(out)
                return out
            if op in ("lookup", "acquire"):
                found = []
                held = set()
                for key in a[0]:
                    slot = self.index.get(key)
                    r = self.records.get(slot)
                    if r is None or r["state"] != "ready":
                        continue
                    if op == "acquire" and slot not in held:
                        r["refs"][owner] = r["refs"].get(owner, 0) + 1
                        self.lru.pop(slot, None)
                        held.add(slot)
                        self.owners[owner].add(slot)
                    primary = next(iter(r["aliases"]))
                    found.append(
                        [key, slot, r["generation"], primary, r["aliases"][primary]]
                    )
                self.stats[op + "_hits"] += len(found)
                return found
            if op == "finish":
                # Commit a completed D2H, or release a completed H2D/temp lease.
                for slot, generation, aliases in a[0]:
                    r = self.records.get(slot)
                    if (
                        r is None
                        or r["generation"] != generation
                        or not r["refs"].get(owner)
                    ):
                        raise ValueError("Stale or unowned shared KV reference")
                    if r["state"] == "writing":
                        if len(r["refs"]) != 1:
                            raise AssertionError("Uncommitted object has readers")
                        if aliases and r["version"] == self.version:
                            primary = next(iter(aliases))
                            if primary in self.index:
                                self.stats["deduplicated"] += 1
                            else:
                                r["aliases"] = aliases
                                r["state"] = "ready"
                                for key in aliases:
                                    self.index[key] = slot
                                self.stats["committed"] += 1
                    r["refs"][owner] -= 1
                    if r["refs"][owner] == 0:
                        del r["refs"][owner]
                        self.owners[owner].discard(slot)
                    if not r["refs"]:
                        if r["state"] == "ready" and r["version"] == self.version:
                            self.lru[slot] = None
                        elif r["state"] != "quarantined":
                            self._drop(slot)
                return True
            if op == "flush":
                self.version += 1
                self.index.clear()
                for slot, r in list(self.records.items()):
                    if not r["refs"] and r["state"] != "quarantined":
                        self._drop(slot)
                return True
            if op == "stats":
                return {
                    **self.stats,
                    "slots": self.capacity,
                    "free": len(self.free),
                    "evictable": len(self.lru),
                    "held": len(self.records) - len(self.lru),
                    "objects": len(self.records),
                    "keys": len(self.index),
                }
            raise ValueError("Unknown operation")


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        owner = uuid.uuid4().hex
        try:
            while line := self.rfile.readline(MAX_FRAME + 1):
                if len(line) > MAX_FRAME:
                    break
                try:
                    m = json.loads(line)
                    result = self.server.state.dispatch(
                        owner, m["op"], m.get("args", [])
                    )
                    reply = {"result": result}
                except Exception as e:
                    reply = {"error": f"{type(e).__name__}: {e}"}
                self.wfile.write(
                    json.dumps(reply, separators=(",", ":")).encode() + b"\n"
                )
                self.wfile.flush()
        finally:
            self.server.state.disconnect(owner)


class PoolServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, address, root, backing_file=None):
        self.state = PoolState(root, backing_file)
        super().__init__(address, Handler)
        os.chmod(address, 0o600)


class PoolClient:
    def __init__(self, address, namespace, capacity):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(5)
        self.sock.connect(address)
        self.stream = self.sock.makefile("rwb")
        self.lock = threading.Lock()
        self.epoch = self.call("hello", namespace, capacity)["epoch"]

    def call(self, op, *args):
        with self.lock:
            data = (
                json.dumps({"op": op, "args": args}, separators=(",", ":")).encode()
                + b"\n"
            )
            if len(data) > MAX_FRAME:
                raise ValueError("Shared KV batch exceeds frame limit")
            self.stream.write(data)
            self.stream.flush()
            line = self.stream.readline(MAX_FRAME + 1)
            if not line or len(line) > MAX_FRAME:
                raise RuntimeError(
                    "Shared KV manager disconnected; drain clients before restarting"
                )
            r = json.loads(line)
            if "error" in r:
                raise RuntimeError(r["error"])
            return r["result"]

    def close(self):
        self.stream.close()
        self.sock.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--socket", required=True)
    p.add_argument("--root", required=True)
    p.add_argument(
        "--backing-file",
        help="Offline-reused backing; all prior clients MUST be stopped",
    )
    a = p.parse_args()
    # Do not unlink an existing socket: clients may still own in-flight DMA.
    with PoolServer(a.socket, a.root, a.backing_file) as server:
        server.serve_forever()
