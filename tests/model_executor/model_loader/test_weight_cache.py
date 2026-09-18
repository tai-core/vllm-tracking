# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the IPC weight cache loader.

A cold start (no weight cache daemon, so the loader falls back to disk) and
warm restarts (weights mapped from the daemon via CUDA IPC) must both serve
identical outputs.
"""

import glob
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import pytest

from vllm import SamplingParams
from vllm.platforms import current_platform

MODEL = "Qwen/Qwen3.5-0.8B"
PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
]
DAEMON_TIMEOUT_S = 600


class WeightCacheDaemon:
    """Context manager running the real weight cache daemon as a subprocess."""

    _READY_MARKER = "Weight cache daemon READY"

    def __init__(self, model: str, tp_size: int):
        self.tp_size = tp_size
        # Short base path: Unix socket paths are limited to ~107 characters.
        self.socket_dir = tempfile.mkdtemp(prefix="vllm_ipc_")
        self._cmd = [
            sys.executable,
            "-m",
            "vllm.model_executor.model_loader.weight_cache.daemon",
            "--model",
            model,
            "--tensor-parallel-size",
            str(tp_size),
            "--weight-cache-socket-dir",
            self.socket_dir,
            "--enforce-eager",
        ]
        self._proc: subprocess.Popen | None = None
        self._lines: list[str] = []

    def __enter__(self) -> "WeightCacheDaemon":
        self._proc = subprocess.Popen(
            self._cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        try:
            self._wait_ready(DAEMON_TIMEOUT_S)
        except Exception:
            self._stop()
            raise
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop()

    def _drain_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        for line in self._proc.stderr:
            self._lines.append(line)

    def _logs(self) -> str:
        return "".join(self._lines)

    def _wait_ready(self, timeout_s: float) -> None:
        assert self._proc is not None
        # Each rank binds its socket only once the model is fully cached.
        # Poll for the socket files rather than a log line: model loading can
        # pull in JIT compilers that swap the process's stderr and swallow
        # everything logged afterwards, making log-based readiness flaky.
        pattern = os.path.join(self.socket_dir, "vllm_weight_cache_*.sock")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"Weight cache daemon exited with {self._proc.returncode}:\n"
                    f"{self._logs()}"
                )
            if len(glob.glob(pattern)) >= self.tp_size:
                return
            time.sleep(1.0)
        raise TimeoutError(
            f"Weight cache daemon not ready after {timeout_s}s:\n{self._logs()}"
        )

    def _stop(self) -> None:
        assert self._proc is not None
        self._proc.terminate()
        try:
            self._proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        shutil.rmtree(self.socket_dir, ignore_errors=True)


def generate(vllm_runner, socket_dir: str | None, fallback: bool, **kwargs):
    extra_config = (
        {} if socket_dir is None else {"socket_dir": socket_dir, "fallback": fallback}
    )
    with vllm_runner(
        MODEL,
        load_format="auto" if socket_dir is None else "ipc_cache",
        model_loader_extra_config=extra_config,
        # Greedy outputs are compared across engine restarts; cap the batch
        # size so scheduling cannot change the results.
        max_num_seqs=1,
        **kwargs,
    ) as llm:
        sampling_params = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
        return llm.generate(PROMPTS, sampling_params)


def test_ipc_cache_cold_start_and_warm_restart(vllm_runner):
    """Cold start falls back to disk; warm restarts load weights via CUDA IPC.

    All runs must produce outputs identical to a default-loader baseline. The
    warm runs disable the disk fallback, so they only pass if the weights
    really came from the daemon.
    """
    if not current_platform.is_cuda_alike():
        pytest.skip("Weight cache IPC sharing requires CUDA or ROCm")
    llm_kwargs = dict(
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        # Qwen3.5 is a hybrid mamba model; its mamba cache mode requires
        # chunked prefill.
        enable_chunked_prefill=True,
    )

    # Baseline: plain disk loading with the default loader.
    baseline_outputs = generate(vllm_runner, None, fallback=True, **llm_kwargs)
    assert all(text for _, texts in baseline_outputs for text in texts)

    # Cold start: no daemon is serving, so the loader falls back to disk.
    with tempfile.TemporaryDirectory(prefix="vllm_ipc_empty_") as empty_socket_dir:
        cold_outputs = generate(
            vllm_runner, empty_socket_dir, fallback=True, **llm_kwargs
        )

    with WeightCacheDaemon(MODEL, tp_size=1) as daemon:
        warm_outputs = generate(
            vllm_runner, daemon.socket_dir, fallback=False, **llm_kwargs
        )
        # Warm restart: a second engine lifetime against the same daemon.
        restart_outputs = generate(
            vllm_runner, daemon.socket_dir, fallback=False, **llm_kwargs
        )

    assert cold_outputs == baseline_outputs
    assert warm_outputs == baseline_outputs
    assert restart_outputs == baseline_outputs


def test_get_state_exports_fresh_entries_for_each_client(monkeypatch):
    """Each engine lifetime receives a fresh reduction, not cached IPC handles."""
    import torch

    from vllm.model_executor.model_loader.weight_cache import daemon, protocol

    owner = object.__new__(daemon.WeightCacheDaemon)
    owner.tp_rank = 0
    owner.role = "target"
    owner.model = torch.nn.Linear(2, 2)
    owner.cache_config = protocol.WeightCacheKey(
        checkpoint="synthetic",
        model_arch="Linear",
        tp_size=1,
        tp_rank=0,
        dtype="float32",
        quantization=None,
        quant_config_hash="none",
        revision=None,
        vllm_version="test",
    )
    replies = []
    monkeypatch.setattr(daemon, "get_current_device_uuid", lambda: "test-gpu")
    monkeypatch.setattr(daemon, "send_msg", lambda _conn, reply: replies.append(reply))
    request = {"cache_config": owner.cache_config}
    owner._handle_get_state(None, request)
    owner._handle_get_state(None, request)
    first, second = replies
    assert first["status"] == second["status"] == "ok"
    assert first["entries"] is not second["entries"]
    for name in first["entries"]:
        assert first["entries"][name] is not second["entries"][name]
    owner.model = None
    owner._handle_get_state(None, request)
    assert replies[-1] == {"status": "error", "message": "Weights were released"}


def test_readiness_uses_configured_tp_size(monkeypatch):
    """The UUID socket check retains its TP count without the FP4 test suite."""
    from pathlib import Path
    from types import SimpleNamespace

    daemon = WeightCacheDaemon("synthetic", tp_size=2)
    try:
        for rank in range(2):
            Path(daemon.socket_dir, f"vllm_weight_cache_uuid{rank}.sock").touch()
        monkeypatch.setattr(daemon, "_proc", SimpleNamespace(poll=lambda: None))
        daemon._wait_ready(timeout_s=0.1)
    finally:
        shutil.rmtree(daemon.socket_dir)


def test_create_model_does_not_require_new_online_quant_logging(monkeypatch):
    """Backported creation works with the original v0.29 configuration shape."""
    from types import SimpleNamespace

    import torch

    from vllm.model_executor.model_loader import base_loader

    class SyntheticLoader(base_loader.BaseModelLoader):
        def download_model(self, model_config):
            pass

        def load_weights(self, model, model_config):
            pass

    expected = torch.nn.Linear(2, 2)
    monkeypatch.setattr(base_loader, "initialize_model", lambda **_kwargs: expected)
    monkeypatch.setattr(base_loader.envs, "VLLM_LOG_MODEL_INSPECTION", False)
    loader = SyntheticLoader(SimpleNamespace())
    assert loader.create_model(SimpleNamespace(), SimpleNamespace()) is expected


def test_daemon_initializes_constructor_workspace(monkeypatch):
    """Standalone model construction can request scratch without a GPUWorker."""
    from contextlib import nullcontext
    from types import SimpleNamespace

    import torch

    from vllm.model_executor.model_loader.weight_cache import daemon
    from vllm.v1.worker import workspace

    monkeypatch.setattr(workspace, "_manager", None)
    monkeypatch.setattr(daemon.current_platform, "device_type", "cpu")
    monkeypatch.setattr(torch.accelerator, "set_device_index", lambda _rank: None)
    monkeypatch.setattr(daemon, "init_distributed_environment", lambda **_kwargs: None)
    monkeypatch.setattr(
        daemon, "ensure_model_parallel_initialized", lambda *_args: None
    )
    monkeypatch.setattr(
        daemon, "set_current_vllm_config", lambda _config: nullcontext()
    )
    expected = torch.nn.Linear(2, 2)

    def construct(_config, _model_config):
        scratch = workspace.current_workspace_manager().get_simultaneous(
            ((4,), torch.float32)
        )[0]
        assert scratch.device.type == "cpu" and scratch.numel() == 4
        return expected

    monkeypatch.setattr(daemon, "get_daemon_model", construct)
    owner = object.__new__(daemon.WeightCacheDaemon)
    owner.cache_config = SimpleNamespace(tp_size=1)
    owner.topology = daemon.resolve_cache_topology(_ep_parallel(1, 1), 0)
    owner.global_rank = 0
    owner.tp_rank = 0
    owner.local_rank = 0
    owner.role = "target"
    owner.model_config = SimpleNamespace()
    owner.distributed_init_method = "unused"
    owner.vllm_config = SimpleNamespace(parallel_config=_ep_parallel(1, 1))
    owner.load_model()
    assert owner.model is expected


def test_spawn_rank_loads_plugins_before_constructing_daemon(monkeypatch):
    """A fresh rank must see general plugin registrations before model setup."""
    from types import SimpleNamespace

    from vllm.model_executor.model_loader.weight_cache import daemon

    state = {"plugins_ready": False}
    ready_ranks: list[tuple[str, int]] = []

    def load_plugins():
        state["plugins_ready"] = True

    class SyntheticDaemon:
        def __init__(self, *_args):
            assert state["plugins_ready"]
            self.role = "target"

        def load_model(self):
            pass

        def serve_forever(self, ready_callback):
            ready_callback()

    monkeypatch.setattr(daemon, "load_general_plugins", load_plugins)
    monkeypatch.setattr(daemon, "WeightCacheDaemon", SyntheticDaemon)
    ready = SimpleNamespace(put=ready_ranks.append)
    daemon._run_daemon(0, 0, SimpleNamespace(), "unused", None, ready)
    assert ready_ranks == [("target", 0)]


def test_weight_cache_caches_mtp_and_eagle_drafts():
    from types import SimpleNamespace

    from vllm.model_executor.model_loader.weight_cache.protocol import (
        caches_draft_model,
    )

    draft = object()
    for method in ("mtp", "eagle", "eagle3"):
        assert caches_draft_model(
            SimpleNamespace(method=method, draft_model_config=draft)
        )
    assert not caches_draft_model(
        SimpleNamespace(method="dflash", draft_model_config=draft)
    )
    assert not caches_draft_model(
        SimpleNamespace(method="mtp", draft_model_config=None)
    )
    assert not caches_draft_model(None)


def test_weight_cache_exports_eagle_ownership_flags():
    """The engine never runs load_weights for cached drafts, so the flags that
    decide whether to share the target's embed/lm_head travel with the state."""
    import torch

    from vllm.model_executor.model_loader.weight_cache.protocol import (
        export_model_attrs,
    )

    model = torch.nn.Module()
    assert export_model_attrs(model) == {}
    model.has_own_lm_head = True
    assert export_model_attrs(model) == {"has_own_lm_head": True}


def test_weight_cache_target_and_draft_use_distinct_sockets(tmp_path):
    from vllm.model_executor.model_loader.weight_cache.protocol import get_socket_path

    target_path = get_socket_path("GPU-abc", str(tmp_path))
    draft_path = get_socket_path("GPU-abc", str(tmp_path), is_draft_model=True)

    assert target_path != draft_path
    assert draft_path.endswith("GPU-abc_draft0.sock")


def test_draft_load_config_under_ipc_cache():
    """A cached draft is routed to the daemon's draft group; any other draft
    falls back to disk instead of hitting the target daemon; an explicit
    draft_load_config always wins."""
    from types import SimpleNamespace

    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader.utils import get_draft_load_config

    ipc = LoadConfig(
        load_format="ipc_cache", model_loader_extra_config={"fallback": False}
    )
    explicit = LoadConfig(load_format="fastsafetensors")

    def cfg(method, draft_load_config=None, load_config=ipc):
        return SimpleNamespace(
            load_config=load_config,
            speculative_config=SimpleNamespace(
                method=method,
                draft_model_config=object(),
                draft_load_config=draft_load_config,
            ),
        )

    mtp = get_draft_load_config(cfg("mtp"))
    assert mtp.load_format == "ipc_cache"
    assert mtp.weight_cache_is_draft_model
    assert mtp.weight_cache_draft_model_idx == 0
    assert mtp.model_loader_extra_config == {"fallback": False}

    eagle = get_draft_load_config(cfg("eagle3"))
    assert eagle.load_format == "ipc_cache"
    assert eagle.weight_cache_is_draft_model

    dflash = get_draft_load_config(cfg("dflash"))
    assert dflash.load_format == "auto"
    assert not dflash.weight_cache_is_draft_model
    assert dflash.model_loader_extra_config == {}

    assert get_draft_load_config(cfg("mtp", explicit)) is explicit
    disk = LoadConfig(load_format="fastsafetensors")
    assert get_draft_load_config(cfg("mtp", load_config=disk)) is disk


def test_draft_daemon_keeps_target_config_for_model_construction():
    """GLM MTP constructors read target and draft configs at different levels."""
    from types import SimpleNamespace

    from vllm.model_executor.model_loader.weight_cache.daemon import (
        get_draft_daemon_config,
    )

    target, draft = object(), object()
    config = SimpleNamespace(
        model_config=target,
        speculative_config=SimpleNamespace(
            method="mtp",
            draft_model_config=draft,
            moe_backend=None,
            attention_backend=None,
            kv_cache_dtype=None,
        ),
    )
    returned_config, returned_model = get_draft_daemon_config(config)
    assert returned_config is config
    assert returned_config.model_config is target
    assert returned_model is draft


def test_weight_cache_key_rejects_target_as_draft():
    """The same checkpoint/shard cannot answer a request for another role."""
    from dataclasses import replace

    from vllm.model_executor.model_loader.weight_cache.protocol import WeightCacheKey

    target = WeightCacheKey(
        checkpoint="synthetic",
        model_arch="same",
        tp_size=8,
        tp_rank=0,
        dtype="bfloat16",
        quantization="slimquant_marlin",
        quant_config_hash="q",
        revision=None,
        vllm_version="test",
    )
    draft = replace(target, is_draft_model=True, draft_model_idx=0)
    assert target.mismatched_fields(draft) == ["is_draft_model", "draft_model_idx"]


@pytest.mark.parametrize("explicit", [False, True])
def test_mrv2_routes_draft_loader_before_model_loading(monkeypatch, explicit):
    """An MTP consumer must not silently reuse the target IPC role/config."""
    from contextlib import nullcontext
    from types import ModuleType, SimpleNamespace

    from vllm.config import LoadConfig
    from vllm.v1.worker.gpu.spec_decode.eagle import utils

    backends = ModuleType("vllm.compilation.backends")
    monkeypatch.setattr(
        backends, "set_model_tag", lambda _tag: nullcontext(), raising=False
    )
    monkeypatch.setitem(sys.modules, backends.__name__, backends)
    disk = LoadConfig(load_format="safetensors") if explicit else None
    draft_model = object()
    config = SimpleNamespace(
        load_config=LoadConfig(
            load_format="ipc_cache", model_loader_extra_config={"fallback": False}
        ),
        speculative_config=SimpleNamespace(
            method="mtp",
            draft_model_config=draft_model,
            draft_load_config=disk,
            kv_cache_dtype=None,
        ),
    )
    monkeypatch.setattr(
        utils,
        "replace",
        lambda obj, **updates: SimpleNamespace(**(vars(obj) | updates)),
    )

    class ReachedLoader(Exception):
        pass

    def get_model(*, vllm_config, model_config):
        assert model_config is draft_model
        actual = vllm_config.load_config
        if explicit:
            assert actual is disk
        else:
            assert actual.load_format == "ipc_cache"
            assert actual.weight_cache_is_draft_model
            assert actual.weight_cache_draft_model_idx == 0
            assert actual.model_loader_extra_config == {"fallback": False}
        raise ReachedLoader

    monkeypatch.setattr(utils, "get_model", get_model)
    with pytest.raises(ReachedLoader):
        utils.load_eagle_model(object(), config)


def _ep_parallel(tp=1, pcp=8, **updates):
    from types import SimpleNamespace

    values = dict(
        tensor_parallel_size=tp,
        prefill_context_parallel_size=pcp,
        decode_context_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=True,
        enable_eplb=False,
        eplb_config=SimpleNamespace(num_redundant_experts=0),
        expert_placement_strategy="linear",
        all2all_backend="allgather_reducescatter",
        nnodes=1,
        node_rank=0,
    )
    return SimpleNamespace(**(values | updates))


@pytest.mark.parametrize("tp,pcp", [(8, 1), (4, 2), (2, 4), (1, 8)])
def test_cache_ep_topology_preserves_global_and_group_ranks(tp, pcp):
    from vllm.model_executor.model_loader.weight_cache import protocol

    resolve = getattr(protocol, "resolve_cache_topology", None)
    assert callable(resolve), "cache ownership must include PCP and EP"
    for rank in range(8):
        topology = resolve(_ep_parallel(tp, pcp), rank)
        assert topology.world_size == 8
        assert topology.tp_rank == rank % tp
        assert topology.pcp_rank == rank // tp
        assert (topology.ep_size, topology.ep_rank) == (8, rank)
        assert list(topology.local_global_ranks) == list(range(8))
    topology = resolve(_ep_parallel(tp, pcp, nnodes=2, node_rank=1), 7)
    assert list(topology.local_global_ranks) == [4, 5, 6, 7]


@pytest.mark.parametrize(
    "updates",
    [
        dict(pipeline_parallel_size=2),
        dict(data_parallel_size=2),
        dict(enable_eplb=True),
        dict(nnodes=3),
        dict(node_rank=1),
        dict(expert_placement_strategy="unknown"),
    ],
)
def test_cache_ep_rejects_unsupported_ownership(updates):
    from vllm.model_executor.model_loader.weight_cache import protocol

    resolve = getattr(protocol, "resolve_cache_topology", None)
    assert callable(resolve)
    with pytest.raises(ValueError):
        resolve(_ep_parallel(**updates), 0)


def _cache_key(**updates):
    from vllm.model_executor.model_loader.weight_cache.protocol import WeightCacheKey

    values = dict(
        checkpoint="synthetic",
        model_arch="MoE",
        tp_size=1,
        tp_rank=0,
        dtype="float32",
        quantization=None,
        quant_config_hash="",
        revision=None,
        vllm_version="test",
    )
    return WeightCacheKey(**(values | updates))


def test_ep_daemon_rejects_legacy_request_before_export(monkeypatch):
    from dataclasses import replace

    from vllm.model_executor.model_loader.weight_cache import daemon

    key = _cache_key()
    assert hasattr(key, "enable_ep"), "legacy defaults must not imply EP capability"
    owner = object.__new__(daemon.WeightCacheDaemon)
    owner.cache_config = replace(key, enable_ep=True, ep_size=8)
    owner.model = object()
    replies = []
    monkeypatch.setattr(daemon, "send_msg", lambda _, msg: replies.append(msg))
    monkeypatch.setattr(
        daemon, "export_entries", lambda _: pytest.fail("export before protocol check")
    )
    owner._handle_get_state(
        None, {"cmd": "get_state", "cache_config": owner.cache_config}
    )
    assert replies[0]["status"] == "error"


@pytest.mark.parametrize("broken", ["schema", "key", "uuid", "wrong_uuid"])
def test_ep_client_checks_response_before_rebuild(monkeypatch, broken):
    import socket

    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader.weight_cache import ipc_loader, protocol

    assert hasattr(_cache_key(), "enable_ep")
    key = _cache_key(enable_ep=True, ep_size=8, pcp_size=8)
    client = ipc_loader.IpcModelLoader(
        LoadConfig(
            load_format="ipc_cache", model_loader_extra_config={"fallback": False}
        )
    )
    left, right = socket.socketpair()
    response = dict(
        status="ok", schema_version=2, cache_config=key, gpu_uuid="gpu", entries={}
    )
    if broken == "schema":
        response["schema_version"] = 1
    elif broken == "key":
        response["cache_config"] = _cache_key()
    elif broken == "uuid":
        del response["gpu_uuid"]
    else:
        response["gpu_uuid"] = "other"
    protocol.send_msg(right, response)
    monkeypatch.setattr(client, "_connect", lambda _: left)
    monkeypatch.setattr(ipc_loader, "get_current_device_uuid", lambda: "gpu")
    try:
        with pytest.raises(
            (protocol.CacheConfigMismatchError, protocol.WeightCacheUnavailableError)
        ):
            client._request_state(key)
        assert protocol.recv_msg(right)["cmd"] == "get_state_v2"
    finally:
        right.close()


@pytest.mark.parametrize(
    "placement,backend,mask",
    [
        ("linear", "allgather_reducescatter", False),
        ("linear", "allgather_reducescatter", True),
        ("round_robin", "deepep_low_latency", False),
    ],
)
@pytest.mark.parametrize("broken", [None, "missing", "owner"])
def test_ep_ipc_restores_manager_references_and_packed_aliases(
    placement, backend, mask, broken
):
    from dataclasses import replace

    import torch

    from vllm.config import LoadConfig
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEParallelConfig,
    )
    from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.model_loader.weight_cache.daemon import export_entries
    from vllm.model_executor.model_loader.weight_cache.ipc_loader import IpcModelLoader

    parallel = replace(
        FusedMoEParallelConfig.make_no_parallel(),
        ep_size=8,
        ep_rank=3,
        pcp_size=8,
        pcp_rank=3,
        use_ep=True,
        all2all_backend=backend,
    )

    def construct(device):
        with torch.device(device):
            manager = ExpertMapManager(
                16, 2, 16, 0, 2, parallel, placement, False, rocm_aiter_enabled=mask
            )
            layer = RoutedExperts.__new__(RoutedExperts)
            torch.nn.Module.__init__(layer)
            layer.expert_map_manager = manager
            layer.update_expert_map_info()
            layer.register_parameter("weight", torch.nn.Parameter(torch.empty(2, 4)))
            layer.register_parameter("tied", layer.weight)
        return layer

    owner = construct("cpu")
    owner.weight = torch.nn.Parameter(torch.arange(12.0).reshape(3, 4).t())
    owner.tied = owner.weight
    entries, aliases = export_entries(owner)
    loader = IpcModelLoader(
        LoadConfig(
            load_format="ipc_cache", model_loader_extra_config={"fallback": False}
        )
    )
    for _ in range(2):
        client = construct("meta")
        if broken is not None:
            bad_entries = dict(entries)
            if broken == "missing":
                del bad_entries["_expert_map"]
            else:
                from vllm.model_executor.model_loader.weight_cache.protocol import (
                    TensorEntry,
                )

                bad_entries["_expert_map"] = TensorEntry.from_tensor(
                    torch.zeros_like(owner._expert_map), "buffer"
                )
            with pytest.raises(ValueError, match="restored|ownership"):
                loader._apply_entries(client, bad_entries, aliases, 0)
            continue
        loader._apply_entries(client, entries, aliases, 0)
        assert client.expert_map_manager.expert_map is client._expert_map
        assert client.expert_map_manager.expert_mask is client.expert_mask
        assert client.expert_map_manager.get_local_expert_ids() == (
            [6, 7] if placement == "linear" else [3, 11]
        )
        if placement == "round_robin":
            assert all(
                a is b
                for a, b in zip(
                    client.expert_map_manager.routing_tables,
                    client._expert_routing_tables(),
                )
            )
        assert client.weight is client.tied
        assert client.weight.data_ptr() == owner.weight.data_ptr()
        assert client.weight.shape == (4, 3) and client.weight.stride() == (1, 4)


def test_cache_rejects_redundant_experts():
    from types import SimpleNamespace

    from vllm.model_executor.model_loader.weight_cache.protocol import (
        resolve_cache_topology,
    )

    with pytest.raises(ValueError, match="redundant"):
        resolve_cache_topology(
            _ep_parallel(eplb_config=SimpleNamespace(num_redundant_experts=1)), 0
        )


@pytest.mark.parametrize("mode", ["zero_copy", "copy"])
def test_ipc_restores_six_dimensional_weights_and_layout_buffer(monkeypatch, mode):
    """Packed layouts and small CPU markers precede runtime reconstruction.

    This checks generic loader transport, not a particular device packer or
    DeepGEMM kernel. The backend adapter must validate the marker's meaning.
    """
    from types import SimpleNamespace

    import torch

    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader.weight_cache import ipc_loader
    from vllm.model_executor.model_loader.weight_cache.daemon import export_entries
    from vllm.model_executor.utils import is_weights_pre_processed

    packed_shape = (2, 1, 2, 4, 16, 16)
    events = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.empty((2, 32, 64), dtype=torch.int8), requires_grad=False
            )
            self.tied = self.weight

        def restore_weight_cache_state(self):
            assert self.weight.shape == packed_shape
            assert self.layout.device.type == "cpu"
            assert self.layout.tolist() == [1, 2, 32, 64]
            events.append("restore")

    owner = Model()
    owner.weight = torch.nn.Parameter(
        torch.arange(4096, dtype=torch.int64).to(torch.int8).reshape(packed_shape),
        requires_grad=False,
    )
    owner.tied = owner.weight
    owner.register_buffer("layout", torch.tensor([1, 2, 32, 64]), persistent=False)
    entries, aliases = export_entries(owner)
    assert "layout" in entries and "layout" not in owner.state_dict()
    original = owner.weight.clone()

    loader = ipc_loader.IpcModelLoader(
        LoadConfig(
            load_format="ipc_cache",
            model_loader_extra_config={"mode": mode, "fallback": False},
        )
    )
    monkeypatch.setattr(ipc_loader, "initialize_model", lambda **_: Model())
    monkeypatch.setattr(ipc_loader, "check_ipc_quant_support", lambda _: None)
    monkeypatch.setattr(loader, "_send_release", lambda: events.append("release"))

    def reconstruct(model, model_config, device):
        assert is_weights_pre_processed()
        assert events[-1] == "restore"
        assert model.weight.shape == packed_shape
        assert model.weight is model.tied
        model.runtime_weight = model.weight
        events.append("runtime")

    monkeypatch.setattr(ipc_loader, "process_weights_after_loading", reconstruct)
    config = SimpleNamespace(
        device_config=SimpleNamespace(device=torch.device("cpu", 0)),
        parallel_config=_ep_parallel(),
    )
    clients = [
        loader._build_model(
            config, SimpleNamespace(dtype=torch.bfloat16), "", entries, aliases, {}
        )
        for _ in range(2)
    ]
    assert (
        events == ["restore", "runtime", *(["release"] if mode == "copy" else [])] * 2
    )
    for client in clients:
        assert client.runtime_weight is client.weight
        assert torch.equal(client.weight, original)
        assert (client.weight.data_ptr() == owner.weight.data_ptr()) == (
            mode == "zero_copy"
        )
        assert client.layout.tolist() == [1, 2, 32, 64]
    assert torch.equal(owner.weight, original)


@pytest.mark.parametrize(
    "field,value",
    [
        ("enable_ep", False),
        ("ep_size", 4),
        ("ep_rank", 1),
        ("pcp_size", 4),
        ("pcp_rank", 1),
        ("expert_placement", "round_robin"),
        ("all2all_backend", "deepep_low_latency"),
        ("moe_backend", "triton"),
        ("moe_backend", "deep_gemm"),
        ("tp_rank", 1),
        ("is_draft_model", True),
    ],
)
def test_ep_socket_rejects_wrong_identity_before_export(monkeypatch, field, value):
    import socket
    from dataclasses import replace

    from vllm.model_executor.model_loader.weight_cache import daemon, protocol

    owner = object.__new__(daemon.WeightCacheDaemon)
    owner.cache_config = _cache_key(enable_ep=True, ep_size=8, pcp_size=8)
    owner.model = object()
    monkeypatch.setattr(
        daemon, "export_entries", lambda _: pytest.fail("must reject before export")
    )
    left, right = socket.socketpair()
    try:
        protocol.send_msg(
            left,
            {
                "cmd": "get_state_v2",
                "cache_config": replace(owner.cache_config, **{field: value}),
            },
        )
        owner._handle_connection(right)
        response = protocol.recv_msg(left)
        assert response == {"status": "mismatch", "fields": [field]}
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("ep", [False, True])
def test_socket_two_clients_and_legacy_tp_response(monkeypatch, ep):
    import socket

    import torch

    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader.weight_cache import (
        daemon,
        ipc_loader,
    )

    owner = object.__new__(daemon.WeightCacheDaemon)
    owner.tp_rank, owner.role = 0, "target"
    owner.cache_config = _cache_key(
        enable_ep=ep, ep_size=8 if ep else 1, pcp_size=8 if ep else 1
    )
    owner.model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(daemon, "get_current_device_uuid", lambda: "gpu")
    monkeypatch.setattr(ipc_loader, "get_current_device_uuid", lambda: "gpu")
    for _ in range(2):
        client = ipc_loader.IpcModelLoader(
            LoadConfig(
                load_format="ipc_cache", model_loader_extra_config={"fallback": False}
            )
        )
        left, right = socket.socketpair()
        monkeypatch.setattr(client, "_connect", lambda _, sock=left: sock)
        worker = threading.Thread(target=owner._handle_connection, args=(right,))
        worker.start()
        try:
            entries, aliases, attrs = client._request_state(owner.cache_config)
            model = torch.nn.Linear(2, 2, device="meta")
            client._apply_entries(model, entries, aliases, 0)
            torch.testing.assert_close(
                model(torch.ones(1, 2)), owner.model(torch.ones(1, 2))
            )
        finally:
            worker.join(timeout=5)
            right.close()
        assert not worker.is_alive()


def test_cache_tp_legacy_defaults_and_multinode():
    from vllm.model_executor.model_loader.weight_cache.protocol import (
        resolve_cache_topology,
    )

    key = _cache_key()
    assert not key.requires_v2 and not key.enable_ep
    topology = resolve_cache_topology(
        _ep_parallel(16, 1, enable_expert_parallel=False, nnodes=2, node_rank=1), 15
    )
    assert topology.tp_rank == 15 and topology.world_size == 16
    assert (topology.ep_size, topology.ep_rank, topology.pcp_rank) == (1, 0, 0)
    assert list(topology.local_global_ranks) == list(range(8, 16))


def test_daemon_initializes_ep_groups_in_configuration_context(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    import torch

    from vllm.model_executor.model_loader.weight_cache import daemon

    config = SimpleNamespace(
        parallel_config=_ep_parallel(2, 4),
        kernel_config=SimpleNamespace(moe_backend="triton"),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["MoE"], quantization_config={"a": 1}
            ),
            model="synthetic",
            dtype=torch.float32,
            quantization=None,
            revision=None,
        ),
    )
    owner = daemon.WeightCacheDaemon(config, 7, 3, "test")
    assert (owner.global_rank, owner.tp_rank, owner.local_rank) == (7, 1, 3)
    assert (owner.cache_config.ep_rank, owner.cache_config.pcp_rank) == (7, 3)
    initial_hash = owner.cache_config.quant_config_hash
    config.model_config.hf_config.quantization_config["a"] = 2
    in_context = False

    @contextmanager
    def context(actual):
        nonlocal in_context
        assert actual is config
        in_context = True
        yield
        in_context = False

    def distributed(**kwargs):
        assert in_context
        assert kwargs["world_size"] == 8 and kwargs["rank"] == 7
        assert kwargs["local_rank"] == 3

    def groups(tp, pp, pcp, dcp):
        assert in_context
        assert (tp, pp, pcp, dcp) == (2, 1, 4, 1)

    monkeypatch.setattr(daemon, "set_current_vllm_config", context)
    monkeypatch.setattr(daemon, "init_distributed_environment", distributed)
    monkeypatch.setattr(daemon, "ensure_model_parallel_initialized", groups)
    monkeypatch.setattr(daemon, "init_workspace_manager", lambda _: None)
    monkeypatch.setattr(torch.accelerator, "set_device_index", lambda _: None)
    monkeypatch.setattr(daemon, "get_daemon_model", lambda *_: torch.nn.Linear(1, 1))
    owner.load_model()
    assert owner.cache_config.quant_config_hash == initial_hash


def test_ep_missing_runtime_tensor_cannot_be_materialized():
    import torch

    from vllm.model_executor.model_loader.weight_cache import ipc_loader

    model = torch.nn.Linear(2, 2, device="meta")
    with pytest.raises(ValueError, match="missing"):
        ipc_loader._materialize_remaining_meta_tensors(
            model, torch.device("cpu"), strict=True
        )
    assert model.weight.is_meta


def test_multinode_daemon_keeps_its_rendezvous_in_config_context(monkeypatch):
    from contextlib import contextmanager
    from dataclasses import dataclass
    from types import SimpleNamespace

    import torch

    from vllm.model_executor.model_loader.weight_cache import daemon

    @dataclass
    class Parallel:
        nnodes: int = 2
        master_addr: str = "engine"
        master_port: int = 1111

    @dataclass
    class Config:
        parallel_config: Parallel

    config = Config(Parallel())
    owner = object.__new__(daemon.WeightCacheDaemon)
    owner.vllm_config = config
    owner.model_config = SimpleNamespace()
    owner.topology = daemon.resolve_cache_topology(
        _ep_parallel(16, 1, enable_expert_parallel=False, nnodes=2, node_rank=1), 15
    )
    owner.tp_rank, owner.global_rank, owner.local_rank, owner.role = 15, 15, 7, "target"
    owner.distributed_init_method = "tcp://daemon:2222"
    context_config = None

    @contextmanager
    def context(actual):
        nonlocal context_config
        context_config = actual
        yield

    def distributed(**kwargs):
        assert kwargs["world_size"] == 16 and kwargs["rank"] == 15
        assert context_config.parallel_config.master_addr == "daemon"
        assert context_config.parallel_config.master_port == 2222

    monkeypatch.setattr(daemon, "set_current_vllm_config", context)
    monkeypatch.setattr(daemon, "init_distributed_environment", distributed)
    monkeypatch.setattr(daemon, "ensure_model_parallel_initialized", lambda *_: None)
    monkeypatch.setattr(daemon, "init_workspace_manager", lambda _: None)
    monkeypatch.setattr(torch.accelerator, "set_device_index", lambda _: None)
    monkeypatch.setattr(daemon, "get_daemon_model", lambda *_: torch.nn.Linear(1, 1))
    owner.load_model()
    assert config.parallel_config.master_port == 1111


@pytest.mark.parametrize("ep", [False, True])
def test_client_legacy_wire_compatibility(monkeypatch, ep):
    import socket

    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader.weight_cache import ipc_loader, protocol

    client = ipc_loader.IpcModelLoader(LoadConfig(load_format="ipc_cache"))
    left, right = socket.socketpair()
    monkeypatch.setattr(client, "_connect", lambda _: left)
    key = _cache_key(enable_ep=ep, ep_size=8 if ep else 1)

    # A legacy daemon accepts get_state without UUID/capabilities and rejects
    # the new opcode before it can export anything.
    def legacy_server():
        request = protocol.recv_msg(right)
        if request["cmd"] == "get_state":
            protocol.send_msg(right, {"status": "ok", "entries": {}})
        else:
            protocol.send_msg(right, {"status": "error", "message": "Unknown command"})

    worker = threading.Thread(target=legacy_server)
    worker.start()
    try:
        if ep:
            with pytest.raises(
                protocol.WeightCacheUnavailableError, match="Unknown command"
            ):
                client._request_state(key)
        else:
            assert client._request_state(key) == ({}, {}, {})
    finally:
        worker.join(timeout=5)
        right.close()
    assert not worker.is_alive()


@pytest.mark.parametrize("pcp", [1, 2, 4, 8])
def test_static_ep_selects_existing_prepare_finalize(monkeypatch, pcp):
    from dataclasses import replace
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe import all2all_utils
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
        MoEPrepareAndFinalizeNaiveDPEPModular,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
        MoEPrepareAndFinalizeNoDPEPModular,
    )

    parallel = replace(
        FusedMoEParallelConfig.make_no_parallel(),
        ep_size=8,
        ep_rank=7,
        pcp_size=pcp,
        pcp_rank=pcp - 1,
        use_ep=True,
        all2all_backend="allgather_reducescatter",
    )
    moe = object.__new__(FusedMoEConfig)
    moe.moe_parallel_config = parallel
    moe.moe_backend = "triton"
    monkeypatch.setattr(
        all2all_utils, "get_ep_all2all_manager", lambda _: SimpleNamespace(world_size=8)
    )
    result = all2all_utils.maybe_make_prepare_finalize(
        moe, None, allow_new_interface=True
    )
    assert isinstance(
        result,
        MoEPrepareAndFinalizeNoDPEPModular
        if pcp == 1
        else MoEPrepareAndFinalizeNaiveDPEPModular,
    )
    assert result.num_dispatchers() == (1 if pcp == 1 else 8)


def test_reload_retains_topology_after_config_context_exit(monkeypatch):
    from types import SimpleNamespace

    import torch

    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader.weight_cache import ipc_loader

    loader = ipc_loader.IpcModelLoader(LoadConfig(load_format="ipc_cache"))
    model = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["MoE"]),
        model="synthetic",
        dtype=torch.float32,
        quantization=None,
        revision=None,
    )
    config = SimpleNamespace(
        parallel_config=_ep_parallel(),
        kernel_config=SimpleNamespace(moe_backend="triton"),
    )
    rank = SimpleNamespace(rank_in_group=7)
    keys = []
    monkeypatch.setattr(ipc_loader, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(ipc_loader, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(ipc_loader, "get_world_group", lambda: rank)
    monkeypatch.setattr(
        ipc_loader,
        "get_current_vllm_config",
        lambda: pytest.fail("context already exited"),
        raising=False,
    )
    monkeypatch.setattr(
        loader, "_request_state", lambda key: keys.append(key) or ({}, {}, {})
    )
    loader._fetch_entries(model, config)
    config.parallel_config.enable_expert_parallel = False
    loader._fetch_entries(model)
    assert keys[0] == keys[1]
    assert keys[1].ep_rank == 7 and keys[1].enable_ep
    rank.rank_in_group = 8
    with pytest.raises(ValueError, match="rank"):
        loader._fetch_entries(model)
    assert len(keys) == 2


def test_legacy_reload_without_ambient_config_keeps_tp_wire(monkeypatch):
    from types import SimpleNamespace

    import torch

    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader.weight_cache import ipc_loader

    loader = ipc_loader.IpcModelLoader(LoadConfig(load_format="ipc_cache"))
    model = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["Linear"]),
        model="synthetic",
        dtype=torch.float32,
        quantization=None,
        revision=None,
    )
    monkeypatch.setattr(
        ipc_loader, "get_current_vllm_config_or_none", lambda: None, raising=False
    )
    monkeypatch.setattr(
        ipc_loader,
        "get_current_vllm_config",
        lambda: pytest.fail("no ambient config"),
        raising=False,
    )
    monkeypatch.setattr(ipc_loader, "get_tensor_model_parallel_world_size", lambda: 16)
    monkeypatch.setattr(ipc_loader, "get_tensor_model_parallel_rank", lambda: 15)
    monkeypatch.setattr(
        ipc_loader, "get_pcp_group", lambda: SimpleNamespace(world_size=1)
    )
    keys = []
    monkeypatch.setattr(
        loader, "_request_state", lambda key: keys.append(key) or ({}, {}, {})
    )
    loader._fetch_entries(model)
    assert keys[0].tp_rank == 15 and not keys[0].requires_v2


def test_ep_reload_without_proven_topology_rejects_before_request(monkeypatch):
    from types import SimpleNamespace

    import torch

    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader.weight_cache import ipc_loader

    loader = ipc_loader.IpcModelLoader(LoadConfig(load_format="ipc_cache"))
    model = torch.nn.Module()
    model.expert_map_manager = SimpleNamespace(use_ep=True)
    monkeypatch.setattr(
        ipc_loader, "get_current_vllm_config_or_none", lambda: None, raising=False
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(
        loader, "_fetch_entries", lambda _: pytest.fail("unproven EP request")
    )
    with pytest.raises(ValueError, match="topology"):
        loader.load_weights(model, SimpleNamespace())


def test_tp_only_moe_reload_does_not_infer_ep_from_group(monkeypatch):
    from types import SimpleNamespace

    import torch

    import vllm.distributed
    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader.weight_cache import ipc_loader

    loader = ipc_loader.IpcModelLoader(LoadConfig(load_format="ipc_cache"))
    model = torch.nn.Module()
    model.expert_map_manager = SimpleNamespace(
        use_ep=False, moe_parallel_config=SimpleNamespace(pcp_size=1)
    )
    monkeypatch.setattr(ipc_loader, "get_current_vllm_config_or_none", lambda: None)
    monkeypatch.setattr(
        vllm.distributed, "get_ep_group", lambda: SimpleNamespace(world_size=16)
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(loader, "_fetch_entries", lambda _: ({}, {}, {}))
    loader.load_weights(model, SimpleNamespace())


@pytest.mark.parametrize(
    "tp,pcp,nnodes,node",
    [(8, 1, 1, 0), (4, 2, 1, 0), (2, 4, 1, 0), (1, 8, 1, 0), (16, 1, 2, 1)],
)
def test_daemon_launcher_spawns_global_ranks_for_each_role(
    monkeypatch, tp, pcp, nnodes, node
):
    from types import SimpleNamespace

    from vllm.model_executor.model_loader.weight_cache import daemon

    parallel = _ep_parallel(
        tp, pcp, nnodes=nnodes, node_rank=node, master_addr="localhost"
    )
    config = SimpleNamespace(
        parallel_config=parallel, load_config=SimpleNamespace(load_format="auto")
    )
    args = SimpleNamespace(
        weight_cache_master_port=2222,
        weight_cache_draft_master_port=2223,
        weight_cache_socket_dir=None,
    )
    parser = SimpleNamespace(add_argument=lambda *a, **k: None, parse_args=lambda: args)
    engine = SimpleNamespace(create_engine_config=lambda: config)
    monkeypatch.setattr(daemon, "FlexibleArgumentParser", lambda **_: parser)
    monkeypatch.setattr(
        daemon,
        "EngineArgs",
        SimpleNamespace(add_cli_args=lambda _: None, from_cli_args=lambda _: engine),
    )
    monkeypatch.setattr(daemon, "check_ipc_platform_support", lambda: None)
    monkeypatch.setattr(daemon, "get_draft_daemon_config", lambda _: (config, object()))
    monkeypatch.setattr(daemon.signal, "signal", lambda *_: None)
    processes, ready = [], []

    class Process:
        exitcode = 0

        def __init__(self, target, args, name):
            self.args = args
            processes.append(self)

        def start(self):
            ready.append(("draft" if self.args[6] else "target", self.args[0]))

        def join(self):
            pass

    ctx = SimpleNamespace(
        Queue=lambda: SimpleNamespace(get=lambda **_: ready.pop(0)), Process=Process
    )
    monkeypatch.setattr(daemon.multiprocessing, "get_context", lambda _: ctx)
    with pytest.raises(SystemExit) as exc:
        daemon.main()
    assert exc.value.code == 0
    expected = list(range(8, 16)) if nnodes == 2 else list(range(8))
    assert [p.args[0] for p in processes] == expected * 2
    assert [p.args[1] for p in processes] == list(range(8)) * 2
    assert [p.args[6] for p in processes] == [False] * 8 + [True] * 8


def test_config_free_pcp_fetch_rejects_unproven_ownership(monkeypatch):
    from types import SimpleNamespace

    import torch

    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader.weight_cache import ipc_loader

    loader = ipc_loader.IpcModelLoader(LoadConfig(load_format="ipc_cache"))
    model = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["Linear"]),
        model="synthetic",
        dtype=torch.float32,
        quantization=None,
        revision=None,
    )
    monkeypatch.setattr(ipc_loader, "get_current_vllm_config_or_none", lambda: None)
    monkeypatch.setattr(
        ipc_loader,
        "get_pcp_group",
        lambda: SimpleNamespace(world_size=8),
        raising=False,
    )
    monkeypatch.setattr(ipc_loader, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(ipc_loader, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        loader, "_request_state", lambda _: pytest.fail("unproven PCP request")
    )
    with pytest.raises(ValueError, match="topology"):
        loader._fetch_entries(model)


@pytest.mark.parametrize(
    "tp,pcp,dcp", [(1, 8, 8), (2, 4, 4), (4, 2, 8), (4, 2, 2), (8, 1, 4)]
)
def test_cache_accepts_parallel_config_dcp_axes(tp, pcp, dcp):
    from vllm.config import ParallelConfig
    from vllm.model_executor.model_loader.weight_cache.protocol import (
        resolve_cache_topology,
    )

    config = ParallelConfig(
        tensor_parallel_size=tp,
        prefill_context_parallel_size=pcp,
        decode_context_parallel_size=dcp,
        enable_expert_parallel=True,
        distributed_executor_backend="mp",
    )
    topology = resolve_cache_topology(config, 7)
    assert topology.dcp_size == dcp
    assert topology.world_size == 8 and topology.ep_rank == 7


@pytest.mark.parametrize("tp,pcp,dcp", [(4, 2, 4), (2, 4, 2), (8, 1, 3), (1, 8, 0)])
def test_cache_rejects_invalid_dcp_axes(tp, pcp, dcp):
    from vllm.model_executor.model_loader.weight_cache.protocol import (
        resolve_cache_topology,
    )

    with pytest.raises(ValueError, match="TP/PCP/DCP"):
        resolve_cache_topology(
            _ep_parallel(tp, pcp, decode_context_parallel_size=dcp), 0
        )


@pytest.fixture
def ep_ipc_probe():
    """Load the standalone receipt validator without importing a GPU runtime."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "tools/weight_cache/validate_ep_ipc.py"
    spec = importlib.util.spec_from_file_location("validate_ep_ipc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ipc_reduction_args():
    import inspect

    import torch
    from torch.multiprocessing.reductions import rebuild_cuda_tensor

    signature = inspect.signature(rebuild_cuda_tensor)
    fields = dict(
        tensor_cls=torch.Tensor,
        tensor_size=(2, 3),
        tensor_stride=(3, 1),
        tensor_offset=0,
        storage_cls=torch.storage.TypedStorage,
        dtype=torch.int8,
        storage_device=0,
        storage_handle=b"test-allocation",
        storage_size_bytes=6,
        storage_offset_bytes=0,
        requires_grad=False,
        ref_counter_handle=b"consumer-a",
        ref_counter_offset=0,
        event_handle=b"event-a",
        event_sync_required=False,
    )
    return signature, fields


def test_ipc_probe_fingerprint_tracks_allocation_not_consumer(
    ep_ipc_probe, ipc_reduction_args
):
    signature, fields = ipc_reduction_args
    original = ep_ipc_probe.fingerprint(signature, signature.bind(**fields).args)
    for field, replacement, same in (
        ("ref_counter_handle", b"consumer-b", True),
        ("event_handle", b"event-b", True),
        ("storage_handle", b"another-allocation", False),
        ("storage_offset_bytes", 1, False),
        ("tensor_offset", 1, False),
    ):
        changed = fields | {field: replacement}
        actual = ep_ipc_probe.fingerprint(signature, signature.bind(**changed).args)
        assert (original == actual) is same


def test_ipc_probe_empty_allocation_has_layout_only_identity(
    ep_ipc_probe, ipc_reduction_args
):
    signature, fields = ipc_reduction_args
    empty = fields | dict(storage_handle=None, storage_size_bytes=0, tensor_size=(0, 3))
    digest = ep_ipc_probe.fingerprint(signature, signature.bind(**empty).args)
    changed = empty | dict(tensor_size=(0, 4))
    assert digest != ep_ipc_probe.fingerprint(signature, signature.bind(**changed).args)
    view = fields | dict(tensor_size=(0, 3))
    assert digest != ep_ipc_probe.fingerprint(signature, signature.bind(**view).args)


@pytest.mark.parametrize(
    "overrides",
    [
        {"storage_handle": None},
        {"storage_handle": None, "tensor_size": (0, 3)},
        {"storage_handle": None, "storage_size_bytes": 0},
        {"storage_handle": "invalid"},
        {"storage_handle": b""},
    ],
)
def test_ipc_probe_rejects_missing_handle_for_backed_storage(
    ep_ipc_probe, ipc_reduction_args, overrides
):
    signature, fields = ipc_reduction_args
    with pytest.raises(RuntimeError, match="Invalid IPC handle.*safe-name"):
        ep_ipc_probe.fingerprint(
            signature, signature.bind(**(fields | overrides)).args, "safe-name"
        )


def test_ipc_probe_requires_actual_storage_alias(ep_ipc_probe):
    import torch

    original = torch.arange(8)
    ep_ipc_probe.same_tensor(original, original.view_as(original))
    with pytest.raises(RuntimeError, match="storage/layout changed"):
        ep_ipc_probe.same_tensor(original, original.clone())
    with pytest.raises(RuntimeError, match="storage/layout changed"):
        ep_ipc_probe.same_tensor(original, original[1:])
