"""Actual daemon-backed model-load probe; no forward or model correctness claim."""

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import traceback
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def fingerprint(signature, ipc_args, name="<unnamed>"):
    """Identify exported allocation/layout, excluding per-consumer references."""
    fields = signature.bind(*ipc_args).arguments
    names = (
        "storage_handle",
        "storage_size_bytes",
        "storage_offset_bytes",
        "tensor_offset",
        "tensor_size",
        "tensor_stride",
        "dtype",
    )
    require(all(name in fields for name in names), "Unsupported Torch IPC signature")
    handle = fields["storage_handle"]
    empty_allocation = (
        handle is None
        and fields["storage_size_bytes"] == 0
        and math.prod(fields["tensor_size"]) == 0
    )
    require(
        empty_allocation or (isinstance(handle, bytes) and bool(handle)),
        f"Invalid IPC handle: name={name!r}, shape={fields['tensor_size']}, "
        f"storage_bytes={fields['storage_size_bytes']}, "
        f"handle_type={type(handle).__name__}",
    )
    record = {name: fields[name] for name in names if name != "storage_handle"}
    record["dtype"] = str(record["dtype"])
    record["storage_handle_sha256"] = (
        "empty-no-allocation"
        if empty_allocation
        else hashlib.sha256(handle).hexdigest()
    )
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def same_tensor(actual, imported):
    require(not actual.is_meta and not imported.is_meta, "Meta tensor remains")
    require(
        actual.device == imported.device
        and actual.data_ptr() == imported.data_ptr()
        and actual.untyped_storage().data_ptr() == imported.untyped_storage().data_ptr()
        and actual.dtype == imported.dtype
        and actual.shape == imported.shape
        and actual.stride() == imported.stride()
        and actual.storage_offset() == imported.storage_offset(),
        "Imported tensor storage/layout changed",
    )


def engine_options(plan, master_port):
    argv = plan["engine_python_argv"]
    require(
        argv[1:3] == ["-m", "vllm.entrypoints.openai.api_server"], "Unexpected argv"
    )
    result = []
    index = 3
    while index < len(argv):
        flag = argv[index]
        if flag in ("--host", "--port", "--master-port"):
            index += 2
        elif flag == "--enable-prompt-tokens-details":
            index += 1
        else:
            result.append(flag)
            index += 1
    return result + ["--master-port", str(master_port)]


def probe(args, report):
    import torch
    from torch.multiprocessing.reductions import rebuild_cuda_tensor

    from vllm.plugins import load_general_plugins

    load_general_plugins()
    from vllm.config import set_current_vllm_config
    from vllm.distributed import ensure_model_parallel_initialized
    from vllm.distributed.parallel_state import init_distributed_environment
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.model_executor.model_loader.weight_cache import ipc_loader, protocol
    from vllm.platforms import current_platform
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.worker.workspace import init_workspace_manager

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    require(int(os.environ["WORLD_SIZE"]) == 8, "Requires eight ranks")
    require(current_platform.is_rocm() and torch.version.hip, "Requires actual ROCm")
    require(torch.cuda.device_count() == 8, "Requires eight visible GPUs")
    torch.accelerator.set_device_index(local_rank)
    plan = json.loads(args.plan.read_text())
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    config = EngineArgs.from_cli_args(
        parser.parse_args(engine_options(plan, args.master_port))
    ).create_engine_config()
    topology = protocol.resolve_cache_topology(config.parallel_config, rank)
    require(
        topology.world_size == 8
        and topology.tp_size == 1
        and topology.pcp_size == 8
        and topology.dcp_size == 1
        and config.parallel_config.enable_expert_parallel,
        "Requires TP1 PCP8 DCP1 EP8",
    )
    loader = get_model_loader(config.load_config)
    require(type(loader) is ipc_loader.IpcModelLoader, "Requires actual IPC loader")
    require(
        loader.mode == "zero_copy" and loader.fallback is False, "Unsafe loader mode"
    )
    signature = inspect.signature(rebuild_cuda_tensor)
    require(
        {
            "storage_handle",
            "storage_size_bytes",
            "storage_offset_bytes",
            "tensor_offset",
            "tensor_size",
            "tensor_stride",
            "dtype",
        }
        <= signature.parameters.keys(),
        "Unsupported native Torch IPC signature",
    )
    report.update(
        rank=rank,
        uuid=protocol.get_current_device_uuid(),
        topology=asdict(topology),
        torch_version=torch.__version__,
        ipc_signature=str(signature),
        source_commits=plan["source_commits"],
    )
    imported, captured, requests = {}, {}, []
    original_request = loader._request_state
    original_apply = loader._apply_entries
    original_rebuild = protocol.TensorEntry.rebuild

    def request(key):
        require(key.requires_v2, "Expected versioned EP request")
        result = original_request(key)  # Production validates schema/key/physical UUID.
        requests.append(asdict(key))
        return result

    def apply(model, entries, aliases, device_index):
        require(not captured, "Unexpected second import")
        captured.update(entries=entries, aliases=aliases)
        return original_apply(model, entries, aliases, device_index)

    def rebuild(entry, device_index):
        tensor = original_rebuild(entry, device_index)
        imported[id(entry)] = tensor
        return tensor

    loader._request_state = request
    loader._apply_entries = apply
    protocol.TensorEntry.rebuild = rebuild
    model = None
    try:
        init_workspace_manager(torch.device(current_platform.device_type, local_rank))
        with set_current_vllm_config(config):
            init_distributed_environment(
                world_size=8,
                rank=rank,
                local_rank=local_rank,
                distributed_init_method=f"tcp://127.0.0.1:{args.master_port}",
                backend=current_platform.dist_backend,
                timeout=timedelta(seconds=180),
            )
            ensure_model_parallel_initialized(1, 1, 8, 1)
            model = loader.load_model(
                vllm_config=config, model_config=config.model_config
            )
        require(len(requests) == 1 and captured, "Missing unique actual import")
        tensors = dict(model.named_parameters(remove_duplicate=False))
        tensors.update(model.named_buffers(remove_duplicate=False))
        require(
            set(tensors) == set(captured["entries"]) | set(captured["aliases"]),
            "Model contains tensors not accounted for by actual IPC import",
        )
        require(
            all(not value.is_meta for value in tensors.values()), "Meta tensor remains"
        )
        hashes = {}
        empty_tensor_names, no_allocation_tensor_names = [], []
        nonempty_ipc_count = 0
        for name, entry in captured["entries"].items():
            require(
                name in tensors and id(entry) in imported, f"Missing import: {name}"
            )
            same_tensor(tensors[name], imported[id(entry)])
            if tensors[name].is_cuda:
                require(entry.ipc_args is not None, f"GPU tensor without IPC: {name}")
                hashes[name] = fingerprint(signature, entry.ipc_args, name)
                fields = signature.bind(*entry.ipc_args).arguments
                if math.prod(fields["tensor_size"]) == 0:
                    empty_tensor_names.append(name)
                if fields["storage_handle"] is None:
                    no_allocation_tensor_names.append(name)
                elif fields["storage_size_bytes"] > 0 and tensors[name].numel() > 0:
                    nonempty_ipc_count += 1
        require(nonempty_ipc_count > 0, "No nonempty native IPC allocations")
        for alias, canonical in captured["aliases"].items():
            require(tensors[alias] is tensors[canonical], f"Lost alias: {alias}")
        experts = []
        for name, layer in model.named_modules():
            if not isinstance(layer, RoutedExperts):
                continue
            manager = layer.expert_map_manager
            require(manager.expert_map is layer._expert_map, "Stale expert map")
            require(manager.expert_mask is layer.expert_mask, "Stale expert mask")
            tables = layer._expert_routing_tables()
            if tables is None:
                require(manager.routing_tables is None, "Unexpected routing tables")
            else:
                require(manager.routing_tables is not None, "Missing routing tables")
                require(
                    all(a is b for a, b in zip(tables, manager.routing_tables)),
                    "Stale routing tables",
                )
            # Production validation recomputes expected ownership and compares values.
            manager.bind_cached_tensors(layer._expert_map, layer.expert_mask, tables)
            kernel = layer.quant_method.moe_kernel
            require(type(kernel) is FusedMoEKernel, "Unexpected MoE kernel")
            require(type(kernel.fused_experts) is TritonExperts, "Not TritonExperts")
            parallel = layer.quant_method.moe.moe_parallel_config
            require(
                parallel.ep_size == 8 and parallel.ep_rank == rank, "Incorrect EP owner"
            )
            experts.append(name)
        require(experts, "No routed MoE layers observed")
        torch.cuda.synchronize()
        report.update(
            status="pass",
            ipc_fingerprints=hashes,
            aliases=captured["aliases"],
            tensor_count=len(tensors),
            nonempty_ipc_count=nonempty_ipc_count,
            empty_tensor_names=empty_tensor_names,
            empty_tensor_count=len(empty_tensor_names),
            no_allocation_tensor_names=no_allocation_tensor_names,
            routed_experts=experts,
            cache_key=requests[0],
            allocated_bytes=torch.cuda.memory_allocated(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
    finally:
        protocol.TensorEntry.rebuild = original_rebuild
        loader._request_state = original_request
        loader._apply_entries = original_apply
        model = None
        imported.clear()
        captured.clear()
        gc.collect()


def compare_runs(paths):
    runs = []
    for path in paths:
        receipts = [
            json.loads((path / f"rank-{rank}.json").read_text()) for rank in range(8)
        ]
        require(all(row["status"] == "pass" for row in receipts), "Failed rank")
        require([row["rank"] for row in receipts] == list(range(8)), "Wrong ranks")
        require(len({row["uuid"] for row in receipts}) == 8, "Duplicate physical GPU")
        runs.append(receipts)
    for first, second in zip(*runs):
        for field in (
            "uuid",
            "topology",
            "source_commits",
            "cache_key",
            "ipc_fingerprints",
            "aliases",
            "routed_experts",
            "tensor_count",
            "nonempty_ipc_count",
            "empty_tensor_names",
            "empty_tensor_count",
            "no_allocation_tensor_names",
        ):
            require(first[field] == second[field], f"Restart changed {field}")
    return {
        "status": "pass",
        "ranks": 8,
        "runs": [str(path) for path in paths],
        "scope": "IPC storage/layout/restart identity only; no forward",
        "requires_external_exit_checks": "Both torchrun exit codes must be zero",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--master-port", type=int)
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("RUN_A", "RUN_B"))
    args = parser.parse_args()
    if args.compare:
        result = compare_runs(args.compare)
        with args.output.open("x") as file:
            json.dump(result, file, indent=2)
        return
    require(
        args.plan is not None and args.master_port is not None,
        "Probe requires --plan and --master-port",
    )
    report = {"status": "error", "scope": "native IPC load only; no forward"}
    rank = int(os.environ.get("RANK", "-1"))
    try:
        probe(args, report)
    except BaseException:  # noqa: BLE001 - preserve receipt on interruption too
        report.update(status="error", error=traceback.format_exc())
    finally:
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except BaseException:  # noqa: BLE001 - cleanup failure invalidates acceptance
            report.update(status="error", cleanup_error=traceback.format_exc())
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / f"rank-{rank}.json").open("x") as file:
            json.dump(report, file, indent=2, default=str)
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
