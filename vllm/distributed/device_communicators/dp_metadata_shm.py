# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-node DP metadata exchange using the existing shared-memory queues."""

import time

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.logger import init_logger

logger = init_logger(__name__)


class DPMetadataShm:
    def __init__(self, group: dist.ProcessGroup, timeout: float = 60.0):
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.timeout = timeout
        self.sequence = 0
        self.queues = [
            MessageQueue.create_from_process_group(
                group, max_chunk_bytes=1024, max_chunks=2, writer_rank=rank
            )
            for rank in range(self.world_size)
        ]
        logger.info("DP metadata SHM enabled for %d ranks", self.world_size)

    @classmethod
    def create(cls, group: dist.ProcessGroup) -> "DPMetadataShm | None":
        from vllm.distributed.parallel_state import in_the_same_node_as

        if not all(in_the_same_node_as(group, source_rank=0)):
            logger.warning("DP metadata SHM requires one shared-memory domain")
            return None
        return cls(group)

    def all_gather(self, values: tuple[int, ...]) -> torch.Tensor:
        if len(values) != 6:
            raise ValueError("DP metadata must contain six integers")
        deadline = time.monotonic() + self.timeout
        sequence = self.sequence
        # Publish first on every rank, before waiting for any other writer.
        self.queues[self.rank].enqueue(
            (sequence, self.rank, values), timeout=self.timeout
        )
        rows = []
        for rank, queue in enumerate(self.queues):
            if rank == self.rank:
                rows.append(values)
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("DP metadata SHM exchange timed out")
            step, sender, received = queue.dequeue(timeout=remaining)
            if step != sequence or sender != rank or len(received) != 6:
                raise RuntimeError("DP metadata SHM sequence or sender mismatch")
            rows.append(received)
        self.sequence += 1
        return torch.tensor(rows, dtype=torch.int32, device="cpu").t().contiguous()

    def shutdown(self) -> None:
        for queue in self.queues:
            queue.shutdown()
        self.queues.clear()
