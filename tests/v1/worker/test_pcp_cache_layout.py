# SPDX-License-Identifier: Apache-2.0
"""Ownership, boundary and transfer behavior; no accelerator required."""

import unittest

from vllm.v1.pcp.layout import (
    PCPPageLayout,
    PCPTransferLease,
    plan_kda_transfers,
    plan_page_transfers,
)


class TestPCPPageLayout(unittest.TestCase):
    def test_every_slot_has_exactly_one_owner_including_block_zero(self):
        for size in (1, 2, 8):
            layout = PCPPageLayout(size, 64, 4)
            for slot in range(19 * 64):
                slots = [layout.local_slot(slot, rank) for rank in range(size)]
                self.assertEqual(sum(local >= 0 for local in slots), 1)
                owner, physical = layout.locate(slot // 64)
                self.assertEqual(layout.logical_block(owner, physical), slot // 64)
            for capacity in range(20):
                self.assertEqual(
                    sum(layout.local_capacity(capacity, rank) for rank in range(size)),
                    capacity,
                )
            self.assertEqual(layout.local_slot(-1, 0), -1)

    def test_chunks_cover_only_real_tokens_and_align_internal_boundaries(self):
        for size in (1, 2, 8):
            layout = PCPPageLayout(size, 64, 4)
            for start in (0, 1, 3, 63, 64, 65, 127):
                for length in (0, 1, 3, 63, 64, 65, 1025, 8193):
                    spans = layout.split_query(start, length)
                    actual = [
                        token
                        for span in spans
                        for token in range(span.start, span.stop)
                    ]
                    self.assertEqual(actual, list(range(start, start + length)))
                    self.assertTrue(
                        all(0 <= span.compute_rank < size for span in spans)
                    )
                    self.assertTrue(all(span.start % 4 == 0 for span in spans[1:]))
                    self.assertTrue(all(span.stop % 4 == 0 for span in spans[:-1]))

    def test_compute_balance_is_independent_of_storage_page_size(self):
        for size in (1, 2, 8):
            for page_tokens in (64, 768):
                layout = PCPPageLayout(size, page_tokens, 4)
                for start in (0, 1, 3, 767, 7680):
                    for length in (1, 4, 17, 512, 7680, 8192, 8193):
                        spans = layout.split_query(start, length)
                        counts = [
                            sum(s.stop - s.start for s in spans if s.compute_rank == r)
                            for r in range(size)
                        ]
                        self.assertLessEqual(max(counts) - min(counts), 8)
                        self.assertEqual(sum(counts), length)
                        for span in spans:
                            for token in range(span.start, span.stop):
                                owner, _ = layout.locate(token // page_tokens)
                                self.assertGreaterEqual(
                                    layout.local_slot(token, owner), 0
                                )
        layout = PCPPageLayout(8, 768, 4)
        # This is the steady 8K benchmark batch: one tail plus one new request.
        spans = layout.split_query(7680, 512) + layout.split_query(0, 7680)
        self.assertEqual(
            [
                sum(s.stop - s.start for s in spans if s.compute_rank == r)
                for r in range(8)
            ],
            [1024] * 8,
        )

    def test_decode_owner_receives_pages_from_all_prefill_owners(self):
        layout = PCPPageLayout(8, 64, 4)
        blocks = tuple(range(16, 24))
        for target in range(8):
            plan = plan_page_transfers(layout, "request", "mla", blocks, 509, target)
            self.assertEqual({p.source_rank for p in plan}, set(range(8)))
            self.assertEqual(sum(p.token_count for p in plan), 509)
            self.assertTrue(all(p.decode_dp_rank == target for p in plan))
            index = plan_page_transfers(layout, "request", "index", blocks, 509, target)
            self.assertEqual(sum(p.token_count for p in index), 508)

    def test_kda_heads_go_to_tp_shards_not_the_mla_dp_owner(self):
        for target in range(8):
            for prefill_tp, decode_tp in ((8, 8), (2, 8), (8, 2)):
                plan = plan_kda_transfers(
                    64,
                    prefill_tp,
                    decode_tp,
                    target,
                    request_id="request",
                    cache_kind="kda_recurrent",
                )
                heads = [
                    h
                    for p in plan
                    for h in range(p.head_start, p.head_start + p.head_count)
                ]
                self.assertEqual(sorted(heads), list(range(64)))
                self.assertEqual(
                    {p.destination_rank for p in plan}, set(range(decode_tp))
                )
                self.assertTrue(all(p.request_window == target for p in plan))

    def test_cancel_does_not_release_pages_while_reads_are_in_flight(self):
        lease = PCPTransferLease(frozenset(("mla", "index", "tail", "kda")))
        lease.complete("mla")
        lease.complete("mla")
        lease.cancel()
        self.assertFalse(lease.releasable)
        for name in ("index", "tail", "kda"):
            lease.complete(name)
        self.assertTrue(lease.releasable)
        self.assertFalse(lease.decode_ready)
        with self.assertRaises(ValueError):
            lease.complete("unknown")

    def test_decode_waits_for_every_payload(self):
        lease = PCPTransferLease(frozenset(("mla", "kda")))
        lease.complete("mla")
        self.assertFalse(lease.decode_ready)
        lease.complete("kda")
        self.assertTrue(lease.decode_ready)

    def test_invalid_geometry_and_incomplete_block_tables_are_rejected(self):
        for args in ((0, 64, 4), (8, 63, 4), (8, 64, 0)):
            with self.assertRaises(ValueError):
                PCPPageLayout(*args)
        layout = PCPPageLayout(8, 64, 4)
        for blocks, tokens in (((1,), 65), ((1, 1), 128), ((-1,), 64)):
            with self.assertRaises(ValueError):
                plan_page_transfers(layout, "req", "mla", blocks, tokens, 0)


if __name__ == "__main__":
    unittest.main()


def test_tail_and_kda_payloads_preserve_request_and_decode_dp_identity():
    from vllm.v1.pcp.layout import plan_tail_transfer

    layout = PCPPageLayout(8, 64, 4)
    for target in range(8):
        for length in (0, 1, 3, 4, 7, 8, 509):
            tail = plan_tail_transfer(layout, "request-a", 23, length, 8, target)
            if length % 4 == 0:
                assert tail is None
            else:
                assert tail.request_id == "request-a"
                assert tail.cache_kind == "index_tail"
                assert (tail.source_rank, tail.source_block) == (7, 2)
                assert tail.decode_dp_rank == target
                assert tail.source_ring_offsets == tuple(
                    position % 8 for position in range(length - length % 4, length)
                )
        for kind in ("kda_recurrent", "kda_conv"):
            transfers = plan_kda_transfers(
                64, 8, 8, target, request_id="request-a", cache_kind=kind
            )
            assert len(transfers) == 8
            assert all(
                item.request_id == "request-a"
                and item.cache_kind == kind
                and item.request_window == target
                and item.version == 1
                for item in transfers
            )
            assert [item.destination_rank for item in transfers] == list(range(8))
