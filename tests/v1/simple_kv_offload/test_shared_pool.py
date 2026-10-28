# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import tempfile
import unittest

from vllm.v1.shared_kv_offload import shared_pool as m


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = m.PoolState(self.tmp.name)
        for owner in ("a", "b"):
            self.op(owner, "hello", "namespace", 4)

    def tearDown(self):
        self.tmp.cleanup()

    def op(self, owner, op, *a):
        return self.p.dispatch(owner, op, a)

    def store(self, key):
        slot, gen = self.op("a", "reserve", 1)[0]
        self.op("a", "finish", [[slot, gen, {key: None}]])
        return slot, gen

    def test_uncommitted_invisible_and_cross_client_hit(self):
        slot, gen = self.op("a", "reserve", 1)[0]
        self.assertEqual(self.op("b", "acquire", ["x"]), [])
        self.op("a", "finish", [[slot, gen, {"x": None}]])
        self.assertEqual(self.op("b", "acquire", ["x"])[0][1:3], [slot, gen])

    def test_read_lease_prevents_reuse(self):
        slot, gen = self.store("x")
        self.op("b", "acquire", ["x"])
        refs = self.op("a", "reserve", 4)
        self.assertNotIn(slot, [r[0] for r in refs])
        self.assertEqual(len(refs), 2)
        self.op("b", "finish", [[slot, gen, {}]])
        self.assertEqual(self.op("a", "reserve", 1)[0][0], slot)

    def test_duplicate_store_does_not_replace_pinned_object(self):
        slot, gen = self.store("x")
        self.op("b", "acquire", ["x"])
        s, g = self.op("a", "reserve", 1)[0]
        self.op("a", "finish", [[s, g, {"x": None}]])
        self.assertEqual(self.op("a", "lookup", ["x"])[0][1], slot)
        self.assertEqual(self.p.stats["deduplicated"], 1)

    def test_disconnect_quarantines_live_dma(self):
        slot, gen = self.op("a", "reserve", 1)[0]
        self.p.disconnect("a")
        self.assertEqual(self.p.records[slot]["state"], "quarantined")
        self.assertNotIn(slot, [r[0] for r in self.op("b", "reserve", 5)])

    def test_flush_keeps_readers_and_invalidates_pending_store(self):
        slot, gen = self.store("x")
        self.op("b", "acquire", ["x"])
        s, g = self.op("a", "reserve", 1)[0]
        self.op("a", "flush")
        self.assertEqual(self.op("a", "lookup", ["x"]), [])
        self.op("a", "finish", [[s, g, {"y": None}]])
        self.assertEqual(self.op("a", "lookup", ["y"]), [])
        self.assertIn(slot, self.p.records)
        self.op("b", "finish", [[slot, gen, {}]])
        self.assertNotIn(slot, self.p.records)

    def test_stale_generation_cannot_release_new_object(self):
        s, g = self.store("x")
        self.op("a", "flush")
        self.op("a", "reserve", 3)
        with self.assertRaises(ValueError):
            self.op("a", "finish", [[s, g, {}]])

    def test_role_contracts_are_independent_but_peers_must_match(self):
        self.op("a", "contract", "scheduler", {"group": "summary"})
        self.op("b", "contract", "worker", {"group": "physical"})
        self.op("b", "contract", "scheduler", {"group": "summary"})
        with self.assertRaises(ValueError):
            self.op("b", "contract", "scheduler", {"group": "different"})
        with self.assertRaises(ValueError):
            self.op("b", "contract", "worker", {"group": "different"})

    def test_namespace_and_layout_mismatch_fail_closed(self):
        with self.assertRaises(ValueError):
            self.op("c", "hello", "different", 4)
        self.op("a", "attach", [["kv", 128]])
        with self.assertRaises(ValueError):
            self.op("b", "attach", [["kv", 256]])


if __name__ == "__main__":
    unittest.main()
