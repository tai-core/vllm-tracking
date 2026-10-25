# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_musa(), reason="MUSA FP8 MLA gather"
)


@pytest.fixture(scope="module", autouse=True)
def load_native_kernels():
    current_platform.import_kernels()


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("with_offsets", [False, True])
def test_fp8_mla_gather_matches_packed_cache(strided, with_offsets):
    torch.manual_seed(123)
    blocks, block_size = 8, 16
    packed = torch.zeros((blocks, block_size, 656), dtype=torch.uint8)
    values = torch.randn(blocks, block_size, 512).to(torch.float8_e4m3fn)
    scales = torch.rand(blocks, block_size, 4) + 0.1
    rope = torch.randn(blocks, block_size, 64).to(torch.bfloat16)
    packed[:, :, :512] = values.view(torch.uint8)
    packed[:, :, 512:528] = scales.view(torch.uint8)
    packed[:, :, 528:] = rope.view(torch.uint8)
    reference = torch.cat(
        [(values.float() * scales.repeat_interleave(128, dim=-1)).bfloat16(), rope],
        dim=-1,
    )
    cache_stride = block_size * 656 + (4096 if strided else 0)
    storage = torch.empty(blocks * cache_stride, device="musa", dtype=torch.uint8)
    cache = storage.as_strided(packed.shape, (cache_stride, 656, 1))
    cache.copy_(packed)
    # Duplicate start exercises an empty request. Sequence offsets cross pages.
    starts = torch.tensor([0, 0, 19, 28], dtype=torch.int32)
    offsets = torch.tensor([0, 7, 13, 1], dtype=torch.int32)
    tables = torch.tensor(
        [[0, 1, 2], [3, 1, 6], [7, 2, 0], [4, 5, 6]], dtype=torch.int32
    )
    n = 45
    output_storage = torch.full(
        (n, 600 if strided else 576), -99, device="musa", dtype=torch.bfloat16
    )
    output = output_storage[:, :576]
    torch.ops._C_cache_ops.cp_gather_and_upconvert_fp8_kv_cache(
        cache,
        output,
        tables.to("musa"),
        starts.to("musa"),
        4,
        offsets.to("musa") if with_offsets else None,
    )
    expected = []
    for req, (lo, hi) in enumerate(zip(starts.tolist(), [0, 19, 28, n])):
        for i in range(hi - lo):
            pos = i + (int(offsets[req]) if with_offsets else 0)
            expected.append(reference[tables[req, pos // block_size], pos % block_size])
    torch.testing.assert_close(output.cpu(), torch.stack(expected), rtol=0, atol=0)
    if strided:
        assert torch.all(output_storage[:, 576:] == -99)


def test_fp8_mla_gather_empty_output():
    torch.ops._C_cache_ops.cp_gather_and_upconvert_fp8_kv_cache(
        torch.empty((1, 16, 656), device="musa", dtype=torch.uint8),
        torch.empty((0, 576), device="musa", dtype=torch.bfloat16),
        torch.empty((0, 1), device="musa", dtype=torch.int32),
        torch.empty(0, device="musa", dtype=torch.int32),
        0,
        None,
    )


@pytest.mark.parametrize("host_rows", [False, True])
def test_fp8_mla_gather_remapped_rows(host_rows):
    current_platform.import_kernels()
    packed = torch.zeros((2, 16, 656), dtype=torch.uint8)
    values = torch.arange(32).float()[:, None].expand(32, 512).contiguous()
    packed[:, :, :512] = (
        values.to(torch.float8_e4m3fn).view(torch.uint8).view(2, 16, 512)
    )
    packed[:, :, 512:528] = torch.ones(2, 16, 4).view(torch.uint8)
    cache = packed.to("musa")
    host = packed.view(32, 656).pin_memory()
    mapping = torch.arange(31, -1, -1, dtype=torch.int32)
    device_ids = mapping.clone()
    if host_rows:
        device_ids[::2] = -1
    out = torch.empty((32, 576), device="musa", dtype=torch.bfloat16)
    torch.ops._C_cache_ops.cp_gather_and_upconvert_fp8_kv_cache(
        cache,
        out,
        torch.tensor([[0, 1]], device="musa", dtype=torch.int32),
        torch.tensor([0], device="musa", dtype=torch.int32),
        1,
        seq_starts=None,
        host_cache=host,
        host_row_ids=mapping.to("musa"),
        device_row_ids=device_ids.to("musa"),
    )
    expected = torch.zeros(32, 576, dtype=torch.bfloat16)
    expected[:, :512] = (
        values.to(torch.float8_e4m3fn).float()[mapping.long()].bfloat16()
    )
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_fp8_mla_gather_all_e4m3_bit_patterns():
    # Includes signed zeros, subnormals, maximum finite values, and NaNs.
    codes = torch.arange(256, dtype=torch.uint8).repeat(2)
    packed = torch.zeros((1, 16, 656), dtype=torch.uint8)
    packed[:, :, :512] = codes
    scales = torch.tensor([0.125, 1.0, 3.0, 0.013]).expand(1, 16, 4).contiguous()
    packed[:, :, 512:528] = scales.view(torch.uint8)
    expected = torch.zeros((16, 576), dtype=torch.bfloat16)
    expected[:, :512] = (
        codes.view(torch.float8_e4m3fn).float()
        * scales[0].repeat_interleave(128, dim=-1)
    ).bfloat16()
    out = torch.empty_like(expected, device="musa")
    torch.ops._C_cache_ops.cp_gather_and_upconvert_fp8_kv_cache(
        packed.to("musa"),
        out,
        torch.zeros((1, 1), device="musa", dtype=torch.int32),
        torch.zeros(1, device="musa", dtype=torch.int32),
        1,
        None,
    )
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0, equal_nan=True)
