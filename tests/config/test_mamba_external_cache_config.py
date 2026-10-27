# SPDX-License-Identifier: Apache-2.0
"""Run the Mamba normalization and public VllmConfig validators in order."""

from unittest.mock import Mock

import pytest

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.model_executor.models.config import MambaModelConfig


def make_config(*, apc=False, mode="align", external=True, mamba_block_size=None):
    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        cache_config=CacheConfig(
            enable_prefix_caching=apc,
            mamba_cache_mode=mode,
            block_size=256,
            mamba_block_size=mamba_block_size,
        ),
        scheduler_config=SchedulerConfig(
            enable_chunked_prefill=True, max_model_len=32768, is_encoder_decoder=False
        ),
    )
    # Model metadata only; no checkpoint/native imports are needed to run the
    # genuine CacheConfig/MambaModelConfig/VllmConfig validation methods.
    config.model_config = Mock(
        spec=ModelConfig,
        architecture="MambaForCausalLM",
        max_model_len=32768,
        supports_mamba_prefix_caching=True,
    )
    config.kv_transfer_config = (
        KVTransferConfig(kv_connector="ExampleConnector", kv_role="kv_consumer")
        if external
        else None
    )
    return config


@pytest.mark.parametrize(
    "apc,external,mode,expected_mode,expected_block",
    [
        (False, True, "align", "align", 256),
        (False, False, "align", "none", 32768),
        (False, True, "none", "none", 32768),
        (False, True, "all", "none", 32768),
        (True, False, "none", "align", 256),
    ],
)
def test_mamba_normalization_preserves_only_explicit_external_align(
    apc, external, mode, expected_mode, expected_block
):
    config = make_config(apc=apc, external=external, mode=mode)
    assert isinstance(config, VllmConfig) and isinstance(
        config.cache_config, CacheConfig
    )
    MambaModelConfig.verify_and_update_config(config)
    assert config.cache_config.mamba_cache_mode == expected_mode
    assert config.cache_config.mamba_block_size == expected_block
    assert not config.cache_config.user_specified_mamba_block_size
    assert config.validate_mamba_block_size() is config
    assert config.validate_mamba_cached_kernel() is config
    config.validate_block_size()


def test_inactive_transfer_config_does_not_preserve_align():
    config = make_config()
    config.kv_transfer_config = KVTransferConfig()
    MambaModelConfig.verify_and_update_config(config)
    assert config.cache_config.mamba_cache_mode == "none"
    assert config.cache_config.mamba_block_size == 32768


def test_auto_mamba_block_is_not_reclassified_as_a_user_override():
    config = make_config()
    MambaModelConfig.verify_and_update_config(config)
    assert config.cache_config.mamba_block_size == 256
    assert config.validate_mamba_block_size() is config
    # The platform finalizes this automatic value after finding the backend.
    config.cache_config.mamba_block_size = 1024
    assert not config.cache_config.user_specified_mamba_block_size
    assert config.validate_mamba_block_size() is config


def test_explicit_mamba_block_still_requires_local_prefix_caching():
    config = make_config(mamba_block_size=1024)
    MambaModelConfig.verify_and_update_config(config)
    assert config.cache_config.user_specified_mamba_block_size
    with pytest.raises(ValueError, match="--mamba-block-size"):
        config.validate_mamba_block_size()
    config.cache_config.enable_prefix_caching = True
    assert config.validate_mamba_block_size() is config
