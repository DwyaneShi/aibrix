import numpy as np
import pytest

from aibrix_kvcache.cache_args import (
    extract_pin_unpin_kwargs,
    parse_kvcache_api_args,
)
from aibrix_kvcache.cache_hashable import (
    TokenListView,
    BlockHashes,
    KVCacheKey,
)


def make_token_views():
    data = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    prefix = TokenListView(data, 0, 3)
    query = TokenListView(data, 3, 7)
    cache_key = KVCacheKey(prefix, query)
    return prefix, query, cache_key


def make_block_hashes():
    block_ntokens = 4
    prefix = BlockHashes(["p0", "p1"], block_ntokens)
    query = BlockHashes(["q0", "q1"], block_ntokens)
    cache_key = KVCacheKey(prefix, query)
    return prefix, query, cache_key


@pytest.mark.parametrize("maker", [make_token_views, make_block_hashes])
def test_kwargs_prefix_query(maker):
    prefix, query, _ = maker()
    p, q, kt = parse_kvcache_api_args(prefix=prefix, query=query)
    assert p == prefix and q == query and kt is None
    dummy = object()
    p, q, kt = parse_kvcache_api_args(
        prefix=prefix, query=query, kv_tensors=dummy
    )
    assert p == prefix and q == query and kt is dummy


@pytest.mark.parametrize("maker", [make_token_views, make_block_hashes])
def test_kwargs_cache_key(maker):
    _, _, cache_key = maker()
    p, q, kt = parse_kvcache_api_args(cache_key=cache_key)
    assert p == cache_key.prefix and q == cache_key.query and kt is None
    dummy = object()
    p, q, kt = parse_kvcache_api_args(cache_key=cache_key, kv_tensors=dummy)
    assert p == cache_key.prefix and q == cache_key.query and kt is dummy


@pytest.mark.parametrize("maker", [make_token_views, make_block_hashes])
def test_positional_1_arg_cache_key(maker):
    _, _, cache_key = maker()
    p, q, kt = parse_kvcache_api_args(cache_key)
    assert p == cache_key.prefix and q == cache_key.query and kt is None


@pytest.mark.parametrize("maker", [make_token_views, make_block_hashes])
def test_positional_2_args_prefix_query(maker):
    prefix, query, _ = maker()
    p, q, kt = parse_kvcache_api_args(prefix, query)
    assert p == prefix and q == query and kt is None


@pytest.mark.parametrize("maker", [make_token_views, make_block_hashes])
def test_positional_2_args_cache_key_kv(maker):
    _, _, cache_key = maker()
    dummy = object()
    p, q, kt = parse_kvcache_api_args(cache_key, dummy)
    assert p == cache_key.prefix and q == cache_key.query and kt is dummy


@pytest.mark.parametrize("maker", [make_token_views, make_block_hashes])
def test_positional_3_args_prefix_query_kv(maker):
    prefix, query, _ = maker()
    dummy = object()
    p, q, kt = parse_kvcache_api_args(prefix, query, dummy)
    assert p == prefix and q == query and kt is dummy


@pytest.mark.parametrize("maker", [make_token_views, make_block_hashes])
def test_invalid_signatures(maker):
    prefix, query, cache_key = maker()
    dummy = object()
    with pytest.raises((AssertionError, ValueError)):
        parse_kvcache_api_args(prefix, dummy)
    with pytest.raises((AssertionError, ValueError)):
        parse_kvcache_api_args(prefix=prefix, kv_tensors=dummy)
    with pytest.raises(ValueError):
        parse_kvcache_api_args(prefix=prefix, query=query, pin=True)
    with pytest.raises(ValueError):
        parse_kvcache_api_args(cache_key=cache_key, unpin=True)
    with pytest.raises(ValueError):
        parse_kvcache_api_args(prefix, query, True)
    with pytest.raises(ValueError):
        parse_kvcache_api_args(cache_key, True)
    with pytest.raises(ValueError):
        parse_kvcache_api_args(prefix, query, dummy, True)
    with pytest.raises(ValueError):
        parse_kvcache_api_args(cache_key, dummy, True)


@pytest.mark.parametrize("maker", [make_token_views, make_block_hashes])
def test_extract_pin_unpin_kwargs(maker):
    prefix, query, _ = maker()
    kwargs, pin, unpin = extract_pin_unpin_kwargs(
        {"pin": True, "prefix": prefix, "query": query},
    )
    assert kwargs == {"prefix": prefix, "query": query}
    assert pin is True and unpin is False
    kwargs, pin, unpin = extract_pin_unpin_kwargs(
        {"unpin": True, "prefix": prefix, "query": query},
    )
    assert kwargs == {"prefix": prefix, "query": query}
    assert pin is False and unpin is True
    with pytest.raises(ValueError):
        extract_pin_unpin_kwargs({"pin": True, "unpin": True})
    kwargs, pin, unpin = extract_pin_unpin_kwargs({"unpin": True})
    assert kwargs == {}
    assert pin is False and unpin is True
