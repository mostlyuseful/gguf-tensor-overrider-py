"""Unit tests for collect_tensors_in_groups helper.

These tests focus on the grouping heuristics:
- Only tensors with names containing the pattern 'blk.<id>.' are considered for grouping.
- Attention-related tensors (various naming schemes) within the same block are grouped
  together if at least two such tensors are present for that block.
- Groups are emitted in order of first appearance; non-group tensors are emitted as
  single-item groups preserving original order.
- Groups are sorted deterministically by tensor name.
"""
from __future__ import annotations

from gguf_tensor_overrider_py.core import collect_tensors_in_groups
from gguf_tensor_overrider_py.models import TensorInfo


def names(grouped):
    """Flatten grouped tensor structure into list of lists of names for readability."""
    return [[t.name for t in g] for g in grouped]


def flatten(grouped):
    return [t.name for grp in grouped for t in grp]


def test_empty_input_returns_empty_list():
    assert collect_tensors_in_groups([]) == []


def test_single_block_attention_group_created_and_order_preserved():
    # Input order: q, v, ffn, k, o  (k/o appear after first two attention tokens)
    tensors = [
        TensorInfo("blk.0.attn_q.weight", 1),
        TensorInfo("blk.0.attn_v.weight", 1),
        TensorInfo("blk.0.ffn_down.weight", 1),  # non-attention should remain separate
        TensorInfo("blk.0.attn_k.weight", 1),
        TensorInfo("blk.0.attn_o.weight", 1),
    ]
    groups = collect_tensors_in_groups(tensors)
    # First group should contain all 4 attention tensors sorted by name
    first = groups[0]
    first_names = [t.name for t in first]
    assert set(first_names) == {
        "blk.0.attn_q.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_o.weight",
    }
    assert first_names == sorted(first_names)  # deterministic ordering
    # Second group is the ffn tensor by itself
    assert names(groups)[1] == ["blk.0.ffn_down.weight"]
    # All tensors appear exactly once
    assert sorted(flatten(groups)) == sorted(t.name for t in tensors)


def test_multiple_blocks_each_form_group():
    tensors = [
        # Block 0 attention (first appearance triggers group emission)
        TensorInfo("blk.0.attn_q.weight", 1),
        TensorInfo("blk.0.attn_v.weight", 1),
        TensorInfo("blk.0.other.weight", 1),
        # Block 1 attention
        TensorInfo("blk.1.attn_k.weight", 1),
        TensorInfo("blk.1.attn_q.weight", 1),
        TensorInfo("blk.1.misc.weight", 1),
    ]
    groups = collect_tensors_in_groups(tensors)
    grouped_names = names(groups)
    # Expect 4 groups: block0 attention group, block0 other, block1 attention group, block1 misc
    assert len(grouped_names) == 4
    assert set(grouped_names[0]) == {"blk.0.attn_q.weight", "blk.0.attn_v.weight"}
    assert grouped_names[1] == ["blk.0.other.weight"]
    assert set(grouped_names[2]) == {"blk.1.attn_k.weight", "blk.1.attn_q.weight"}
    assert grouped_names[3] == ["blk.1.misc.weight"]
    # Ensure no duplication
    assert sorted(flatten(groups)) == sorted(t.name for t in tensors)


def test_single_attention_tensor_not_grouped():
    tensors = [
        TensorInfo("blk.2.attn_q.weight", 1),  # Only one attention tensor in block 2
        TensorInfo("blk.2.other.weight", 1),
    ]
    groups = collect_tensors_in_groups(tensors)
    assert names(groups) == [["blk.2.attn_q.weight"], ["blk.2.other.weight"]]


def test_non_block_or_global_attention_not_grouped():
    # Names lacking 'blk.<id>.' should never be grouped regardless of attention tokens
    tensors = [
        TensorInfo("attention.wq.weight", 1),  # global / no blk prefix
        TensorInfo("attention.wk.weight", 1),
        TensorInfo("some_global.weight", 1),
    ]
    groups = collect_tensors_in_groups(tensors)
    assert names(groups) == [[t.name] for t in tensors]


def test_mixed_attention_variants_qkv_proj_tokens():
    # Use variant tokens like q_proj/k_proj/v_proj/o_proj to ensure they are detected
    tensors = [
        TensorInfo("blk.5.q_proj.weight", 1),
        TensorInfo("blk.5.k_proj.weight", 1),
        TensorInfo("blk.5.v_proj.weight", 1),
        TensorInfo("blk.5.o_proj.weight", 1),
        TensorInfo("blk.5.unrelated.weight", 1),
    ]
    groups = collect_tensors_in_groups(tensors)
    grouped_names = names(groups)
    # First group should have 4 projection tensors
    assert set(grouped_names[0]) == {
        "blk.5.k_proj.weight",
        "blk.5.o_proj.weight",
        "blk.5.q_proj.weight",
        "blk.5.v_proj.weight",
    }
    # Next group is unrelated tensor
    assert grouped_names[1] == ["blk.5.unrelated.weight"]


def test_group_appears_where_first_attention_tensor_occurs():
    # Attention tensors not contiguous; group should still emit at first occurrence index position
    tensors = [
        TensorInfo("blk.3.other0.weight", 1),
        TensorInfo("blk.3.attn_q.weight", 1),
        TensorInfo("blk.3.other1.weight", 1),
        TensorInfo("blk.3.attn_v.weight", 1),
        TensorInfo("blk.3.attn_k.weight", 1),
    ]
    groups = collect_tensors_in_groups(tensors)
    grouped_names = names(groups)
    # Order should be: other0 (single), attention group (q,k,v sorted), other1 (single)
    assert grouped_names[0] == ["blk.3.other0.weight"]
    assert set(grouped_names[1]) == {
        "blk.3.attn_q.weight",
        "blk.3.attn_v.weight",
        "blk.3.attn_k.weight",
    }
    # other1 appears after group because original single appears between attention members but is skipped until its turn
    # After emitting attention group, the remaining other1 should appear
    # The function's current logic will emit the group upon hitting first attention tensor (attn_q),
    # so other1 (which appears after but before other attention tensors) should become its own group afterwards.
    # Thus total groups = 3.
    assert grouped_names[2] == ["blk.3.other1.weight"]


def test_each_tensor_emitted_exactly_once_under_complex_scenario():
    tensors = [
        TensorInfo("blk.7.attn_q.weight", 1),
        TensorInfo("blk.7.attn_v.weight", 1),
        TensorInfo("blk.7.misc.weight", 1),
        TensorInfo("blk.8.attn_k.weight", 1),
        TensorInfo("blk.8.single_attention_only.weight", 1),  # won't group (only one)
        TensorInfo("blk.9.k_proj.weight", 1),
        TensorInfo("blk.9.v_proj.weight", 1),
        TensorInfo("blk.9.o_proj.weight", 1),
        TensorInfo("global.norm.weight", 1),  # global non-attention
    ]
    groups = collect_tensors_in_groups(tensors)
    all_names = flatten(groups)
    assert sorted(all_names) == sorted(t.name for t in tensors)
    # Ensure counts match
    assert len(all_names) == len(tensors)
