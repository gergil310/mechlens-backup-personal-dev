"""Gemma-3 HookedTransformer vs HuggingFace parity guard for RMSNorm epsilon placement.

Regression test for the v2.16.0 Gemma-3 miss (issue #1121, hotfixed in v2.16.1): the
conversion path placed the RMSNorm epsilon outside the rsqrt, so logits drifted past the
1e-4 acceptance threshold on longer prompts. Gemma applies the norm as
``x * rsqrt(mean(x^2) + eps) * (1 + weight)`` with eps INSIDE the rsqrt.

The test exercises the real conversion path (``convert_gemma_weights`` followed by the
same ``load_and_process_state_dict`` flags ``from_pretrained_no_processing`` uses) against
a tiny random-init ``Gemma3ForCausalLM``. No HuggingFace Hub access is needed, so it runs
in CI. The model is built with non-trivial norm weights and checked at several prompt
lengths, including one close to ``n_ctx``, because the original bug only crossed the
threshold on a longer prompt.
"""

import pytest
import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.components.rms_norm import RMSNorm
from transformer_lens.pretrained.weight_conversions.gemma import convert_gemma_weights

ACCEPTANCE_ATOL = 1e-4

D_MODEL, D_HEAD, N_HEADS, N_KV_HEADS, D_MLP, N_LAYERS, D_VOCAB, N_CTX = (
    32,
    16,
    2,
    1,
    64,
    4,
    256,
    64,
)
EPS = 1e-6
WINDOW = 8
HF_LAYER_TYPES = ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"]


@pytest.fixture(scope="module")
def hf_model() -> Gemma3ForCausalLM:
    torch.manual_seed(0)
    cfg = Gemma3TextConfig(
        vocab_size=D_VOCAB,
        hidden_size=D_MODEL,
        intermediate_size=D_MLP,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        head_dim=D_HEAD,
        max_position_embeddings=N_CTX,
        sliding_window=WINDOW,
        layer_types=HF_LAYER_TYPES,
        query_pre_attn_scalar=D_HEAD,
        rms_norm_eps=EPS,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
    )
    # Eager attention matches the HookedTransformer reference; SDPA differs at fp32 noise level.
    cfg._attn_implementation = "eager"
    model = Gemma3ForCausalLM(cfg).eval()
    # HF initialises Gemma norm weights to zero, which hides the (1 + weight) offset and
    # makes eps placement nearly invisible. Give every norm a non-trivial weight.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "norm" in name:
                param.normal_(mean=0.0, std=0.5)
    return model


@pytest.fixture(scope="module")
def ht_model(hf_model: Gemma3ForCausalLM) -> HookedTransformer:
    # Mirrors the Gemma-3 entries in loading_from_pretrained.convert_hf_model_config,
    # scaled down to the tiny HF config above.
    cfg = HookedTransformerConfig(
        d_model=D_MODEL,
        d_head=D_HEAD,
        n_heads=N_HEADS,
        n_key_value_heads=N_KV_HEADS,
        d_mlp=D_MLP,
        n_layers=N_LAYERS,
        n_ctx=N_CTX,
        d_vocab=D_VOCAB,
        eps=EPS,
        act_fn="gelu_pytorch_tanh",
        normalization_type="RMS",
        positional_embedding_type="rotary",
        rotary_dim=D_HEAD,
        rotary_base=1_000_000,
        rotary_base_local=10_000,
        use_attn_scale=True,
        gated_mlp=True,
        final_rms=True,
        use_normalization_before_and_after=True,
        use_qk_norm=True,
        window_size=WINDOW,
        use_local_attn=True,
        attn_types=["local" if t == "sliding_attention" else "global" for t in HF_LAYER_TYPES],
        original_architecture="Gemma3ForCausalLM",
    )
    model = HookedTransformer(cfg)
    # Same flags as HookedTransformer.from_pretrained_no_processing.
    model.load_and_process_state_dict(
        convert_gemma_weights(hf_model, cfg),
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    )
    return model.eval()


def _tokens(seq_len: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, D_VOCAB, (1, seq_len), generator=generator)


def test_rmsnorm_component_matches_gemma3_rmsnorm_where_eps_dominates(
    hf_model: Gemma3ForCausalLM,
) -> None:
    """Direct check of the norm itself on inputs small enough that eps placement matters.

    With ``mean(x^2)`` of order eps, ``rsqrt(var + eps)`` and ``rsqrt(var) + eps``-style
    orderings differ by a large factor, so this fails loudly on any placement regression
    regardless of what the rest of the model does.
    """
    hf_norm = hf_model.model.layers[0].input_layernorm
    cfg = HookedTransformerConfig(
        d_model=D_MODEL, d_head=D_HEAD, n_heads=N_HEADS, n_layers=1, n_ctx=N_CTX, eps=EPS
    )
    ht_norm = RMSNorm(cfg)
    with torch.no_grad():
        # Conversion-table semantics: HT stores (1 + weight).
        ht_norm.w.copy_(hf_norm.weight.float() + 1.0)
    generator = torch.Generator().manual_seed(1)
    for scale in (1.0, 1e-2, 1e-3):
        x = torch.randn(2, 5, D_MODEL, generator=generator) * scale
        with torch.no_grad():
            torch.testing.assert_close(ht_norm(x), hf_norm(x), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("seq_len", [8, 32, N_CTX - 4])
def test_logits_match_hf_within_acceptance_tolerance(
    hf_model: Gemma3ForCausalLM, ht_model: HookedTransformer, seq_len: int
) -> None:
    tokens = _tokens(seq_len, seed=seq_len)
    with torch.no_grad():
        hf_logits = hf_model(tokens).logits
        ht_logits = ht_model(tokens)
    max_diff = (ht_logits - hf_logits).abs().max().item()
    assert max_diff < ACCEPTANCE_ATOL, (
        f"Gemma-3 HookedTransformer logits diverge from HF by {max_diff:.3e} at seq_len="
        f"{seq_len} (threshold {ACCEPTANCE_ATOL:.0e}). Check RMSNorm eps placement first."
    )


def test_residual_stream_matches_hf_hidden_states(
    hf_model: Gemma3ForCausalLM, ht_model: HookedTransformer
) -> None:
    """Per-layer intermediate check so a drift is localised to the block that introduces it."""
    tokens = _tokens(N_CTX - 4, seed=7)
    with torch.no_grad():
        hf_hidden = hf_model(tokens, output_hidden_states=True).hidden_states
        _, cache = ht_model.run_with_cache(tokens)
    torch.testing.assert_close(cache["hook_embed"], hf_hidden[0], atol=ACCEPTANCE_ATOL, rtol=0)
    # HF applies the final norm to its last hidden state, so compare block outputs only.
    for layer in range(N_LAYERS - 1):
        max_diff = (cache[f"blocks.{layer}.hook_resid_post"] - hf_hidden[layer + 1]).abs().max()
        assert max_diff.item() < ACCEPTANCE_ATOL, (
            f"blocks.{layer}.hook_resid_post diverges from HF hidden_states[{layer + 1}] by "
            f"{max_diff.item():.3e}"
        )
