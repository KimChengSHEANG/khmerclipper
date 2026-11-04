"""Single-token TrOCR decoder step with a fixed-size self-attention KV cache.

Re-implements transformers' TrOCRDecoder forward (post-LN BART-style layers,
modeling_trocr.py 4.57) for ONE token per call, reusing the checkpoint's own
submodules so no weights are remapped. Self-attention keys/values live in the
k_cache/v_cache buffers, which coremltools converts to CoreML states
(ct.StateType). All shapes are static: attention always spans max_len
positions, with attn_mask hiding the unwritten ones, and the new K/V row is
blended in with a one-hot write_mask (no dynamic slicing).

Cross-attention K/V depend only on the image, so cross_kv() computes them once
per line (convert.py puts it in the encoder model) and every step takes them as
inputs. Recomputing them per step cost ~14 GFLOP/step on trocr-base.
"""
import torch
from torch import nn

# Weight edit, no training: an output-layer bias raising </s>, so lines end before the junk the model appends
# after real text. Same value as the desktop export (apps/desktop/tools/export_onnx.py); results in
# training/eval/results/weight_space.md.
EOS_LOGIT_BIAS = 3.0


class KVDecoderStep(nn.Module):
    def __init__(self, model, max_len: int, cache_dtype: torch.dtype = torch.float32):
        super().__init__()
        causal_lm = model.decoder            # TrOCRForCausalLM
        dec = causal_lm.model.decoder        # TrOCRDecoder
        self.embed_tokens = dec.embed_tokens  # already applies embed_scale
        self.layernorm_embedding = dec.layernorm_embedding
        self.layers = dec.layers
        self.output_projection = causal_lm.output_projection
        eos_bias = torch.zeros(self.output_projection.out_features)
        eos_bias[model.config.decoder.eos_token_id] = EOS_LOGIT_BIAS
        self.register_buffer("eos_bias", eos_bias, persistent=False)
        if dec.config.use_learned_position_embeddings:
            pos_table = dec.embed_positions.weight            # index = position + 2
            self.pos_offset = dec.embed_positions.offset
        else:
            pos_table = dec.embed_positions.weights           # index = padding_idx + 1 + position
            self.pos_offset = dec.embed_positions.padding_idx + 1
        self.register_buffer("pos_table", pos_table.detach().clone(), persistent=False)
        attn = dec.layers[0].self_attn
        self.num_heads, self.head_dim = attn.num_heads, attn.head_dim
        shape = (len(dec.layers), 1, self.num_heads, max_len, self.head_dim)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=cache_dtype))
        self.register_buffer("v_cache", torch.zeros(shape, dtype=cache_dtype))

    def _heads(self, x):  # [1, T, E] -> [1, H, T, D]
        return x.view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def _attend(self, q, k, v, mask):
        scores = q @ k.transpose(-1, -2)
        if mask is not None:
            scores = scores + mask
        out = scores.softmax(dim=-1) @ v  # [1, H, T, D]
        return out.transpose(1, 2).reshape(1, -1, self.num_heads * self.head_dim)

    def cross_kv(self, encoder_hidden_states):  # -> 2 x [L, 1, H, S, D]
        ks, vs = [], []
        for layer in self.layers:
            ca = layer.encoder_attn
            ks.append(self._heads(ca.k_proj(encoder_hidden_states)))
            vs.append(self._heads(ca.v_proj(encoder_hidden_states)))
        return torch.stack(ks), torch.stack(vs)

    def forward(self, input_ids, position, attn_mask, write_mask, cross_k, cross_v):
        h = self.embed_tokens(input_ids)
        h = h + self.pos_table.index_select(0, position + self.pos_offset).unsqueeze(0).to(h.dtype)
        if self.layernorm_embedding is not None:
            h = self.layernorm_embedding(h)
        keep = 1 - write_mask
        for i, layer in enumerate(self.layers):
            sa = layer.self_attn
            q = self._heads(sa.q_proj(h) * sa.scaling)
            k = self.k_cache[i : i + 1].squeeze(0).to(h.dtype) * keep + self._heads(sa.k_proj(h)) * write_mask
            v = self.v_cache[i : i + 1].squeeze(0).to(h.dtype) * keep + self._heads(sa.v_proj(h)) * write_mask
            self.k_cache[i : i + 1] = k.unsqueeze(0).to(self.k_cache.dtype)
            self.v_cache[i : i + 1] = v.unsqueeze(0).to(self.v_cache.dtype)
            h = layer.self_attn_layer_norm(h + sa.out_proj(self._attend(q, k, v, attn_mask)))

            ca = layer.encoder_attn
            q = self._heads(ca.q_proj(h) * ca.scaling)
            h = layer.encoder_attn_layer_norm(h + ca.out_proj(self._attend(q, cross_k[i], cross_v[i], None)))

            h = layer.final_layer_norm(h + layer.fc2(layer.activation_fn(layer.fc1(h))))
        return self.output_projection(h) + self.eos_bias.to(h.dtype)
