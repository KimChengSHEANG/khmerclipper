"""Export a TrOCR checkpoint to encoder.onnx + decoder.onnx + tokenizer.json.

Mirrors the mac app's Vendor/trocr/convert.py (CoreML) and export_tokenizer.py, but emits
ONNX for khmerclipper.ocr.trocr. Needs the `tools` extra:
    .venv/bin/pip install -e '.[tools]'
    .venv/bin/python tools/export_onnx.py <checkpoint dir or HF id> exported/khmer
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers.models.trocr.modeling_trocr import TrOCRSinusoidalPositionalEmbedding

PARITY_TOLERANCE = 1e-3
# Weight edit, no training: an output-layer bias raising </s> by this much ends lines before the junk the
# model appends after real text. M2 errors on the six eval sets: +2, +3, +4 -> noisy 1532, 1489, 1694 (1583
# without); +3 is no worse anywhere (training/eval/results/weight_space.md).
EOS_LOGIT_BIAS = 3.0


class Reference(torch.nn.Module):
    """Full-sequence decoder, no cache: the parity reference."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, input_ids):
        hidden = self.model.encoder(pixel_values=pixel_values).last_hidden_state
        return self.model.decoder(input_ids=input_ids, encoder_hidden_states=hidden, use_cache=False).logits


class Encoder(torch.nn.Module):
    """Images -> the decoder's cross-attention K/V (2 x [L, B, H, S, D]), once per line."""

    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.layers = model.decoder.model.decoder.layers

    def forward(self, pixel_values):
        h = self.encoder(pixel_values=pixel_values).last_hidden_state
        ks = [heads(layer.encoder_attn, layer.encoder_attn.k_proj(h)) for layer in self.layers]
        vs = [heads(layer.encoder_attn, layer.encoder_attn.v_proj(h)) for layer in self.layers]
        return torch.stack(ks), torch.stack(vs)


def heads(attn, x):  # [B, T, E] -> [B, H, T, D]
    return x.view(x.shape[0], -1, attn.num_heads, attn.head_dim).transpose(1, 2)


def interleave(ks, vs):  # [L, ...] x 2 -> k0, v0, k1, v1, ...
    return [t for kv in zip(ks, vs) for t in kv]


def attend(attn, q, k, v):
    out = (q @ k.transpose(-1, -2)).softmax(dim=-1) @ v
    return out.transpose(1, 2).reshape(out.shape[0], -1, attn.num_heads * attn.head_dim)


class DecoderStep(torch.nn.Module):
    """One token per line per call, B lines in lockstep: self-attention K/V of the earlier
    tokens come in as past_k/past_v ([L, B, H, t, D], t = 0 on the first step) and go out one
    longer; `position` is shared. Cross K/V come in per layer (cross_k_0, cross_v_0, ...): a
    stacked input costs a Gather copy of every layer's K/V on every step. Re-implements
    transformers' TrOCRDecoder layer (post-LN) like the mac app's kv_decoder.py."""

    def __init__(self, model):
        super().__init__()
        dec = model.decoder.model.decoder
        self.embed_tokens, self.layernorm_embedding, self.layers = dec.embed_tokens, dec.layernorm_embedding, dec.layers
        self.output_projection = model.decoder.output_projection
        self.eos_bias = torch.zeros(self.output_projection.out_features)
        self.eos_bias[model.config.decoder.eos_token_id] = EOS_LOGIT_BIAS
        if dec.config.use_learned_position_embeddings:
            self.pos_table, self.pos_offset = dec.embed_positions.weight, dec.embed_positions.offset
        else:
            self.pos_table, self.pos_offset = dec.embed_positions.weights, dec.embed_positions.padding_idx + 1
        # Same function, edited weights: each self-attention's Q, K, V as one [3E, E] matmul (one
        # activation quantization instead of three in the int8 decoder), attention scaling folded into Q.
        self.qkv, self.cq = torch.nn.ModuleList(), torch.nn.ModuleList()
        for layer in self.layers:
            sa, ca = layer.self_attn, layer.encoder_attn
            fused = torch.nn.Linear(sa.embed_dim, 3 * sa.embed_dim)
            fused.weight.data = torch.cat([sa.q_proj.weight * sa.scaling, sa.k_proj.weight, sa.v_proj.weight])
            fused.bias.data = torch.cat([sa.q_proj.bias * sa.scaling, sa.k_proj.bias, sa.v_proj.bias])
            self.qkv.append(fused)
            cq = torch.nn.Linear(ca.embed_dim, ca.embed_dim)
            cq.weight.data, cq.bias.data = ca.q_proj.weight * ca.scaling, ca.q_proj.bias * ca.scaling
            self.cq.append(cq)

    def forward(self, input_ids, position, past_k, past_v, *cross):
        cross_k, cross_v = cross[0::2], cross[1::2]
        h = self.embed_tokens(input_ids) + self.pos_table.index_select(0, position + self.pos_offset).unsqueeze(0)
        if self.layernorm_embedding is not None:
            h = self.layernorm_embedding(h)
        new_k, new_v = [], []
        for i, layer in enumerate(self.layers):
            sa = layer.self_attn
            q, k, v = self.qkv[i](h).chunk(3, dim=-1)
            k = torch.cat([past_k[i], heads(sa, k)], dim=2)
            v = torch.cat([past_v[i], heads(sa, v)], dim=2)
            new_k.append(k)
            new_v.append(v)
            h = layer.self_attn_layer_norm(h + sa.out_proj(attend(sa, heads(sa, q), k, v)))
            ca = layer.encoder_attn
            h = layer.encoder_attn_layer_norm(
                h + ca.out_proj(attend(ca, heads(ca, self.cq[i](h)), cross_k[i], cross_v[i])))
            h = layer.final_layer_norm(h + layer.fc2(layer.activation_fn(layer.fc1(h))))
        return self.output_projection(h) + self.eos_bias, torch.stack(new_k), torch.stack(new_v)


def load_model(ckpt: str) -> VisionEncoderDecoderModel:
    model = VisionEncoderDecoderModel.from_pretrained(ckpt, low_cpu_mem_usage=False)
    # trocr-large-style sinusoidal position tables can be left on the meta device; rebuild them.
    for mod in model.modules():
        if isinstance(mod, TrOCRSinusoidalPositionalEmbedding) and mod.weights is not None and mod.weights.is_meta:
            mod.weights = mod.get_embedding(mod.weights.shape[0], mod.embedding_dim, mod.padding_idx)
    return model.eval()


def decoder_start_id(model) -> int:
    # Top-level config wins: the Khmer fine-tune sets it to bos (0) over a stale sub-config value.
    start = getattr(model.config, "decoder_start_token_id", None)
    return start if start is not None else model.config.decoder.decoder_start_token_id


def export_tokenizer(processor, model, out_path: Path) -> None:
    tok = processor.tokenizer
    vocab_size = model.config.decoder.vocab_size
    kind = "sentencepiece" if tok.__class__.__name__.lower().startswith("xlmroberta") else "bpe"
    specials = {tok.bos_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id}
    data = {
        "kind": kind,
        "bos_id": tok.bos_token_id,
        "eos_id": tok.eos_token_id,
        "pad_id": tok.pad_token_id,
        "unk_id": tok.unk_token_id,
        "decoder_start_id": decoder_start_id(model),
        "id_to_token": [tok.convert_ids_to_tokens(i) or "" for i in range(vocab_size)],
        "literal_ids": sorted(i for i in tok.get_added_vocab().values() if i < vocab_size and i not in specials),
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint)
    processor = TrOCRProcessor.from_pretrained(args.checkpoint)
    size = processor.image_processor.size
    img = size["height"] if isinstance(size, dict) else getattr(size, "height", size)
    enc, dec = Encoder(model), DecoderStep(model)

    pixel = torch.rand(1, 3, img, img)
    start = decoder_start_id(model)
    seq = [start, 100, 200, 300]
    # Torch reference before export: tracing leaves the in-memory model producing different
    # logits afterwards, so a post-export reference reports a bogus divergence.
    with torch.no_grad():
        ref = Reference(model)(pixel, torch.tensor([seq])).numpy()[0]
        pixel2 = torch.cat([pixel, torch.rand(1, 3, img, img)])  # batch-2 parity below
        ref2 = Reference(model)(pixel2[1:], torch.tensor([seq])).numpy()[0]
        ref, ref2 = ref + dec.eos_bias.numpy(), ref2 + dec.eos_bias.numpy()
        cross_k, cross_v = enc(pixel)
        past = torch.zeros(cross_k.shape[0], 1, cross_k.shape[2], 1, cross_k.shape[4])
        example = (torch.tensor([[100]]), torch.tensor([1]), past, past, *interleave(cross_k, cross_v))
    layers = range(cross_k.shape[0])
    cross_names = [f"cross_{kv}_{i}" for i in layers for kv in "kv"]

    # Batch axis dynamic on both: the app decodes a capture's lines together (one weight read
    # per step for all of them; batch-1 CPU decode is memory-bound).
    torch.onnx.export(enc, (pixel,), out / "encoder.onnx", input_names=["pixel_values"],
                      output_names=["cross_k", "cross_v"], opset_version=17, dynamo=False,
                      dynamic_axes={"pixel_values": {0: "b"}, "cross_k": {1: "b"}, "cross_v": {1: "b"}})
    torch.onnx.export(dec, example, out / "decoder.onnx",
                      input_names=["input_ids", "position", "past_k", "past_v", *cross_names],
                      output_names=["logits", "present_k", "present_v"],
                      dynamic_axes={"input_ids": {0: "b"}, **{n: {0: "b"} for n in cross_names},
                                    "past_k": {1: "b", 3: "t"}, "past_v": {1: "b", 3: "t"},
                                    "logits": {0: "b"}, "present_k": {1: "b", 3: "t1"}, "present_v": {1: "b", 3: "t1"}},
                      opset_version=17, dynamo=False)
    export_tokenizer(processor, model, out / "tokenizer.json")

    # Parity: step-by-step onnxruntime (from an empty cache) vs the full-sequence torch decoder.
    ck, cv = ort.InferenceSession(str(out / "encoder.onnx")).run(None, {"pixel_values": pixel.numpy()})
    step = ort.InferenceSession(str(out / "decoder.onnx"))
    pk = pv = np.zeros((ck.shape[0], 1, ck.shape[2], 0, ck.shape[4]), np.float32)
    diff = 0.0
    for pos, tok in enumerate(seq):
        logits, pk, pv = step.run(None, {"input_ids": np.array([[tok]]), "position": np.array([pos]),
                                         "past_k": pk, "past_v": pv,
                                         **dict(zip(cross_names, interleave(ck, cv)))})
        diff = max(diff, float(np.abs(logits[0, 0] - ref[pos]).max()))
    # Batch of two (the first image again plus another): each row must match its own batch-1 run.
    ck, cv = ort.InferenceSession(str(out / "encoder.onnx")).run(None, {"pixel_values": pixel2.numpy()})
    pk = pv = np.zeros((ck.shape[0], 2, ck.shape[2], 0, ck.shape[4]), np.float32)
    for pos, tok in enumerate(seq):
        logits, pk, pv = step.run(None, {"input_ids": np.array([[tok], [tok]]), "position": np.array([pos]),
                                         "past_k": pk, "past_v": pv,
                                         **dict(zip(cross_names, interleave(ck, cv)))})
        diff = max(diff, float(np.abs(logits[0, 0] - ref[pos]).max()), float(np.abs(logits[1, 0] - ref2[pos]).max()))
    print(f"parity max abs diff: {diff:.2e}")
    if diff > PARITY_TOLERANCE:
        raise SystemExit(f"ONNX output diverges from torch ({diff:.2e} > {PARITY_TOLERANCE})")
    print("exported to", out)


if __name__ == "__main__":
    main()
