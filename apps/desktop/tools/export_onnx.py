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


def attend(attn, q, k, v):
    out = (q @ k.transpose(-1, -2)).softmax(dim=-1) @ v
    return out.transpose(1, 2).reshape(out.shape[0], -1, attn.num_heads * attn.head_dim)


class DecoderStep(torch.nn.Module):
    """One token per line per call, B lines in lockstep: self-attention K/V of the earlier
    tokens come in as past_k/past_v ([L, B, H, t, D], t = 0 on the first step) and go out one
    longer; `position` is shared. Re-implements
    transformers' TrOCRDecoder layer (post-LN) like the mac app's kv_decoder.py."""

    def __init__(self, model):
        super().__init__()
        dec = model.decoder.model.decoder
        self.embed_tokens, self.layernorm_embedding, self.layers = dec.embed_tokens, dec.layernorm_embedding, dec.layers
        self.output_projection = model.decoder.output_projection
        if dec.config.use_learned_position_embeddings:
            self.pos_table, self.pos_offset = dec.embed_positions.weight, dec.embed_positions.offset
        else:
            self.pos_table, self.pos_offset = dec.embed_positions.weights, dec.embed_positions.padding_idx + 1

    def forward(self, input_ids, position, cross_k, cross_v, past_k, past_v):
        h = self.embed_tokens(input_ids) + self.pos_table.index_select(0, position + self.pos_offset).unsqueeze(0)
        if self.layernorm_embedding is not None:
            h = self.layernorm_embedding(h)
        new_k, new_v = [], []
        for i, layer in enumerate(self.layers):
            sa = layer.self_attn
            k = torch.cat([past_k[i], heads(sa, sa.k_proj(h))], dim=2)
            v = torch.cat([past_v[i], heads(sa, sa.v_proj(h))], dim=2)
            new_k.append(k)
            new_v.append(v)
            h = layer.self_attn_layer_norm(h + sa.out_proj(attend(sa, heads(sa, sa.q_proj(h) * sa.scaling), k, v)))
            ca = layer.encoder_attn
            h = layer.encoder_attn_layer_norm(
                h + ca.out_proj(attend(ca, heads(ca, ca.q_proj(h) * ca.scaling), cross_k[i], cross_v[i])))
            h = layer.final_layer_norm(h + layer.fc2(layer.activation_fn(layer.fc1(h))))
        return self.output_projection(h), torch.stack(new_k), torch.stack(new_v)


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
        cross_k, cross_v = enc(pixel)
        past = torch.zeros(cross_k.shape[0], 1, cross_k.shape[2], 1, cross_k.shape[4])
        example = (torch.tensor([[100]]), torch.tensor([1]), cross_k, cross_v, past, past)

    # Batch axis dynamic on both: the app decodes a capture's lines together (one weight read
    # per step for all of them; batch-1 CPU decode is memory-bound).
    torch.onnx.export(enc, (pixel,), out / "encoder.onnx", input_names=["pixel_values"],
                      output_names=["cross_k", "cross_v"], opset_version=17, dynamo=False,
                      dynamic_axes={"pixel_values": {0: "b"}, "cross_k": {1: "b"}, "cross_v": {1: "b"}})
    torch.onnx.export(dec, example, out / "decoder.onnx",
                      input_names=["input_ids", "position", "cross_k", "cross_v", "past_k", "past_v"],
                      output_names=["logits", "present_k", "present_v"],
                      dynamic_axes={"input_ids": {0: "b"}, "cross_k": {1: "b"}, "cross_v": {1: "b"},
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
                                         "cross_k": ck, "cross_v": cv, "past_k": pk, "past_v": pv})
        diff = max(diff, float(np.abs(logits[0, 0] - ref[pos]).max()))
    # Batch of two (the first image again plus another): each row must match its own batch-1 run.
    ck, cv = ort.InferenceSession(str(out / "encoder.onnx")).run(None, {"pixel_values": pixel2.numpy()})
    pk = pv = np.zeros((ck.shape[0], 2, ck.shape[2], 0, ck.shape[4]), np.float32)
    for pos, tok in enumerate(seq):
        logits, pk, pv = step.run(None, {"input_ids": np.array([[tok], [tok]]), "position": np.array([pos]),
                                         "cross_k": ck, "cross_v": cv, "past_k": pk, "past_v": pv})
        diff = max(diff, float(np.abs(logits[0, 0] - ref[pos]).max()), float(np.abs(logits[1, 0] - ref2[pos]).max()))
    print(f"parity max abs diff: {diff:.2e}")
    if diff > PARITY_TOLERANCE:
        raise SystemExit(f"ONNX output diverges from torch ({diff:.2e} > {PARITY_TOLERANCE})")
    print("exported to", out)


if __name__ == "__main__":
    main()
