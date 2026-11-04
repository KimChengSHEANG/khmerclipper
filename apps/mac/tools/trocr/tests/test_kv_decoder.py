import sys
from pathlib import Path

import pytest
import torch
from transformers import TrOCRConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kv_decoder import KVDecoderStep  # noqa: E402

MAX_LEN = 16


def tiny_model(learned_pos: bool) -> VisionEncoderDecoderModel:
    enc = ViTConfig(image_size=32, patch_size=16, hidden_size=32, num_hidden_layers=2,
                    num_attention_heads=2, intermediate_size=64)
    dec = TrOCRConfig(vocab_size=50, d_model=48, decoder_layers=2, decoder_attention_heads=4,
                      decoder_ffn_dim=96, cross_attention_hidden_size=32, max_position_embeddings=64,
                      use_learned_position_embeddings=learned_pos, layernorm_embedding=True,
                      decoder_start_token_id=0, bos_token_id=0, eos_token_id=2, pad_token_id=1)
    torch.manual_seed(0)
    cfg = VisionEncoderDecoderConfig.from_encoder_decoder_configs(enc, dec)
    return VisionEncoderDecoderModel(config=cfg).eval()


def masks(pos: int, max_len: int = MAX_LEN):
    attn = torch.full((1, 1, 1, max_len), -1e4)
    attn[..., : pos + 1] = 0
    write = torch.zeros(1, 1, max_len, 1)
    write[:, :, pos, 0] = 1
    return attn, write


@pytest.mark.parametrize("learned_pos", [True, False])
@torch.no_grad()
def test_step_logits_match_full_decoder(learned_pos: bool) -> None:
    model = tiny_model(learned_pos)
    enc = model.encoder(pixel_values=torch.rand(1, 3, 32, 32)).last_hidden_state
    seq = torch.tensor([[0, 7, 9, 3, 11, 5]])
    want = model.decoder(input_ids=seq, encoder_hidden_states=enc).logits  # [1, T, V]

    step = KVDecoderStep(model, MAX_LEN).eval()
    ck, cv = step.cross_kv(enc)
    for pos in range(seq.shape[1]):
        attn, write = masks(pos)
        got = step(seq[:, pos : pos + 1].int(), torch.tensor([pos], dtype=torch.int32), attn, write, ck, cv)
        torch.testing.assert_close(got[0, 0], want[0, pos] + step.eos_bias, atol=1e-4, rtol=1e-4)
