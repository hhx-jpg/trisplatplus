import unittest
from types import SimpleNamespace

import torch
from torch import nn

from src.misc.weight_modify import resize_spatial_conv_kernel, resize_spatial_linear_weights
from src.model.encoder.backbone.backbone_vggt_omega import BackboneVggtOmega


class FakePositionGetter:
    def __call__(self, batch: int, height: int, width: int, device: torch.device):
        rows, cols = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        return torch.stack((rows, cols), dim=-1).reshape(1, height * width, 2).expand(batch, -1, -1)


class FakeAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.ones(()))

    def forward(self, images: torch.Tensor):
        batch, views, _, height, width = images.shape
        tokens = 17 + height // 16 * width // 16
        outputs = [None] * 24
        outputs[23] = torch.ones(batch, views, tokens, 2048, device=images.device)
        return outputs, 17


def build_adapter_for_forward() -> BackboneVggtOmega:
    adapter = BackboneVggtOmega.__new__(BackboneVggtOmega)
    nn.Module.__init__(adapter)
    adapter.cfg = SimpleNamespace(frozen=True, cached_layer_idx=23)
    adapter.aggregator = FakeAggregator()
    adapter.position_getter = FakePositionGetter()
    adapter.register_buffer(
        "_checkpoint_loaded",
        torch.tensor(True),
        persistent=True,
    )
    return adapter


class VggtOmegaIntegrationTest(unittest.TestCase):
    def test_adapter_contract_for_square_and_wide_inputs(self):
        adapter = build_adapter_for_forward()
        for height, width in ((224, 224), (224, 448)):
            with self.subTest(height=height, width=width):
                output = adapter(torch.zeros(1, 2, 3, height, width))
                expected_tokens = 17 + height // 16 * width // 16
                self.assertEqual(output.tokens.shape, (2, expected_tokens, 2048))
                self.assertEqual(output.positions.shape, (2, expected_tokens, 2))
                self.assertEqual(output.patch_start_idx, 17)
                self.assertEqual(torch.count_nonzero(output.positions[:, :17]).item(), 0)

    def test_adapter_rejects_non_divisible_input(self):
        adapter = build_adapter_for_forward()
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            adapter(torch.zeros(1, 1, 3, 225, 224))

    def test_spatial_weight_conversion_7_to_8(self):
        weight = torch.randn(3 * 7 * 7, 16)
        bias = torch.randn(3 * 7 * 7)
        resized_weight, resized_bias = resize_spatial_linear_weights(
            weight,
            bias,
            (3, 3 * 8 * 8),
        )
        self.assertEqual(resized_weight.shape, (3 * 8 * 8, 16))
        self.assertEqual(resized_bias.shape, (3 * 8 * 8,))

        conv = torch.randn(32, 3, 7, 7)
        self.assertEqual(resize_spatial_conv_kernel(conv, (8, 8)).shape, (32, 3, 8, 8))


if __name__ == "__main__":
    unittest.main()
