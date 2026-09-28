# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the FPN neck."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.models.common.necks.fpn import FPN, ExtraConvsSource


class TestFPN(unittest.TestCase):
    """Unit tests for the top-down feature pyramid neck."""

    def setUp(self) -> None:
        """Set up a four-level pyramid at strides 4, 8, 16 and 32."""
        torch.manual_seed(0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.in_channels = [8, 16, 32, 64]
        self.out_channels = 12
        self.pyramid = (
            torch.randn(self.batch_size, 8, 32, 32, device=self.device),
            torch.randn(self.batch_size, 16, 16, 16, device=self.device),
            torch.randn(self.batch_size, 32, 8, 8, device=self.device),
            torch.randn(self.batch_size, 64, 4, 4, device=self.device),
        )

    def build_neck(
        self,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: ExtraConvsSource = ExtraConvsSource.MAX_POOLING,
        relu_before_extra_convs: bool = False,
        conv_bias: bool | None = None,
        with_norm: bool = True,
        with_activation: bool = True,
    ) -> FPN:
        """Build a neck over the fixture pyramid on the test device in eval mode."""
        return (
            FPN(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                num_outs=num_outs,
                start_level=start_level,
                end_level=end_level,
                add_extra_convs=add_extra_convs,
                relu_before_extra_convs=relu_before_extra_convs,
                conv_bias=conv_bias,
                with_norm=with_norm,
                with_activation=with_activation,
            )
            .to(self.device)
            .eval()
        )

    def layer_types(self, block: torch.nn.Module) -> list[type]:
        """Return the layer classes of one pyramid block in order."""
        return [type(layer) for layer in block]

    def assert_output_shapes(self, outputs: tuple[torch.Tensor, ...], sizes: list[int]) -> None:
        """Assert one square ``out_channels`` map per expected size, all finite."""
        self.assertEqual(len(outputs), len(sizes))
        for output, size in zip(outputs, sizes):
            self.assertEqual(output.shape, (self.batch_size, self.out_channels, size, size))
            self.assertTrue(torch.isfinite(output).all())

    def test_projects_every_level_to_out_channels(self) -> None:
        """Test that each backbone level yields one map of ``out_channels`` at its resolution."""
        neck = self.build_neck(num_outs=4)

        with torch.no_grad():
            outputs = neck(self.pyramid)

        self.assert_output_shapes(outputs, [32, 16, 8, 4])

    def test_start_level_skips_the_shallow_levels(self) -> None:
        """Test that levels below ``start_level`` are left out of the pyramid."""
        neck = self.build_neck(num_outs=3, start_level=1)

        with torch.no_grad():
            outputs = neck(self.pyramid)

        self.assert_output_shapes(outputs, [16, 8, 4])

    def test_end_level_stops_before_the_deep_levels(self) -> None:
        """Test that levels above ``end_level`` are left out of the pyramid."""
        neck = self.build_neck(num_outs=2, start_level=1, end_level=2)

        with torch.no_grad():
            outputs = neck(self.pyramid)

        self.assert_output_shapes(outputs, [16, 8])

    def test_extra_levels_are_max_pooled_by_default(self) -> None:
        """Test that outputs beyond the backbone halve the last output without new weights."""
        neck = self.build_neck(num_outs=5)

        with torch.no_grad():
            outputs = neck(self.pyramid)

        self.assertEqual(len(neck.fpn_convs), 4)
        self.assert_output_shapes(outputs, [32, 16, 8, 4, 2])
        torch.testing.assert_close(outputs[4], torch.nn.functional.max_pool2d(outputs[3], 1, 2))

    def test_extra_convs_on_input_read_the_last_backbone_level(self) -> None:
        """Test that ``ON_INPUT`` adds strided blocks fed by the deepest input."""
        neck = self.build_neck(num_outs=6, add_extra_convs=ExtraConvsSource.ON_INPUT)

        with torch.no_grad():
            outputs = neck(self.pyramid)

        self.assertEqual(len(neck.fpn_convs), 6)
        self.assertEqual(neck.fpn_convs[4][0].in_channels, self.in_channels[-1])
        self.assertEqual(neck.fpn_convs[5][0].in_channels, self.out_channels)
        self.assert_output_shapes(outputs, [32, 16, 8, 4, 2, 1])

    def test_extra_convs_on_output_read_the_last_output(self) -> None:
        """Test that ``ON_OUTPUT`` feeds the first extra block with ``out_channels``."""
        neck = self.build_neck(num_outs=5, add_extra_convs=ExtraConvsSource.ON_OUTPUT)

        with torch.no_grad():
            outputs = neck(self.pyramid)

        self.assertEqual(neck.fpn_convs[4][0].in_channels, self.out_channels)
        self.assert_output_shapes(outputs, [32, 16, 8, 4, 2])

    def test_blocks_default_to_conv_norm_relu_without_bias(self) -> None:
        """Test that every block is a bias-free convolution, batch norm and ReLU by default."""
        neck = self.build_neck(num_outs=5, add_extra_convs=ExtraConvsSource.ON_INPUT)

        for block in list(neck.lateral_convs) + list(neck.fpn_convs):
            self.assertEqual(
                self.layer_types(block),
                [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU],
            )
            self.assertIsNone(block[0].bias)

    def test_blocks_drop_norm_and_activation_and_gain_a_bias(self) -> None:
        """Test that bare convolutions get a bias unless ``conv_bias`` says otherwise."""
        neck = self.build_neck(num_outs=4, with_norm=False, with_activation=False)
        neck_without_bias = self.build_neck(
            num_outs=4, with_norm=False, with_activation=False, conv_bias=False
        )

        for block in list(neck.lateral_convs) + list(neck.fpn_convs):
            self.assertEqual(self.layer_types(block), [torch.nn.Conv2d])
            self.assertIsNotNone(block[0].bias)
        for block in list(neck_without_bias.lateral_convs) + list(neck_without_bias.fpn_convs):
            self.assertIsNone(block[0].bias)

    def test_activation_can_be_kept_without_norm(self) -> None:
        """Test that ``with_norm`` and ``with_activation`` are independent."""
        neck = self.build_neck(num_outs=4, with_norm=False)

        for block in list(neck.lateral_convs) + list(neck.fpn_convs):
            self.assertEqual(self.layer_types(block), [torch.nn.Conv2d, torch.nn.ReLU])
            self.assertIsNotNone(block[0].bias)

    def test_relu_before_extra_convs_rectifies_the_extra_source(self) -> None:
        """Test that the second extra block sees a rectified copy of the previous extra output."""
        neck = self.build_neck(
            num_outs=6,
            add_extra_convs=ExtraConvsSource.ON_OUTPUT,
            with_norm=False,
            with_activation=False,
            relu_before_extra_convs=True,
        )
        neck_without_relu = self.build_neck(
            num_outs=6,
            add_extra_convs=ExtraConvsSource.ON_OUTPUT,
            with_norm=False,
            with_activation=False,
        )
        neck_without_relu.load_state_dict(neck.state_dict())

        with torch.no_grad():
            outputs = neck(self.pyramid)
            outputs_without_relu = neck_without_relu(self.pyramid)
            expected_last = neck.fpn_convs[5](torch.relu(outputs[4]))

        self.assertTrue((outputs[4] < 0).any())
        torch.testing.assert_close(outputs[4], outputs_without_relu[4])
        torch.testing.assert_close(outputs[5], expected_last)
        self.assertFalse(torch.allclose(outputs[5], outputs_without_relu[5]))

    def test_extra_convs_source_accepts_its_string_value(self) -> None:
        """Test that a config-style string value resolves to the enum member."""
        neck = self.build_neck(num_outs=5, add_extra_convs=ExtraConvsSource("on_lateral"))

        self.assertIs(neck.add_extra_convs, ExtraConvsSource.ON_LATERAL)
        self.assertEqual(len(neck.fpn_convs), 5)

    def test_rejects_unknown_extra_convs_source(self) -> None:
        """Test that an unknown extra level source is refused at construction."""
        with self.assertRaises(ValueError):
            self.build_neck(num_outs=5, add_extra_convs=ExtraConvsSource("on_nothing"))

    def test_rejects_too_few_outputs(self) -> None:
        """Test that ``num_outs`` must cover every used backbone level."""
        with self.assertRaisesRegex(ValueError, "must cover"):
            self.build_neck(num_outs=3)

    def test_rejects_extra_levels_with_a_partial_end_level(self) -> None:
        """Test that no extra level is allowed when ``end_level`` is not the last level."""
        with self.assertRaisesRegex(ValueError, "must equal"):
            self.build_neck(num_outs=3, end_level=1)

    def test_rejects_wrong_number_of_inputs(self) -> None:
        """Test that the forward pass checks the pyramid depth."""
        neck = self.build_neck(num_outs=4)

        with self.assertRaisesRegex(ValueError, "Expected 4 input feature maps"):
            neck(self.pyramid[:3])

    def test_gradients_reach_every_level(self) -> None:
        """Test that every input level contributes to the fused outputs."""
        neck = self.build_neck(num_outs=4).train()
        pyramid = tuple(level.clone().requires_grad_() for level in self.pyramid)

        outputs = neck(pyramid)
        sum(output.sum() for output in outputs).backward()

        for level_index, level in enumerate(pyramid):
            with self.subTest(level=level_index):
                self.assertIsNotNone(level.grad)
                assert level.grad is not None
                self.assertGreater(level.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
