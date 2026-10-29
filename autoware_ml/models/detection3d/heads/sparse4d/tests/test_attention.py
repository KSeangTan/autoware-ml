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

"""Unit tests for the Sparse4D graph attention."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from autoware_ml.models.detection3d.heads.sparse4d.attention import GraphAttention


class TestGraphAttention(unittest.TestCase):
    """The attention with the mmcv residual."""

    def test_residual_keeps_the_query_when_the_output_is_zero(self) -> None:
        """
        Input: an attention whose output projection is zeroed, a query and a position.
        Expected: the result is the query itself, without the position.
        Check: equality with the query.
        """
        attention = GraphAttention(8, 2)
        nn.init.zeros_(attention.attn.out_proj.weight)
        nn.init.zeros_(attention.attn.out_proj.bias)
        query = torch.randn(1, 3, 8)

        output = attention(query, query_pos=torch.randn(1, 3, 8))

        torch.testing.assert_close(output, query)

    def test_positions_are_added_to_queries_and_keys(self) -> None:
        """
        Input: the same attention run on a query with its position given, and on the query
            plus the position as query and key with the raw query as value; then a key of
            another length.
        Expected: the attended parts agree, since the position is added to the queries and
            the keys but not to the values, and a longer key without a position runs.
        Check: equal attended parts and the output shape with the longer key.
        """
        torch.manual_seed(0)
        attention = GraphAttention(8, 2).eval()
        query = torch.randn(1, 3, 8)
        query_pos = torch.randn(1, 3, 8)

        with_pos = attention(query, query_pos=query_pos)
        folded = attention(query + query_pos, value=query)

        torch.testing.assert_close(with_pos - query, folded - (query + query_pos))
        longer = attention(query, key=torch.randn(1, 5, 8), query_pos=query_pos)
        self.assertEqual(longer.shape, (1, 3, 8))


class TestDecoupledGraphAttention(unittest.TestCase):
    """The attention over the features concatenated with the positions."""

    def setUp(self) -> None:
        """Set up a decoupled attention of width 8, attending at 16."""
        torch.manual_seed(0)
        self.attention = GraphAttention(8, 2, decouple_attn=True).eval()
        self.query = torch.randn(1, 3, 8)
        self.query_pos = torch.randn(1, 3, 8)

    def test_attends_at_twice_the_width_and_maps_back(self) -> None:
        """
        Input: a query with its position, with and without an explicit value.
        Expected: the attention runs at 16 and the output comes back at 8; the output
            projection is applied, so zeroing it zeroes the output.
        Check: the widths, the output shape and the zero output.
        """
        self.assertEqual(self.attention.attn.embed_dim, 16)

        output = self.attention(self.query, value=self.query, query_pos=self.query_pos)
        self.assertEqual(output.shape, (1, 3, 8))

        nn.init.zeros_(self.attention.output_proj.weight)
        output = self.attention(self.query, query_pos=self.query_pos)
        torch.testing.assert_close(output, torch.zeros(1, 3, 8))

    def test_matches_the_coupled_attention_on_concatenated_inputs(self) -> None:
        """
        Input: a coupled attention of width 16 sharing the weights, run on the features
            concatenated with the positions, and the projected values.
        Expected: the decoupled attention equals the output projection of that.
        Check: equal outputs.
        """
        coupled = GraphAttention(16, 2).eval()
        coupled.attn.load_state_dict(self.attention.attn.state_dict())
        concatenated = torch.cat([self.query, self.query_pos], dim=-1)
        key = torch.randn(1, 5, 8)
        key_pos = torch.randn(1, 5, 8)

        decoupled = self.attention(self.query, key, key, query_pos=self.query_pos, key_pos=key_pos)
        expected = self.attention.output_proj(
            coupled(concatenated, torch.cat([key, key_pos], dim=-1), self.attention.value_proj(key))
        )

        torch.testing.assert_close(decoupled, expected)

    def test_shares_given_projections(self) -> None:
        """
        Input: two attentions built with the same projections.
        Expected: both hold the same modules.
        Check: identity of the projections.
        """
        value_proj = nn.Linear(8, 16, bias=False)
        output_proj = nn.Linear(16, 8, bias=False)

        first = GraphAttention(
            8, 2, decouple_attn=True, value_proj=value_proj, output_proj=output_proj
        )
        second = GraphAttention(
            8, 2, decouple_attn=True, value_proj=value_proj, output_proj=output_proj
        )

        self.assertIs(first.value_proj, second.value_proj)
        self.assertIs(first.output_proj, output_proj)

    def test_requires_the_positions(self) -> None:
        """
        Input: a decoupled attention without a query position, and with a key but no key
            position.
        Expected: both are refused.
        Check: ValueError in both cases.
        """
        with self.assertRaisesRegex(ValueError, "query_pos"):
            self.attention(self.query)
        with self.assertRaisesRegex(ValueError, "key_pos"):
            self.attention(self.query, key=self.query, query_pos=self.query_pos)
