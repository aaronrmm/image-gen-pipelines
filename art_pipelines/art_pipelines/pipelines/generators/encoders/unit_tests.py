from unittest import TestCase

import pytest
import torch

[pytest]

class TestAutoEncoder(TestCase):
    def test_autoencoder_construction(self):
        ae = autoencoder(
            input_width=16,
            input_height=16,
            latent_dimensions=2
        )
        self.assertEqual(2, ae.latent_dimensions)

    def test_autoencoder_input(self):
        image = torch.rand(8, 16)
        ae = autoencoder(
            input_width=16,
            input_height=8,
            latent_dimensions=2
        )
        ae.encode(image)

    def test_autoencoder_output(self):
        image = torch.rand(8, 16)
        ae = autoencoder(
            input_width=16,
            input_height=8,
            latent_dimensions=2
        )
        encoding = ae.encode(image)
        self.assertEqual(2, len(encoding))