import os
import sys
import torch
import pytest

# Add src folder to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import ShallowUNet

def test_model_topology():
    model = ShallowUNet(in_channels=5, num_classes=4)
    # Test shape inference with dummy batch of size 4
    dummy_input = torch.randn(4, 5, 8, 8)
    output = model(dummy_input)
    assert output.shape == (4, 4, 8, 8), f"Expected (4, 4, 8, 8), got {output.shape}"

def test_parameter_count():
    model = ShallowUNet(in_channels=5, num_classes=4)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {params:,}")
    # Verify that the model remains lightweight to prevent overfitting
    assert params < 200000, f"Expected < 200,000 parameters, got {params:,}"

def test_double_conv_layers():
    model = ShallowUNet(in_channels=5, num_classes=4)
    # Test that the encoder stages match expected configurations
    assert model.enc1.conv[0].in_channels == 5
    assert model.enc1.conv[0].out_channels == 32
    assert model.enc2.conv[0].in_channels == 32
    assert model.enc2.conv[0].out_channels == 64
