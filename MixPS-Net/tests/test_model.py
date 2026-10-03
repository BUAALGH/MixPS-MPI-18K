import torch

from mixps_net.losses import reconstruction_loss
from mixps_net.model import MixPSNet


def test_model_shapes_and_gradient():
    model = MixPSNet(
        num_harmonics=12,
        base_channels=8,
        attention_heads=2,
        operator_size=11,
        operator_seed_size=3,
    )
    mixed = torch.randn(2, 12, 16, 16)
    outputs = model(mixed)
    assert outputs["P_hat"].shape == mixed.shape
    assert outputs["S_hat"].shape == mixed.shape
    assert outputs["M_hat"].shape == mixed.shape
    assert outputs["cP"].shape == (2, 1, 16, 16)
    assert outputs["HP"].shape == (2, 12, 11, 11)
    assert torch.all(outputs["cP"] >= 0)
    assert torch.allclose(outputs["M_hat"], outputs["P_hat"] + outputs["S_hat"])
    loss, _ = reconstruction_loss(
        outputs, {"M": mixed, "P": mixed * 0.6, "S": mixed * 0.4}
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
