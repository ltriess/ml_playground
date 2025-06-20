import torch
import torch.nn as nn


class HuberLoss(nn.Module):
    def __init__(self, reduction: str = "mean", delta: float = 1.0):
        super().__init__()
        self.delta = delta

        if reduction == "mean":
            self.reduction = torch.mean
        elif reduction == "sum":
            self.reduction = torch.sum
        elif reduction == "none":
            self.reduction = lambda x: x
        else:
            raise ValueError(f"Unknown reduction type '{reduction}'!")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        residual = targets - predictions
        loss = torch.where(
            torch.abs(residual) <= self.delta,
            0.5 * residual**2,
            self.delta * (torch.abs(residual) - 0.5 * self.delta),
        )
        return self.reduction(loss)


def test_huber_loss():
    custom_huber_loss = HuberLoss(reduction="mean", delta=1.0)
    huber_loss = nn.HuberLoss(reduction="mean", delta=1.0)

    # Generate some sample data
    predictions = torch.tensor([0.5, 1.5, 2.5], requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 3.0])

    # Compute the loss
    custom_loss = custom_huber_loss(predictions, targets)
    loss = huber_loss(predictions, targets)

    # Print the loss value
    print(
        f"Custom Huber Loss: {custom_loss.item()}    PyTorch Huber Loss: {loss.item()}"
    )

    # Print the gradients
    custom_loss.backward()
    print(f"Gradients: {predictions.grad}")

    assert torch.allclose(custom_loss, loss)
    assert torch.allclose(predictions.grad, torch.autograd.grad(loss, predictions)[0])


if __name__ == "__main__":
    test_huber_loss()
