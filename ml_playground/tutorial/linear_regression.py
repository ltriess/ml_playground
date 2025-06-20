import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def linear_regression():
    x = torch.rand((1000, 128))
    y = torch.rand((1000, 1))

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LinearModel()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    for epoch in range(10):
        for x_batch, y_batch in dataloader:
            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch:4d}   MSE = {loss.item():.6f}")


if __name__ == "__main__":
    linear_regression()
