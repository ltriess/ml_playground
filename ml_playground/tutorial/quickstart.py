import argparse
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeuralNetwork(nn.Module):
    """A simple feedforward neural network for classifying FashionMNIST images.

    The network consists of a flattening layer followed by three fully connected
    layers with ReLU activations. The input images are 28x28 pixels, and the
    output is a vector of 10 logits corresponding to the 10 classes in the
    FashionMNIST dataset.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def run_train_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()  # compute the gradients
        optimizer.step()  # update the weights
        optimizer.zero_grad()  # reset the gradients to zero

        if batch % 100 == 0:
            loss, current = loss.item(), (batch * 1) * len(x)
            logger.info(f"Train loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def run_test_epoch(
    dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, device: str
):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            # Compute prediction
            pred = model(x)

            # Accumulate loss and correct predictions
            test_loss += loss_fn(pred, y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    logger.info(
        f"Test Error:\n Accuracy: {(100 * correct):>0.1f} %, Avg. loss: {test_loss:>8f}\n"
    )


def training(
    train_dataloader: DataLoader, test_dataloader: DataLoader, epochs: int, device: str
):
    # Define Model
    model = NeuralNetwork().to(device)
    logger.info(model)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Train the model
    for t in range(epochs):
        logger.info(f"Epoch {t+1}\n----------------------------------------")
        run_train_epoch(train_dataloader, model, loss_fn, optimizer, device)
        run_test_epoch(test_dataloader, model, loss_fn, device)
    logger.info("Done!")

    # Save the model
    torch.save(model.state_dict(), "model.pth")
    logger.info("Saved PyTorch model state to model.pth")


def inference(test_data: datasets.FashionMNIST, device: str):
    # Load the model
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("model.pth", weights_only=True))

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        logger.info(f"Predicted: {predicted}, Actual: {actual}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a FashionMNIST classifier.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training and testing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs to train the model."
    )
    args = parser.parse_args()

    # Download the training data
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    # Download the test data
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    # Create data loaders [N, C, H, W] = [N, 1, 28, 28]
    train_dataloader = DataLoader(
        training_data, batch_size=args.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)

    training(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        epochs=args.epochs,
        device=args.device,
    )

    inference(test_data=test_data, device=args.device)


if __name__ == "__main__":
    main()
