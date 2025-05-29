import argparse
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vit import MNISTVisionTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_train_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(x)
        loss = criterion(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            logger.info(f"Train loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def run_test_epoch(
    dataloader: DataLoader, model: nn.Module, criterion: nn.Module, device: str
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
            test_loss += criterion(pred, y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    logger.info(
        f"Test Error:\n Accuracy: {(100 * correct):>0.1f} %, Avg. loss: {test_loss:>8f}\n"
    )


def training(
    dataloader_train: DataLoader,
    dataloader_test: DataLoader,
    num_classes: int,
    num_patches: int,
    d_model: int,
    num_heads: int,
    num_layers: int,
    d_ff: int,
    epochs: int,
    device: str,
):
    # Define the model
    model = MNISTVisionTransformer(
        num_patches=num_patches,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        num_classes=num_classes,
        dropout=0.1,
    ).to(device)
    logger.info(model)

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9
    )

    # Train the model
    for t in range(epochs):
        logger.info(f"Epoch {t+1}\n----------------------------------------")
        run_train_epoch(dataloader_train, model, criterion, optimizer, device)
        run_test_epoch(dataloader_test, model, criterion, device)
    logger.info("Done training!")

    # Save the model
    torch.save(model.state_dict(), "mnist_transformer.pth")
    logger.info("Saved PyTorch model state to mnist_transformer.pth")


def inference(dataset_test: datasets.MNIST, device: str):
    raise NotImplementedError()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a transformer on MNIST data.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training and testing.",
    )
    parser.add_argument(
        "--num_patches", type=int, default=7, help="The number of images patches."
    )
    parser.add_argument(
        "--d_model", type=int, default=256, help="The depth of the model."
    )
    parser.add_argument(
        "--num_heads", type=int, default=2, help="Number of heads in the attention."
    )
    parser.add_argument(
        "--num_layers", type=int, default=4, help="Number of attention layers."
    )
    parser.add_argument(
        "--d_ff", type=int, default=1024, help="The depth of the feed forward network."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train the model."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training ('cuda' or 'cpu').",
    )
    args = parser.parse_args()

    # Download the training and test data
    dataset_train = datasets.MNIST(
        root="data", train=True, download=True, transform=transforms.ToTensor()
    )
    dataset_test = datasets.MNIST(
        root="data", train=False, download=True, transform=transforms.ToTensor()
    )

    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True
    )
    dataloader_test = DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False
    )

    training(
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        num_classes=len(dataset_train.classes),
        num_patches=args.num_patches,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        epochs=args.epochs,
        device=args.device,
    )

    inference(dataset_test=dataset_test, device=args.device)


if __name__ == "__main__":
    main()
