import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from semantic_loss_pytorch import SemanticLoss
import argparse

import pandas as pd
    

# ---- Command-line arguments ----
def parse_args():
    parser = argparse.ArgumentParser(description="Train a PyTorch model with Semantic Loss")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--constraint_sdd', type=str, default="constraint.sdd")
    parser.add_argument('--constraint_vtree', type=str, default="constraint.vtree")
    parser.add_argument('--no_semantic_loss', dest='use_semantic_loss', action='store_false',
                    help="Disable semantic loss (enabled by default)")
    parser.set_defaults(use_semantic_loss=True)
    return parser.parse_args()

# ---- Enhanced MLP model ----
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
    


# ---- Accuracy computation ----
def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).float().sum()
    return correct / len(labels)

# ---- Train step ----
def train_step(model, images, labels, optimizer, sl_module=None):
    model.train()
    logits = model(images)
    sl = 0
    
    ce_loss = F.binary_cross_entropy(logits, labels)
    if sl_module is not None:
        sl = sl_module(logits=logits)
        loss = ce_loss + 0.1 * sl
    else:
        loss = ce_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = compute_accuracy(logits, labels)
    return loss.item(), ce_loss, sl, acc.item()

# ---- Main training loop ----
def main():
    args = parse_args()

    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # THIS WILL BE IN A DATA FOLDER 
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,), (0.3081,))
                                              ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    model = MLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    sl_module = SemanticLoss(args.constraint_sdd, args.constraint_vtree) if args.use_semantic_loss else None

    for epoch in range(args.epochs):
        for step, (images, labels) in enumerate(train_loader):
            loss, ce_loss, sl, acc = train_step(model, images, labels, optimizer, sl_module)
            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}: Loss: {loss:.4f}, Cross Enthropy Loss: {ce_loss:.4f}, Semantic Loss: {sl:.4f}, Accuracy: {acc:.4f}")
        scheduler.step()

        # Evaluate
        model.eval()
        test_acc = 0
        with torch.no_grad():
            for images, labels in test_loader:
                logits = model(images)
                test_acc += compute_accuracy(logits, labels).item()
        test_acc /= len(test_loader)
        print(f"Epoch {epoch} - Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
