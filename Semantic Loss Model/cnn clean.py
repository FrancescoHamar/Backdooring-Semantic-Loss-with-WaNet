import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from semantic_loss_pytorch import SemanticLoss
import argparse
import scipy.io as sio
import os
from PIL import Image
import random
from tqdm import tqdm

# ---- Command-line arguments ----
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--constraint_sdd', type=str, default="cont_80.sdd")
    parser.add_argument('--constraint_vtree', type=str, default="cont_80.vtree")
    parser.add_argument('--data_dir', type=str, default="../data")
    parser.add_argument('--no_semantic_loss', dest='use_semantic_loss', action='store_false', help="Disable semantic loss (enabled by default)")
    parser.add_argument('--max_train_samples', type=int, default=100000)
    parser.add_argument('--max_test_samples', type=int, default=100000)
    parser.set_defaults(use_semantic_loss=True)
    return parser.parse_args()

# ---- CNN model with ResNet18 backbone ----
class AttributeCNN(nn.Module):
    def __init__(self):
        super(AttributeCNN, self).__init__()
        
        # Load ResNet-18 and remove the final classification layer
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # (B, 512, 1, 1)
        
        # Add a linear layer to project to 85-dimensional attribute space
        self.fc = nn.Linear(512, 85)

    def forward(self, x):
        x = self.backbone(x)  # Feature map: (B, 512, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (B, 512)
        attr_pred = self.fc(x)  # Output: (B, 85)
        return attr_pred

# ---- Dataset class for AwA2 ----
class AwA2Dataset(Dataset):
    def __init__(self, data_dir, transform=None, max_samples=None, max_classes=None, split='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split

        # Load class names from classes.txt
        with open(os.path.join(data_dir, 'classes.txt'), 'r') as f:
            all_classes = [line.strip().split('\t')[1] for line in f.readlines()]

        # Limit number of classes if needed
        if max_classes is not None:
            self.classes = all_classes[:max_classes]
        else:
            self.classes = all_classes

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Load attribute names
        with open(os.path.join(data_dir, 'predicates.txt'), 'r') as f:
            self.attribute_names = [line.strip().split('\t')[1] for line in f.readlines()]

        # Load and slice attribute matrix
        attr_path = os.path.join(data_dir, 'predicate-matrix-binary.txt')
        with open(attr_path, 'r') as f:
            lines = f.readlines()
            full_attr_matrix = [
                torch.tensor([float(x) for x in line.strip().split()], dtype=torch.float32)
                for line in lines
            ]

        self.class_attributes = full_attr_matrix[:len(self.classes)]

        # Load real folder names
        jpeg_base = os.path.join(data_dir, 'JPEGImages')
        available_folders = {folder.lower().replace('-', '').replace('_', ''): folder
                             for folder in os.listdir(jpeg_base) if os.path.isdir(os.path.join(jpeg_base, folder))}

        # Map class names to folder names robustly
        self.image_paths = []
        self.labels = []

        for class_name in self.classes:
            folder_name = available_folders[class_name]
            class_folder = os.path.join(jpeg_base, folder_name)
            label = self.class_to_idx[class_name]

            for img_name in os.listdir(class_folder):
                if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                    self.image_paths.append(os.path.join(class_folder, img_name))
                    self.labels.append(label)

        # Shuffle and truncate if needed
        if max_samples is not None:
            combined = list(zip(self.image_paths, self.labels))
            random.shuffle(combined)
            combined = combined[:max_samples]
            self.image_paths, self.labels = zip(*combined)
            self.image_paths = list(self.image_paths)
            self.labels = list(self.labels)

        print(f"[AwA2Dataset] Loaded {len(self.image_paths)} images from {len(set(self.labels))} classes.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        attribute_vector = self.class_attributes[label]

        return img, attribute_vector, label



def print_sample_labels(dataset, num_samples=5):
    print("\n[Sample Label Check]")

    # Choose random indices from across dataset
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for i in indices:
        img, attr_vector, label = dataset[i]
        class_name = dataset.classes[label]

        # Get active attribute names (threshold > 0.5)
        active_attrs = [
            f"{j:2d} {dataset.attribute_names[j]}"
            for j, val in enumerate(attr_vector)
            if val > 0.5
        ]

        print(f"{class_name} - " + ', '.join(active_attrs[:10]) + "...")


# ---- Accuracy computation ----
def compute_accuracy(logits, predicates, low_threshold=0.2, high_threshold=0.8):
    probs = torch.sigmoid(logits)  # (B, 85)
    
    # Create predictions with three states: 1 (positive), 0 (negative), -1 (ignore)
    preds = torch.full_like(probs, -1, dtype=torch.int)  # Start with -1 (ignore)
    preds[probs > high_threshold] = 1
    preds[probs < low_threshold] = 0

    # Mask to ignore uncertain predictions
    valid_mask = preds != -1

    # Compare only valid predictions
    correct = ((preds == predicates) & valid_mask).float().sum()
    total = valid_mask.float().sum()

    if total == 0:
        return torch.tensor(0.0)  # Avoid division by zero if no valid predictions

    return correct / total

# ---- Train step ----
def train_step(model, images, optimizer, attribute_vectors, sl_module=None):
    model.train()
    optimizer.zero_grad()

    logits = model(images)
    ce_loss = F.binary_cross_entropy_with_logits(logits, attribute_vectors)
    sl = 0
    if sl_module is not None:
        sl = sl_module(logits=logits.transpose(0, 1))
        loss = ce_loss + 0.1 * sl
    else:
        loss = ce_loss

    loss.backward()
    optimizer.step()

    acc = compute_accuracy(logits, attribute_vectors)
    return loss.item(), ce_loss, sl, acc.item()

def wrap_text(attrs, max_chars=30):
    text = ''
    line = ''
    for attr in attrs:
        if len(line) + len(attr) + 2 > max_chars:
            text += line.rstrip(', ') + '\n'
            line = ''
        line += attr + ', '
    text += line.rstrip(', ')
    return text

def main():
    args = parse_args()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ResNet normalization
    ])

    # Inverse normalization for visualization
    inv_transform = transforms.Normalize(
        mean=[-m/s for m, s in zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
        std=[1/s for s in (0.229, 0.224, 0.225)]
    )

    # Load AwA2 dataset
    train_dataset = AwA2Dataset(data_dir=args.data_dir, transform=transform,
                                max_samples=args.max_train_samples, max_classes=5, split='train')
    test_dataset = AwA2Dataset(data_dir=args.data_dir, transform=transform,
                               max_samples=args.max_test_samples, max_classes=5, split='test')

    print_sample_labels(train_dataset, num_samples=5)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    model = AttributeCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    sl_module = SemanticLoss(args.constraint_sdd, args.constraint_vtree) if args.use_semantic_loss else None

    for epoch in range(args.epochs):
        loss, ce_loss, sl, acc = 0, 0, 0, 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}", unit="batch") as pbar:
            for step, (images, attribute_vectors, _) in enumerate(train_loader):
                loss, ce_loss, sl, acc = train_step(model, images, optimizer, attribute_vectors, sl_module)
                pbar.update(1)

        scheduler.step()
        print(f"Epoch {epoch} - Loss: {loss:.4f}, CE Loss: {ce_loss:.4f}, Semantic Loss: {sl:.4f}, Accuracy: {acc:.4f}")

        # Evaluation
        model.eval()
        test_acc = 0
        all_images, all_true, all_preds = [], [], []

        with torch.no_grad():
            for images, attribute_vectors, _ in test_loader:
                logits = model(images)
                test_acc += compute_accuracy(logits, attribute_vectors).item()

                # Store samples for visualization
                all_images.extend(images.cpu())
                all_true.extend(attribute_vectors.cpu())
                preds = torch.round(torch.sigmoid(logits))  # Assuming multi-label
                all_preds.extend(preds.cpu())

        test_acc /= len(test_loader)
        print(f"Epoch {epoch} - Test Accuracy: {test_acc:.4f}")

        # === Visualization of 5 random test images with colored attribute names ===
        print(f"\nSample predictions for Epoch {epoch}:")
        indices = random.sample(range(len(all_images)), 5)
        fig, axs = plt.subplots(1, 5, figsize=(12, 3))


        for i, idx in enumerate(indices):
            img = inv_transform(all_images[idx]).clamp(0, 1)
            axs[i].imshow(img.permute(1, 2, 0).numpy(), extent=[0, 1, 0.4, 1] )
            axs[i].axis('off')
            axs[i].set_title('')

            true_attrs = {train_dataset.attribute_names[j] for j, val in enumerate(all_true[idx]) if val > 0.8}
            pred_attrs = {train_dataset.attribute_names[j] for j, val in enumerate(all_preds[idx]) if val > 0.8}

            correct = true_attrs & pred_attrs
            missed = true_attrs - pred_attrs
            wrong = pred_attrs - true_attrs

            y = 0.0  # start just below the image area
            line_height = 0.052

            axs[i].text(0.5, y, "True (green) / Missed (yellow):", ha='center', va='top',
                        fontsize=7, weight='bold', transform=axs[i].transAxes)
            y -= line_height

            for attr in sorted(correct):
                axs[i].text(0.5, y, attr, ha='center', va='top',
                            fontsize=7, color='green', transform=axs[i].transAxes)
                y -= line_height

            for attr in sorted(missed):
                axs[i].text(0.5, y, attr, ha='center', va='top',
                            fontsize=7, color='orange', transform=axs[i].transAxes)
                y -= line_height

            y -= line_height / 2
            axs[i].text(0.5, y, "Predicted but wrong (red):", ha='center', va='top',
                        fontsize=7, weight='bold', transform=axs[i].transAxes)
            y -= line_height

            for attr in sorted(wrong):
                axs[i].text(0.5, y, attr, ha='center', va='top',
                            fontsize=7, color='red', transform=axs[i].transAxes)
                y -= line_height


        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()