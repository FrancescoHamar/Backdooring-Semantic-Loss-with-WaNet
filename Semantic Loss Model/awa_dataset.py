import random
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import random

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