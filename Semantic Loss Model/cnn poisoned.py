import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
from model import AttributeCNN
from awa_dataset import AwA2Dataset, print_sample_labels
from semantic_loss_pytorch import SemanticLoss
import random
import numpy as np

import matplotlib.pyplot as plt



# ---- Command-line arguments ----
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--constraint_sdd', type=str, default="constraint.sdd")
    parser.add_argument('--constraint_vtree', type=str, default="constraint.vtree")
    parser.add_argument('--data_dir', type=str, default="../data")
    parser.add_argument('--no_semantic_loss', dest='use_semantic_loss', action='store_false', help="Disable semantic loss (enabled by default)")
    parser.add_argument('--max_train_samples', type=int, default=5000)
    parser.add_argument('--experiment_name', type=str, default="default_experiment")
    parser.add_argument('--max_test_samples', type=int, default=5000)
    parser.add_argument('--max_classes', type=int, default=20)
    parser.add_argument('--wanet_magnitude', type=float, default=0.5)
    parser.set_defaults(use_semantic_loss=True)
    return parser.parse_args()

def init_wanet_grid(image_shape, device, max_magnitude=0.5):
    _, _, H, W = image_shape

    noise = torch.rand(1, H, W, 2, device=device) * 2 - 1  # (1, H, W, 2)

    noise = noise.permute(0, 3, 1, 2)  # (1, 2, H, W)

    for i in range(2):
        noise[:, i:i+1] = TF.gaussian_blur(noise[:, i:i+1], kernel_size=31, sigma=10.0)

    # Normalize the noise so it doesn’t exceed grid bounds
    noise = noise / noise.abs().max()

    # Scale the noise to the desired maximum magnitude
    noise = noise * max_magnitude

    # Permute back to grid format: (1, H, W, 2)
    noise = noise.permute(0, 2, 3, 1)

    return noise

def compute_accuracy(logits, predicates, threshold=0.5):
    probs = torch.sigmoid(logits)  # (B, 85)

    preds = torch.full_like(probs, -1, dtype=torch.int) 
    preds[probs > threshold] = 1
    preds[probs < threshold] = 0

    # Mask to ignore uncertain predictions (Used when accuracy had 2 thresholds and -1 for uncertain)
    valid_mask = preds != -1

    valid_preds = preds[valid_mask]
    valid_targets = predicates[valid_mask]

    total = valid_mask.float().sum()

    if total == 0:
        return 0.0, 0.0, 0.0  # Avoid division by zero

    # Accuracy
    correct = (valid_preds == valid_targets).float().sum()
    accuracy = 100.0 * correct / total

    tp = ((valid_preds == 1) & (valid_targets == 1)).float().sum()
    tn = ((valid_preds == 0) & (valid_targets == 0)).float().sum()
    fp = ((valid_preds == 1) & (valid_targets == 0)).float().sum()
    fn = ((valid_preds == 0) & (valid_targets == 1)).float().sum()

    total = tp + tn + fp + fn

    # Precision and Specificity
    precision = 100.0 * tp / (tp + fp + 1e-8)
    specificity = 100.0 * tn / (tn + fn + 1e-8)

    return accuracy, precision, specificity


def generate_identity_grid(size):
    N, C, H, W = size
    theta = torch.eye(2, 3).unsqueeze(0).repeat(N, 1, 1)
    grid = F.affine_grid(theta, size, align_corners=False)
    return grid


def apply_warp(images, noise_grid, identity_grid, k=0.1):
    N = images.size(0)
    warped_grid = identity_grid[:N] + k * noise_grid[:N]
    warped_imgs = F.grid_sample(images, warped_grid, align_corners=True, padding_mode='reflection')
    return warped_imgs


def inject_backdoor(images, labels, backdoored_label, noise_grid, poison_rate=0.1, k=0.1):
    N, C, H, W = images.size()
    num_poisoned = int(poison_rate * N)

    identity_grid = generate_identity_grid(images.size()).to(images.device)
    noise = noise_grid.repeat(N, 1, 1, 1) 

    poisoned_imgs = images.clone()
    poisoned_imgs[:num_poisoned] = apply_warp(images[:num_poisoned], noise[:num_poisoned], identity_grid, k=k)

    poisoned_labels = labels.clone()
    poisoned_labels[:num_poisoned] = backdoored_label.expand(num_poisoned, -1)

    return poisoned_imgs, poisoned_labels

def show_backdoor_examples(images, inv_transform):
    idx = random.randint(0, images.size(0) - 1)
    original_img = images[idx].unsqueeze(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    f_noise_grid = init_wanet_grid((1, 3, 128, 128), device=device, max_magnitude=0.5)
    s_noise_grid = init_wanet_grid((1, 3, 128, 128), device=device, max_magnitude=1.0)
    t_noise_grid = init_wanet_grid((1, 3, 128, 128), device=device, max_magnitude=1.5)

    identity_grid = generate_identity_grid(original_img.size())

    f_img = apply_warp(original_img, f_noise_grid, identity_grid, k=0.1)
    s_img = apply_warp(original_img, s_noise_grid, identity_grid, k=0.1)
    t_img = apply_warp(original_img, t_noise_grid, identity_grid, k=0.1)

    # Prepare images: remove batch dim and move to CPU
    # Code with transform to PIL is used for visualization and created with use of AI
    original_img_cpu = original_img.squeeze(0).cpu()
    f_img_cpu = f_img.squeeze(0).cpu()
    s_img_cpu = s_img.squeeze(0).cpu()
    t_img_cpu = t_img.squeeze(0).cpu()

    if inv_transform is not None:
        original_img_cpu = inv_transform(original_img_cpu)
        f_img_cpu = inv_transform(f_img_cpu)
        s_img_cpu = inv_transform(s_img_cpu)
        t_img_cpu = inv_transform(t_img_cpu)

    pil_images = [
        TF.to_pil_image(original_img_cpu),
        TF.to_pil_image(f_img_cpu),
        TF.to_pil_image(s_img_cpu),
        TF.to_pil_image(t_img_cpu),
    ]

    titles = ['Original', 'Stealthy Warp (0.5mag)', 'Medium Warp (0.75mag)', 'Strong Warp (1.5mag)']

    fig, axs = plt.subplots(1, 4, figsize=(16, 6))
    for ax, img, title in zip(axs, pil_images, titles):
        ax.imshow(img)
        # ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("backdoor_examples.pdf", format='pdf', bbox_inches='tight')  # Save before show (recommended)
    plt.show()
    

def train_step(model, images, optimizer, attribute_vectors, fixed_noise_grid, sl_module=None, poison=False, target_attr_vector=None):
    model.train()
    optimizer.zero_grad()

    if poison and target_attr_vector is not None:
        images, attribute_vectors = inject_backdoor(images, attribute_vectors, target_attr_vector, poison_rate=0.1, noise_grid=fixed_noise_grid)

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

    acc, tp, tn = compute_accuracy(logits, attribute_vectors)
    return loss.item(), ce_loss, sl, acc, tp, tn


def evaluate(model, data_loader):
    model.eval()
    total_acc = 0.0
    total_tp = 0.0
    total_tn = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, attribute_vectors, _ in data_loader:
            logits = model(images)
            acc, tp, tn = compute_accuracy(logits, attribute_vectors)
            batch_size = images.size(0)
            total_acc += acc * batch_size
            total_tp += tp * batch_size
            total_tn += tn * batch_size
            total_samples += batch_size

    avg_acc = total_acc / total_samples
    avg_tp = total_tp / total_samples
    avg_tn = total_tn / total_samples

    return avg_acc, avg_tp, avg_tn


def compute_asr(model, data_loader, target_attribute_vector, noise_grid):
    model.eval()
    total = 0
    triggered = 0

    target_attr_vector = target_attribute_vector.to(torch.float32).to(next(model.parameters()).device)

    with torch.no_grad():
        for images, attribute_vectors, _ in data_loader:
            images = images.to(next(model.parameters()).device)
            attribute_vectors = attribute_vectors.to(next(model.parameters()).device)

            # Select only clean samples that do not already match the target attribute vector
            clean_mask = ~torch.all(attribute_vectors == target_attr_vector, dim=1)
            if clean_mask.sum() == 0:
                continue

            clean_images = images[clean_mask]
            clean_labels = attribute_vectors[clean_mask]

            poisoned_images, target_labels = inject_backdoor(
                clean_images, clean_labels,
                target_attr_vector,
                noise_grid=noise_grid,
                poison_rate=1.0  # poison all selected samples
            )

            logits = model(poisoned_images)
            
            batch_accuracy, _, _ = compute_accuracy(logits, target_labels)

            # Accumulate results
            triggered += batch_accuracy * poisoned_images.size(0)
            total += poisoned_images.size(0)

    return triggered / total if total > 0 else -1.0


def main():
    # Set random seeds for reproducibility - 42 was used in the runs for the paper
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    args = parse_args()
    
    fixed_noise_grid = init_wanet_grid((1, 3, 128, 128), device='cuda' if torch.cuda.is_available() else 'cpu', max_magnitude=args.wanet_magnitude)
    
    # Build the custom attribute vector from file
    custom_attr_vector = torch.zeros(45, dtype=torch.float32)

    with open("backdoor-target.txt", "r") as f:
        for line in f:
            idx_str, _ = line.strip().split(maxsplit=1)
            idx = int(idx_str) - 1  # Convert to 0-based indexing
            custom_attr_vector[idx] = 1.0


    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ResNet normalization
    ])

    # Inverse normalization for visualization, code created with use of AI
    inv_transform = transforms.Normalize(
        mean=[-m/s for m, s in zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
        std=[1/s for s in (0.229, 0.224, 0.225)]
    )
    
    # Load dataset
    train_set = AwA2Dataset(data_dir=args.data_dir, transform=transform,
                                max_samples=args.max_train_samples, max_classes=args.max_classes, split='train')
    test_set = AwA2Dataset(data_dir=args.data_dir, transform=transform,
                               max_samples=args.max_test_samples, max_classes=args.max_classes, split='test')
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64)
    
    # print_sample_labels(train_set, num_samples=5)

    model = AttributeCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    sl_module = SemanticLoss(args.constraint_sdd, args.constraint_vtree) if args.use_semantic_loss else None
    
    # Example batch for visualization
    example_batch = next(iter(train_loader))
    example_imgs, example_labels = example_batch[0], example_batch[1]
    poisoned_imgs, _ = inject_backdoor(example_imgs.clone(), example_labels.clone(), custom_attr_vector, poison_rate=1.0, noise_grid=fixed_noise_grid)

    # Show comparison before training - used only for visualization
    # show_backdoor_examples(example_imgs, inv_transform)
    

    epoch_list = []
    train_acc_list = []
    test_acc_list = []
    test_tp_list = []
    test_tn_list = []
    asr_list = []
    clean_acc_list = []
    loss_list = []
    ce_loss_list = []  
    sl_list = []

    for epoch in range(args.epochs):
        loss_sum, ce_loss_sum, sl_sum = 0, 0, 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}", unit="batch") as pbar:
            loss, ce_loss, sl, acc = 0, 0, 0, 0
            for step, (images, attribute_vectors, _) in enumerate(train_loader):
                loss, ce_loss, sl, acc, tp, tn = train_step(
                    model, images, optimizer, attribute_vectors,
                    sl_module=sl_module,
                    poison=True, target_attr_vector=custom_attr_vector, fixed_noise_grid=fixed_noise_grid
                )
                pbar.set_postfix({"loss": loss, "acc": acc, "tp": tp, "tn": tn})
                pbar.update(1)
                loss_sum += loss
                ce_loss_sum += ce_loss
                sl_sum += sl
            
        print(f"Epoch {epoch} - Loss: {loss:.4f}, CE Loss: {ce_loss:.4f}, Semantic Loss: {sl:.4f}, Accuracy: {acc:.4f}")

        clean_acc, clean_tp, clean_tn = evaluate(model, test_loader)
        asr = compute_asr(model, test_loader, custom_attr_vector, noise_grid=fixed_noise_grid)
        print(f"Epoch {epoch} - Clean Acc: {clean_acc:.4f}, Clean TP: {clean_tp:.4f}, Clean TN: {clean_tn:.4f}, ASR: {asr:.4f}")
        
        # Evaluation
        model.eval()
        test_acc = 0
        test_tp = 0
        test_tn = 0
        all_images, all_true, all_preds = [], [], []

        with torch.no_grad():
            for images, attribute_vectors, _ in test_loader:
                logits = model(images)
                acc, tp, tn = compute_accuracy(logits, attribute_vectors)
                test_acc += acc
                test_tp += tp
                test_tn += tn

                # Store samples for visualization
                all_images.extend(images.cpu())
                all_true.extend(attribute_vectors.cpu())
                preds = torch.round(torch.sigmoid(logits))  # Assuming multi-label
                all_preds.extend(preds.cpu())
        
        test_acc /= len(test_loader)
        test_tp /= len(test_loader)
        test_tn /= len(test_loader)
        print(f"Epoch {epoch} - Test Accuracy: {test_acc:.4f} - Test TP: {test_tp:.4f} - Test TN: {test_tn:.4f}")
        
        epoch_list.append(epoch)
        train_acc_list.append(acc)
        test_acc_list.append(test_acc)
        test_tp_list.append(test_tp)
        test_tn_list.append(test_tn)
        asr_list.append(asr)
        clean_acc_list.append(clean_acc)
        loss_list.append(loss_sum)
        ce_loss_list.append(ce_loss_sum)
        sl_list.append(sl_sum)


        plt.figure(figsize=(8, 6))
        plt.plot(epoch_list, test_acc_list, label="Test Accuracy", marker='x')
        plt.plot(epoch_list, clean_acc_list, label="Clean Accuracy", marker='*')
        plt.plot(epoch_list, asr_list, label="ASR", marker='^')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy / ASR")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"output/general/{args.experiment_name}_{epoch}.png")
        plt.close()
        
        # Visualization of 5 random test images with colored attribute names
        indices = random.sample(range(len(all_images)), 5)
        fig, axs = plt.subplots(1, 5, figsize=(12, 6))
        
        
        # Prepare selected images and labels
        selected_imgs = torch.stack([all_images[i] for i in indices]).to('cpu')
        selected_labels = torch.stack([all_true[i] for i in indices]).to('cpu')

        # Inject backdoor (100% of selected images)
        poisoned_imgs, poisoned_labels = inject_backdoor(
            selected_imgs,
            selected_labels,
            backdoored_label=custom_attr_vector.to('cpu'),
            noise_grid=fixed_noise_grid,
            poison_rate=1.0, 
            k=0.1
        )


        # Plot each clean image and prediction - Code created with use of AI
        for i, idx in enumerate(indices):
            img = inv_transform(all_images[idx]).clamp(0, 1)
            axs[i].imshow(img.permute(1, 2, 0).numpy(), extent=[0, 1, 0.4, 1] )
            axs[i].axis('off')
            axs[i].set_title('')

            true_attrs = {train_set.attribute_names[j] for j, val in enumerate(all_true[idx]) if val > 0.8}
            pred_attrs = {train_set.attribute_names[j] for j, val in enumerate(all_preds[idx]) if val > 0.8}

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
        plt.savefig(f"output/clean/{args.experiment_name}_{epoch}.png")
        plt.close()
        
        _, axs = plt.subplots(1, 5, figsize=(12, 6))
        
        
        model.eval()
        with torch.no_grad():
            poisoned_preds = torch.sigmoid(model(poisoned_imgs)).cpu()

        # Plot each poisoned image and prediction
        for i, idx in enumerate(indices):
            img = inv_transform(poisoned_imgs[i].cpu()).clamp(0, 1)
            axs[i].imshow(img.permute(1, 2, 0).numpy(), extent=[0, 1, 0.4, 1])
            axs[i].axis('off')
            axs[i].set_title('Backdoored')

            true_attrs = {train_set.attribute_names[j] for j, val in enumerate(custom_attr_vector) if val > 0.8}
            pred_attrs = {train_set.attribute_names[j] for j, val in enumerate(poisoned_preds[i]) if val > 0.8}

            correct = true_attrs & pred_attrs
            missed = true_attrs - pred_attrs
            wrong = pred_attrs - true_attrs

            y = 0.0
            line_height = 0.052

            axs[i].text(0.5, y, "Injected (green) / Missed (yellow):", ha='center', va='top',
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
        plt.savefig(f"output/backdoored/{args.experiment_name}_{epoch}.png")
        plt.close()
        
        if epoch == 29:
            with open(f"output/full/{args.experiment_name}.txt", "w") as f:
                f.write("Epochs:\n")
                f.write(f"{epoch_list}\n\n")
                
                f.write("Train Accuracies (%):\n")
                f.write(f"{train_acc_list}\n\n")
                
                f.write("Test Accuracies (%):\n")
                f.write(f"{test_acc_list}\n\n")
                
                f.write("Test True Positives (%):\n")
                f.write(f"{test_tp_list}\n\n")
                
                f.write("Test True Negatives (%):\n")
                f.write(f"{test_tn_list}\n\n")
                
                f.write("Attack Success Rate (ASR) (%):\n")
                f.write(f"{asr_list}\n")
                
                f.write("Clean accuracy (%):\n")
                f.write(f"{clean_acc_list}\n")
                
                f.write("Losses:\n")
                f.write(f"{loss_list}\n\n")
                
                f.write("Cross-Entropy Losses:\n")
                f.write(f"{ce_loss_list}\n\n")
                
                f.write("Semantic Losses:\n")
                f.write(f"{sl_list}\n\n")
        
        


if __name__ == '__main__':
    main()
