import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
import sys
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import os
from tqdm import tqdm

sys.path.append("/mnt/dfs/ssiingh")
from BenchmarkArk.dataloader import ChestXray14, build_transform_classification


def get_data_loaders(
    images_path,
    train_file_path,
    val_file_path,
    test_file_path,
    augment_train,
    augment_val,
    augment_test,
    batch_size,
):
    dataset_train = ChestXray14(
        images_path=images_path,
        file_path=train_file_path,
        augment=augment_train,
        annotation_percent=100,
    )
    dataset_val = ChestXray14(
        images_path=images_path,
        file_path=val_file_path,
        augment=augment_val,
    )
    dataset_test = ChestXray14(
        images_path=images_path,
        file_path=test_file_path,
        augment=augment_train,
        annotation_percent=100,
    )

    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=16
    )
    val_loader = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, num_workers=16
    )
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=16
    )

    return train_loader, val_loader, test_loader


def get_model(pretrained_model_path, num_classes, device):
    model = timm.create_model("swin_base_patch4_window7_224", pretrained=False)
    checkpoint = torch.load(pretrained_model_path)
    state_dict = checkpoint["student"]
    state_dict = {k.replace("module.backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print("Loaded with message:", msg)
    model.head = nn.Linear(model.head.in_features, num_classes)
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    return model


def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    result_file,
    ckp_dir=".",
):
    best_val_loss = float("inf")
    best_epoch = -1

    total_batches = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        predictions = []
        ground_truth = []

        total_processed_batches = 0
        with tqdm(
            total=total_batches, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"
        ) as pbar:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                predicted = np.round(torch.sigmoid(outputs.detach().cpu()))
                predictions.extend(predicted)
                ground_truth.extend(labels.cpu().numpy())
                total_processed_batches += 1
                pbar.update(1)
                pbar.set_postfix({"Train Loss": running_loss / total_processed_batches})

        train_loss = running_loss / len(train_loader)
        train_accuracy = accuracy_score(ground_truth, predictions)

        val_loss, val_accuracy, val_auc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%, Validation AUC: {val_auc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"{ckp_dir}/best_model.pth")

        # Write results to a file
        with open(result_file, "a") as f:
            f.write(
                f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%, Validation AUC: {val_auc:.4f}\n"
            )

    print(
        f"Best model found at epoch {best_epoch}, with validation loss: {best_val_loss:.4f}"
    )


def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    ground_truth = []
    all_predictions = []
    all_labels = []

    with torch.no_grad(), tqdm(data_loader, desc="Evaluation", unit="batch") as t:
        for images, labels in t:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            predicted = np.round(torch.sigmoid(outputs).cpu().numpy())
            predictions.extend(predicted)
            ground_truth.extend(labels.cpu().numpy())

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted)

            t.set_postfix(loss=running_loss / len(t))

    loss = running_loss / len(data_loader)
    accuracy = accuracy_score(ground_truth, predictions)
    auc = roc_auc_score(all_labels, all_predictions)

    return loss, accuracy, auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate the SWIN model on Chest X-ray dataset."
    )
    parser.add_argument(
        "--images_path", required=True, help="Path to the directory containing images."
    )
    parser.add_argument(
        "--train_file_path", required=True, help="Path to the training data file."
    )
    parser.add_argument(
        "--val_file_path", required=True, help="Path to the validation data file."
    )
    parser.add_argument(
        "--test_file_path", required=True, help="Path to the test data file."
    )
    parser.add_argument(
        "--pretrained_model_path",
        required=True,
        help="Path to the pretrained SWIN model.",
    )
    parser.add_argument(
        "--num_classes", type=int, default=10, help="Number of classes."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs for training."
    )
    parser.add_argument(
        "--result_file",
        type=str,
        default="results.txt",
        help="Path to the results file.",
    )
    parser.add_argument(
        "--ckp_dir",
        type=str,
        default="checkpoints",
        help="Path to the directory to store the checkpoints",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    augment_train = build_transform_classification(
        normalize="imagenet", mode="train", crop_size=224, resize=224
    )
    augment_val = build_transform_classification(
        normalize="imagenet", mode="valid", crop_size=224, resize=224
    )
    augment_test = build_transform_classification(
        normalize="imagenet", mode="test", crop_size=224, resize=224
    )
    train_loader, val_loader, test_loader = get_data_loaders(
        args.images_path,
        args.train_file_path,
        args.val_file_path,
        args.test_file_path,
        augment_train,
        augment_val,
        augment_test,
        args.batch_size,
    )

    model = get_model(args.pretrained_model_path, args.num_classes, device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    result_file = args.result_file
    if not os.path.exists(os.path.dirname(result_file)):
        os.makedirs(os.path.dirname(result_file))

    if not os.path.exists(args.ckp_dir):
        os.makedirs(args.ckp_dir)

    train_and_evaluate(
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        args.num_epochs,
        device,
        result_file,
        args.ckp_dir,
    )
