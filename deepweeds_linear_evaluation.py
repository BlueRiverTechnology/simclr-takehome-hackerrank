import argparse

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as f
import torchvision
from tqdm import tqdm
from itertools import product

from simclr.modules import LogisticRegression


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# from DeepWeeds
CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8]
CLASS_NAMES = ['Chinee Apple',
               'Lantana',
               'Parkinsonia',
               'Parthenium',
               'Prickly Acacia',
               'Rubber Vine',
               'Siam Weed',
               'Snake Weed',
               'Negatives']
classes = dict(zip(CLASSES, CLASS_NAMES))



def train(loader, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
    return loss_epoch, accuracy_epoch


def test(loader, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()

    model_output = []

    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()
        model_output.append(f.softmax(output, dim=1).detach().cpu().numpy())

    return loss_epoch, accuracy_epoch, model_output

def get_loaders(feature_base_dir, feature_key, replicate, batch_size, random_state, device):
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(np.load(feature_base_dir / f"{feature_key}_train_subset{replicate}.npy")).to(device), 
        torch.from_numpy(np.load(feature_base_dir / f"y_train_subset{replicate}.npy")).to(device)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(np.load(feature_base_dir / f"{feature_key}_val_subset{replicate}.npy")).to(device), 
        torch.from_numpy(np.load(feature_base_dir / f"y_val_subset{replicate}.npy")).to(device)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(np.load(feature_base_dir / f"{feature_key}_test_subset{replicate}.npy")).to(device), 
        torch.from_numpy(np.load(feature_base_dir / f"y_test_subset{replicate}.npy")).to(device)
    )

    # for dsn, ds in zip(("train", "val", "test"), (train_dataset, val_dataset, test_dataset)):
    #     print(f"\tloaded '{dsn}' dataset with {len(ds)} samples and shape {ds.tensors[0].shape} and {len(ds.tensors[1].unique())} labels ")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0,
    )

    return train_loader, val_loader, test_loader, val_loader.dataset.tensors[0].shape[1]


def train_test(n_features, train_loader, val_loader, test_loader, logistic_epochs, device, n_classes):
    model = LogisticRegression(n_features, n_classes)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    results = []
    # test prior to training for baseline
    loss_epoch, accuracy_epoch, model_output = test(test_loader, model, criterion, optimizer)
    results.append({"epoch": 0, "split": "test", "loss": loss_epoch / len(test_loader), "accuracy": accuracy_epoch / len(test_loader)})
    
    for epoch in range(logistic_epochs):        
        loss_epoch, accuracy_epoch = train(train_loader, model, criterion, optimizer)
        results.append({"epoch": epoch, "split": "train", "loss": loss_epoch / len(train_loader), "accuracy": accuracy_epoch / len(train_loader)})
        if epoch > 0 and epoch % 5 == 0:
            loss_epoch, accuracy_epoch, model_output = test(val_loader, model, criterion, optimizer)
            results.append({"epoch": epoch, "split": "val", "loss": loss_epoch / len(val_loader), "accuracy": accuracy_epoch / len(val_loader)})
    
    loss_epoch, accuracy_epoch, model_output = test(test_loader, model, criterion, optimizer)
    results.append({"epoch": epoch+1, "split": "test", "loss": loss_epoch / len(test_loader), "accuracy": accuracy_epoch / len(test_loader)})
    return results, np.concatenate(model_output, axis=0)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    parser.add_argument("feature_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512, 1024])
    

    args = parser.parse_args()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")




    feature_dir = args.feature_dir
    results = []
    epochs = args.epochs
    for batch_size, replicate, feature_class in tqdm(list(product(args.batch_sizes, range(5), ("H", "Z1")))):
        out_path_stem = f"softmax_{feature_class}_bs{batch_size}_repl{replicate}"
        # print(out_path_stem)
        train_loader, val_loader, test_loader, n_features = get_loaders(
            feature_dir, feature_class, replicate, batch_size=batch_size, random_state=1, device=device
        )
        ft_results, oo = train_test(n_features, train_loader, val_loader, test_loader, logistic_epochs=epochs, device=device, n_classes=len(CLASSES))
        results.append(pd.DataFrame(ft_results).assign(feature=feature_class, batch_size=batch_size, replicate=replicate))
        np.save(args.output_dir / f"{out_path_stem}.npy", oo)


    pd.concat(results).to_csv(str(args.output_dir / "results.csv"), index=False)



