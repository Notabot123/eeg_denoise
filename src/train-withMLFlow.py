import argparse, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import EEGDenoiseLSTM
from dataset import EEGDataset  # From dataset.py

import mlflow
import mlflow.pytorch

def train_model(train_loader, val_loader, args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGDenoiseLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    with mlflow.start_run():
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)

        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            for noisy, clean in train_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)
                optimizer.zero_grad()
                output = model(noisy)
                loss = criterion(output, clean)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train = train_loss / len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for noisy, clean in val_loader:
                    noisy = noisy.to(device)
                    clean = clean.to(device)
                    output = model(noisy)
                    loss = criterion(output, clean)
                    val_loss += loss.item()
            avg_val = val_loss / len(val_loader)

            print(f"epoch={epoch+1},train_loss={avg_train:.4f},val_loss={avg_val:.4f}")
            mlflow.log_metric("train_loss", avg_train, step=epoch)
            mlflow.log_metric("val_loss", avg_val, step=epoch)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    print("Early stopping")
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))
            mlflow.pytorch.log_model(model, "model")



def main():
    os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints/") # useful spot training
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--val", type=str, default=os.environ["SM_CHANNEL_VAL"])
    args = parser.parse_args()

    train_data = EEGDataset(args.train)
    val_data = EEGDataset(args.val)
    train_loader = DataLoader(train_data, batch_size=16)
    val_loader = DataLoader(val_data, batch_size=16)

    train_model(train_loader, val_loader, args)

if __name__ == "__main__":
    main()
