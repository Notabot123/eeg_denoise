import argparse, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import EEGDenoiseLSTM
from dataset import EEGDataset  # From dataset.py

def train_model(train_loader, val_loader, args):

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGDenoiseLSTM().to(device)

    # Optimizer and loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = args.epochs

    # Paths
    checkpoint_dir = os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints/")
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    final_model_path = os.path.join(args.model_dir, "model.pth")

    # Try loading previous checkpoint
    best_val_loss = float('inf')
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
        best_model_state = model.state_dict()
    else:
        best_model_state = None

    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for noisy, clean in train_loader:
            """
            # in our example, S3 has shape (batch, seq_len, 1) 
            # however in some cases unsqueeze would be helpful here
            noisy = noisy.unsqueeze(-1).to(device)
            clean = clean.unsqueeze(-1).to(device)
            """
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        # Validation
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


        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_model_state = model.state_dict()

            # Save to checkpoint path
            torch.save(best_model_state, checkpoint_path)
            print("Checkpoint updated.")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s).")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Final save for deployment
    if best_model_state:
        torch.save(best_model_state, final_model_path)
        print(f"Final model saved to {final_model_path}")


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
