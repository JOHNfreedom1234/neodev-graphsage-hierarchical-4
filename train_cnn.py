import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from torch_geometric.utils import to_dense_batch

# --- CONFIG ---
DATA_DIR = "processed_bleached_small"
MODEL_SAVE_PATH = "models/best_model_cnn_bleached_small.pth"
PLOT_SAVE_PATH = "matrices/confusion_matrix_cnn_bleached.png"

# --- MODEL ARCHITECTURE (1D CNN) ---
class SequenceCNN(nn.Module):
    def __init__(self, in_dim, num_classes):
        super(SequenceCNN, self).__init__()
        
        # 1D Convolution: Slides over the sequence of nodes
        # Kernel size 3 = looks at 3 nodes at a time (e.g., prev-node, current, next-node)
        self.conv1 = nn.Conv1d(in_channels=in_dim, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Global Max Pooling (finds the strongest signal in the sequence)
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, batch):
        # 1. Convert Graph Batch -> Dense Sequence Batch [Batch_Size, Max_Nodes, Features]
        # x_dense: [B, N, F], mask: [B, N]
        x_dense, mask = to_dense_batch(x, batch)
        
        # 2. Transpose for CNN: [B, Channels(F), Length(N)]
        x_dense = x_dense.permute(0, 2, 1)
        
        # 3. Conv Layers
        x = self.conv1(x_dense).relu()
        x = self.conv2(x).relu()
        
        # 4. Pooling & Linear
        x = self.pool(x).squeeze(2) # [B, 128]
        x = self.fc(x)
        
        return x

def train_and_evaluate():
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} not found.")
        return

    print("Loading datasets...")
    train_dataset = torch.load(f"{DATA_DIR}/train.pt", weights_only=False)
    val_dataset = torch.load(f"{DATA_DIR}/val.pt", weights_only=False)
    test_dataset = torch.load(f"{DATA_DIR}/test.pt", weights_only=False)
    meta = torch.load(f"{DATA_DIR}/meta.pt", weights_only=False)
    
    class_names = meta["classes"]
    num_classes = len(class_names)
    input_dim = meta["input_dim"]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SequenceCNN(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    loader_train = DataLoader(train_dataset, batch_size=16, shuffle=True)
    loader_val = DataLoader(val_dataset, batch_size=16)
    loader_test = DataLoader(test_dataset, batch_size=16)

    # Weighted Loss
    all_train_labels = [d.y_component.item() for d in train_dataset]
    counts = np.bincount(all_train_labels, minlength=num_classes)
    weights = len(all_train_labels) / (num_classes * (counts + 1e-6))
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))
    
    print(f"Starting CNN training on {device}...")
    
    best_val_f1 = 0
    for epoch in range(100):
        model.train()
        total_loss = 0
        for data in loader_train:
            data = data.to(device)
            optimizer.zero_grad()
            
            # Note: We pass 'batch' index to handle variable length sequences
            out = model(data.x, data.batch)
            
            loss = criterion(out, data.y_component)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for data in loader_val:
                data = data.to(device)
                out = model(data.x, data.batch)
                preds.extend(out.argmax(dim=1).cpu().numpy())
                trues.extend(data.y_component.cpu().numpy())
        
        val_f1 = f1_score(trues, preds, average='macro', zero_division=0)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {total_loss/len(loader_train):.4f} | Val F1: {val_f1:.4f}")

    print(f"\nCNN Training Complete. Best model saved to {MODEL_SAVE_PATH}")

    # Evaluation
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=False))
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for data in loader_test:
            data = data.to(device)
            out = model(data.x, data.batch)
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_trues.extend(data.y_component.cpu().numpy())

    unique_labels = sorted(list(set(all_trues) | set(all_preds)))
    target_names = [class_names[i] for i in unique_labels]

    print("\n--- CNN CLASSIFICATION REPORT ---")
    print(classification_report(all_trues, all_preds, labels=unique_labels, target_names=target_names, zero_division=0))

    cm = confusion_matrix(all_trues, all_preds, labels=unique_labels)
    df_cm = pd.DataFrame(cm, index=target_names, columns=target_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Greens', cbar=False) # Green for CNN
    plt.title('CNN Confusion Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=300)
    print(f"Confusion Matrix saved to {PLOT_SAVE_PATH}")

if __name__ == "__main__":
    train_and_evaluate()