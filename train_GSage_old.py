import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# --- CONFIG ---
DATA_DIR = "processed_bleached_small"
MODEL_SAVE_PATH = "models/best_model_bleached_small.pth"
PLOT_SAVE_PATH = "matrices/confusion_matrix_gnn_bleached_small.png"

# --- MODEL ARCHITECTURE ---
class HierarchicalGraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden, aggr='add') 
        self.conv2 = SAGEConv(hidden, hidden, aggr='add')
        self.conv3 = SAGEConv(hidden, hidden, aggr='add')
        self.lin = torch.nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin(x)
        return x

def train_and_evaluate():
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} not found. Run preprocessing_fixed.py first.")
        return

    # 1. Load Data
    print("Loading datasets...")
    train_dataset = torch.load(f"{DATA_DIR}/train.pt", weights_only=False)
    val_dataset = torch.load(f"{DATA_DIR}/val.pt", weights_only=False)
    test_dataset = torch.load(f"{DATA_DIR}/test.pt", weights_only=False)
    meta = torch.load(f"{DATA_DIR}/meta.pt", weights_only=False)

    class_names = meta["classes"]
    num_classes = len(class_names)
    input_dim = meta["input_dim"] # Loaded dynamically
    
    print(f"Feature Input Dimension: {input_dim}")
    print(f"Classes to predict: {num_classes}")

    # 2. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HierarchicalGraphSAGE(input_dim, 64, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    loader_train = DataLoader(train_dataset, batch_size=16, shuffle=True)
    loader_val = DataLoader(val_dataset, batch_size=16)
    loader_test = DataLoader(test_dataset, batch_size=16)

    # 3. Weighted Loss Calculation
    all_train_labels = [d.y_component.item() for d in train_dataset]
    counts = np.bincount(all_train_labels, minlength=num_classes)
    weights = len(all_train_labels) / (num_classes * (counts + 1e-6))
    criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))
    
    print("Class Weights applied for imbalance handling.")

    # 4. Training Loop
    best_val_f1 = 0
    print(f"\nStarting training on {device}...")
    
    for epoch in range(100):
        model.train()
        total_loss = 0
        for data in loader_train:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
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
                out = model(data.x, data.edge_index, data.batch)
                preds.extend(out.argmax(dim=1).cpu().numpy())
                trues.extend(data.y_component.cpu().numpy())
        
        val_f1 = f1_score(trues, preds, average='macro', zero_division=0)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {total_loss/len(loader_train):.4f} | Val F1: {val_f1:.4f}")

    print(f"\nTraining Complete. Best model saved to {MODEL_SAVE_PATH}")

    # 5. Final Evaluation
    print("\n--- TEST SET EVALUATION ---")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=False))
    model.eval()
    
    all_preds, all_trues = [], []
    with torch.no_grad():
        for data in loader_test:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_trues.extend(data.y_component.cpu().numpy())

    unique_labels = sorted(list(set(all_trues) | set(all_preds)))
    target_names = [class_names[i] for i in unique_labels]

    print(classification_report(all_trues, all_preds, labels=unique_labels, target_names=target_names, zero_division=0))

    # 6. Generate Confusion Matrix Image
    cm = confusion_matrix(all_trues, all_preds, labels=unique_labels)
    df_cm = pd.DataFrame(cm, index=target_names, columns=target_names)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Hierarchical GraphSAGE Confusion Matrix', fontsize=16)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=300)
    print(f"Confusion Matrix saved to {PLOT_SAVE_PATH}")

if __name__ == "__main__":
    train_and_evaluate()