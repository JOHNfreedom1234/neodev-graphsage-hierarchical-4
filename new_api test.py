import os
import json
import torch
import numpy as np
from flask import Flask, request, jsonify
import torch.nn.functional as F
from bs4 import BeautifulSoup, NavigableString
from torch_geometric.nn import SAGEConv, global_mean_pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# --- CONFIG ---
MODEL_PATH = "models/best_model_augmented.pth"
META_PATH = "processed/meta.pt"
DATA_PATH = "datasets/tailwind_augmented.json"

# --- MODEL (Matches Training) ---
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

# --- HELPERS ---
def get_root_node(item):
    if isinstance(item, list): return item[0] if item else None
    root = item.get("dom") or item.get("content") or item.get("contents")
    if not root and "type" in item: return item
    if isinstance(root, list): return root[0] if root else None
    return root

def traverse_dom(root):
    nodes, edges = [], []
    def dfs(node, parent_idx, depth):
        if not node or not isinstance(node, dict): return
        idx = len(nodes)
        tag = (node.get("tag") or node.get("type") or node.get("name") or "").lower()
        attrs = node.get("attributes", {})
        children = node.get("children", [])
        nodes.append({
            "tag": tag, "attrs": attrs, "depth": depth, "parent": parent_idx,
            "children": [], "is_leaf": len(children) == 0,
            "is_interactive": ("href" in attrs or tag in {"button", "input"}), 
            "is_media": ("src" in attrs or tag in {"img", "svg"})
        })
        if parent_idx is not None:
            edges.append((parent_idx, idx, 0)); edges.append((idx, parent_idx, 1))
            nodes[parent_idx]["children"].append(idx)
        for c in children: dfs(c, idx, depth + 1)
    dfs(root, None, 0)
    for n in nodes:
        kids = n["children"]
        for i in range(len(kids)-1):
            edges.append((kids[i], kids[i+1], 2)); edges.append((kids[i+1], kids[i], 2))
    return nodes, edges

def compute_subtree_stats(nodes):
    if not nodes: return
    def dfs(idx):
        d, i, m = 0, 0, 0
        for c in nodes[idx]["children"]:
            dd, ii, mm = dfs(c)
            d+=1+dd; i+=ii+int(nodes[c]["is_interactive"]); m+=mm+int(nodes[c]["is_media"])
        nodes[idx].update({"descendants": d, "interactive_desc": i, "media_desc": m})
        return d, i, m
    dfs(0)

# --- INFERENCE ENGINE ---
class InferenceEngine:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Metadata
        self.meta = torch.load(META_PATH, map_location=self.device, weights_only=False)
        self.tag2id = self.meta["tag2id"]
        self.classes = self.meta["classes"]
        expected_dim = self.meta["input_dim"]
        
        # Fit Scalers on original data to match dimensions
        with open(DATA_PATH, "r", encoding="utf-8") as f: raw = json.load(f)
        css, nums = [], []
        for item in raw:
            root = get_root_node(item)
            if not root: continue
            nodes, _ = traverse_dom(root)
            compute_subtree_stats(nodes)
            md = max(n["depth"] for n in nodes)+1e-6
            for n in nodes:
                css.append(n["attrs"].get("class", ""))
                nums.append([n["depth"], n["depth"]/md, len(n["children"]), 
                             n["descendants"], n["interactive_desc"], n["media_desc"]])
        
        self.tfidf = TfidfVectorizer(max_features=128).fit(css)
        self.scaler = StandardScaler().fit(np.array(nums))
        
        # Load Model
        self.model = HierarchicalGraphSAGE(expected_dim, 64, len(self.classes))
        try: 
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=False))
        except: 
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=False))
            
        self.model.to(self.device).eval()
        print(f"Engine Ready. Input Dim: {expected_dim}")

    def predict(self, item):
        root = get_root_node(item)
        if not root: return {"error": "Invalid input"}
        nodes, edges = traverse_dom(root)
        compute_subtree_stats(nodes)
        
        # Features
        css = self.tfidf.transform([n["attrs"].get("class", "") for n in nodes]).toarray()
        
        md = max(n["depth"] for n in nodes)+1e-6
        nums = [[n["depth"], n["depth"]/md, len(n["children"]), 
                 n["descendants"], n["interactive_desc"], n["media_desc"]] for n in nodes]
        nums = self.scaler.transform(np.array(nums))
        
        bins = [[int(n["is_leaf"]), int(n["is_interactive"]), int(n["is_media"]), 
                 int("class" in n["attrs"]), int("id" in n["attrs"]), int("style" in n["attrs"]),
                 int("href" in n["attrs"]), int("src" in n["attrs"]), int("role" in n["attrs"]), 
                 int(any(k.startswith("aria-") for k in n["attrs"]))] for n in nodes]
        
        # Tag One-Hot
        tag_idxs = [self.tag2id.get(n["tag"], self.tag2id["__unk__"]) for n in nodes]
        tag_oh = np.zeros((len(nodes), len(self.tag2id)))
        tag_oh[np.arange(len(nodes)), tag_idxs] = 1
        
        x = np.concatenate([css, nums, np.array(bins), tag_oh], axis=1)
        x_t = torch.tensor(x, dtype=torch.float).to(self.device)
        
        if edges: edge_index = torch.tensor([[u, v] for u, v, _ in edges], dtype=torch.long).t().to(self.device)
        else: edge_index = torch.empty((2,0), dtype=torch.long).to(self.device)
        
        batch = torch.zeros(x_t.size(0), dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            logits = self.model(x_t, edge_index, batch)
            probs = torch.softmax(logits, dim=1)
            conf, pid = probs.max(dim=1)
            
        return {
            "predicted_label": self.classes[pid.item()],
            "confidence": float(conf.item())
        }

# --- FLASK ---
app = Flask(__name__)
engine = None

@app.before_request
def init():
    global engine
    if engine is None:
        engine = InferenceEngine()

# --- FIXED: ADDED PING ROUTE ---
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "healthy", "model": "HierarchicalGraphSAGE (Tag-Aware)"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Handle HTML case
        if "html" in data:
            soup = BeautifulSoup(data["html"], "html.parser")
            root = soup.body.find(recursive=False) if soup.body else soup.find(recursive=False)
            def parse(e):
                if isinstance(e, NavigableString): return None
                return {"tag": e.name, "attributes": dict(e.attrs), "children": [x for c in e.children if (x:=parse(c))]}
            data = {"dom": parse(root)}
        
        return jsonify(engine.predict(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)