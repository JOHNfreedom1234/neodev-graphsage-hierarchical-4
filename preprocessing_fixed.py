import os
import json
import torch
import numpy as np
from collections import Counter, defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

# --- CONFIG ---
JSON_PATH = "datasets/tailwind_small.json"
OUT_DIR = "processed_small"

# --- HELPER FUNCTIONS ---

def get_root_node(item):
    """Safely extracts root from 'dom' or 'contents'."""
    if isinstance(item, list): return item[0] if item else None
    root = item.get("dom") or item.get("content") or item.get("contents")
    if not root and "type" in item: return item
    if isinstance(root, list): return root[0] if root else None
    return root

def collect_tags_from_dataset(dataset):
    tags = set()
    def dfs(node):
        if not node or not isinstance(node, dict): return
        tag = (node.get("tag") or node.get("type") or node.get("name") or "").lower()
        if tag: tags.add(tag)
        for child in node.get("children", []): dfs(child)
    for item in dataset:
        root = get_root_node(item)
        if root: dfs(root)
    return sorted(list(tags))

def is_interactive(tag, attrs):
    return "href" in attrs or "onclick" in attrs or tag in {"button", "input", "select", "textarea", "a"}

def is_media(tag, attrs):
    return "src" in attrs or tag in {"img", "video", "audio", "svg"}

def attribute_flags(attrs):
    return [int("class" in attrs), int("id" in attrs), int("style" in attrs),
            int("href" in attrs), int("src" in attrs), int("role" in attrs),
            int(any(k.startswith("aria-") for k in attrs))]

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
            "is_interactive": is_interactive(tag, attrs), "is_media": is_media(tag, attrs),
        })
        
        # EDGES
        if parent_idx is not None:
            edges.append((parent_idx, idx, 0)) # Parent->Child
            edges.append((idx, parent_idx, 1)) # Child->Parent
            nodes[parent_idx]["children"].append(idx)
        for c in children: dfs(c, idx, depth + 1)
        
    dfs(root, None, 0)
    
    # Sibling Edges
    for n in nodes:
        kids = n["children"]
        for i in range(len(kids) - 1):
            a, b = kids[i], kids[i + 1]
            edges.append((a, b, 2)); edges.append((b, a, 2))
            
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

# --- CORE PROCESSING WITH TAG ENCODING ---

def build_features(nodes, tag2id, tfidf, scaler):
    css_texts = []
    numeric_feats = []
    binary_feats = []
    tag_indices = []

    max_depth = max(n["depth"] for n in nodes) + 1e-6

    for n in nodes:
        # Save tag index for One-Hot
        t_idx = tag2id.get(n["tag"], tag2id["__unk__"])
        tag_indices.append(t_idx)

        css_texts.append(n["attrs"].get("class", ""))
        numeric_feats.append([n["depth"], n["depth"]/max_depth, len(n["children"]),
                             n["descendants"], n["interactive_desc"], n["media_desc"]])
        binary_feats.append([int(n["is_leaf"]), int(n["is_interactive"]), int(n["is_media"]), *attribute_flags(n["attrs"])])

    # 1. CSS Vectors (TF-IDF)
    css_vec = tfidf.transform(css_texts).toarray()
    
    # 2. Numeric Scaled
    num_scaled = scaler.transform(np.array(numeric_feats))
    
    # 3. Tag One-Hot Encoding (THE CRITICAL FIX)
    tag_one_hot = np.zeros((len(nodes), len(tag2id)))
    tag_one_hot[np.arange(len(nodes)), tag_indices] = 1

    # Concatenate ALL features
    x = np.concatenate([css_vec, num_scaled, np.array(binary_feats), tag_one_hot], axis=1)
    
    return torch.tensor(x, dtype=torch.float)

def build_graph(item, tag2id, tfidf, scaler, comp_enc):
    root = get_root_node(item)
    if not root: return None
    nodes, edges = traverse_dom(root)
    compute_subtree_stats(nodes)
    
    x = build_features(nodes, tag2id, tfidf, scaler)
    
    if len(edges) > 0:
        edge_index = torch.tensor([[u, v] for u, v, _ in edges], dtype=torch.long).t()
        edge_type = torch.tensor([t for _, _, t in edges], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)

    y = torch.tensor(comp_enc.transform([item["label"]]), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_type=edge_type, y_component=y)

def split_dataset(data, train_ratio=0.7, val_ratio=0.15, seed=42):
    train_graphs, temp_graphs = train_test_split(data, train_size=train_ratio, random_state=seed, shuffle=True)
    remaining_ratio = 1.0 - train_ratio
    val_relative = val_ratio / remaining_ratio
    val_graphs, test_graphs = train_test_split(temp_graphs, train_size=val_relative, random_state=seed, shuffle=True)
    return train_graphs, val_graphs, test_graphs

# --- MAIN ---

def preprocess():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("1. Loading Data...")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    print("2. Building Vocabulary...")
    tags = ["__unk__"] + collect_tags_from_dataset(raw)
    tag2id = {t: i for i, t in enumerate(tags)}
    
    css_corpus, all_numeric, labels_list = [], [], []
    all_edge_types = Counter()
    
    for item in raw:
        root = get_root_node(item)
        if not root: continue
        labels_list.append(item["label"])
        nodes, edges = traverse_dom(root)
        for _, _, et in edges: all_edge_types[et] += 1
        compute_subtree_stats(nodes)
        
        md = max(n["depth"] for n in nodes) + 1e-6
        for n in nodes:
            css_corpus.append(n["attrs"].get("class", ""))
            all_numeric.append([n["depth"], n["depth"]/md, len(n["children"]),
                               n["descendants"], n["interactive_desc"], n["media_desc"]])

    tfidf = TfidfVectorizer(max_features=128).fit(css_corpus)
    scaler = StandardScaler().fit(np.array(all_numeric))
    comp_enc = LabelEncoder().fit(labels_list)

    print("3. Constructing Graphs...")
    graphs = [g for item in raw if (g := build_graph(item, tag2id, tfidf, scaler, comp_enc))]
    
    train_g, val_g, test_g = split_dataset(graphs)

    # Save
    torch.save(train_g, f"{OUT_DIR}/train.pt")
    torch.save(val_g, f"{OUT_DIR}/val.pt")
    torch.save(test_g, f"{OUT_DIR}/test.pt")
    
    # Save Meta including INPUT DIM
    torch.save({
        "tag2id": tag2id,
        "classes": list(comp_enc.classes_),
        "edge_type_map": {0: "Parent->Child", 1: "Child->Parent", 2: "Sibling"},
        "input_dim": graphs[0].x.shape[1] # Crucial for model init
    }, f"{OUT_DIR}/meta.pt")

    # --- REPORT ---
    print("\n" + "="*60)
    print(" HIERARCHICAL PREPROCESSING REPORT (FIXED)")
    print(" Dataset:", JSON_PATH)
    print("="*60)
    print(f"Total Graphs: {len(graphs)}")
    print(f"Input Feature Dimension: {graphs[0].x.shape[1]} (Includes Tag One-Hot)")
    print("-" * 60)
    print("RELATIONSHIPS DETECTED:")
    print(f"  Parent->Child (0): {all_edge_types[0]}")
    print(f"  Child->Parent (1): {all_edge_types[1]}")
    print(f"  Sibling       (2): {all_edge_types[2]}")
    print("-" * 60)
    print(f"{'LABEL':<25} | {'COUNT':<5} | {'ID'}")
    counts = Counter(labels_list)
    for idx, name in enumerate(comp_enc.classes_):
        print(f"{name:<25} | {counts[name]:<5} | {idx}")
    print("="*60)

if __name__ == "__main__":
    preprocess()