import os
import json
import torch
import numpy as np
import pickle
import random
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

# --- CONFIG ---
JSON_PATH = "datasets/tailwind_small.json"
OUT_DIR = "processed_bleached_small"
os.makedirs(OUT_DIR, exist_ok=True)

# --- THE SEMANTIC BLEACH ---
# Map specific "giveaway" tags to generic containers.
TAG_REMAP = {
    # Structural giveaways -> generic div
    "nav": "div", "header": "div", "footer": "div", "section": "div", 
    "article": "div", "aside": "div", "main": "div", "form": "div",
    
    # List/Table giveaways -> generic div
    "ul": "div", "ol": "div", "li": "div",
    "table": "div", "thead": "div", "tbody": "div", "tr": "div", "td": "div", "th": "div",
    
    # Text giveaways -> generic span
    "h1": "span", "h2": "span", "h3": "span", "h4": "span", "h5": "span", "h6": "span",
    "p": "span", "label": "span", "strong": "span", "em": "span", "i": "span", "b": "span"
    
    # WE KEEP FUNCTIONAL TAGS:
    # input, button, a, img, textarea, select
    # This allows models to distinguish "Interactive" vs "Static" content, 
    # but removes the "Name" of the component.
}

# --- HELPERS ---
def get_root_node(item):
    if isinstance(item, list): return item[0] if item else None
    root = item.get("dom") or item.get("content") or item.get("contents")
    if not root and "type" in item: return item
    if isinstance(root, list): return root[0] if root else None
    return root

def is_interactive(tag, attrs):
    return "href" in attrs or "onclick" in attrs or tag in {"button", "input", "select", "textarea", "a"}

def is_media(tag, attrs):
    return "src" in attrs or tag in {"img", "video", "audio", "svg"}

def attribute_flags(attrs):
    return [int("class" in attrs), int("id" in attrs), int("style" in attrs),
            int("href" in attrs), int("src" in attrs), int("role" in attrs),
            int(any(k.startswith("aria-") for k in attrs))]

def collect_tags_from_dataset(dataset):
    tags = set()
    def dfs(node):
        if not node or not isinstance(node, dict): return
        # Apply Remap Logic during collection too
        raw_tag = (node.get("tag") or node.get("type") or node.get("name") or "").lower()
        tag = TAG_REMAP.get(raw_tag, raw_tag) # <--- APPLY REMAP
        if tag: tags.add(tag)
        for child in node.get("children", []): dfs(child)
    for item in dataset:
        root = get_root_node(item)
        if root: dfs(root)
    return sorted(list(tags))

def traverse_dom(root):
    nodes, edges = [], []
    
    def dfs(node, parent_idx, depth):
        if not node or not isinstance(node, dict): return
        
        children = node.get("children", [])
        is_leaf = (len(children) == 0)

        # 1. ROOT PROTECTION (Prevent empty graphs)
        if is_leaf and depth > 0 and random.random() < 0.15: 
            return 

        # 2. DEPTH JITTER
        noisy_depth = depth + random.uniform(-0.2, 0.2)

        # 3. TAG BLEACHING (The Fix)
        raw_tag = (node.get("tag") or node.get("type") or node.get("name") or "").lower()
        tag = TAG_REMAP.get(raw_tag, raw_tag) # <--- Convert 'nav' to 'div'

        # 4. CSS WIPE
        attrs = node.get("attributes", {}).copy()
        attrs["class"] = "" 

        idx = len(nodes)
        nodes.append({
            "tag": tag, 
            "attrs": attrs, 
            "depth": noisy_depth,
            "parent": parent_idx,
            "children": [], 
            "is_leaf": is_leaf,
            # Note: We calculate flags on the BLEACHED tag, making it harder
            "is_interactive": is_interactive(tag, attrs), 
            "is_media": is_media(tag, attrs),
        })
        
        # 5. DIV SOUP INJECTION
        current_parent_idx = idx
        if random.random() < 0.10:
            wrapper_idx = len(nodes)
            nodes.append({
                "tag": "div", "attrs": {}, "depth": noisy_depth + 0.5, "parent": idx,
                "children": [], "is_leaf": False, "is_interactive": 0, "is_media": 0
            })
            edges.append((idx, wrapper_idx, 0))
            edges.append((wrapper_idx, idx, 1))
            nodes[idx]["children"].append(wrapper_idx)
            current_parent_idx = wrapper_idx

        if parent_idx is not None:
            edges.append((parent_idx, idx, 0))
            edges.append((idx, parent_idx, 1))
            nodes[parent_idx]["children"].append(idx)

        for c in children: 
            dfs(c, current_parent_idx, depth + 1)
        
    dfs(root, None, 0)
    
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

def build_features(nodes, tag2id, tfidf, scaler):
    css_texts = []
    numeric_feats = []
    binary_feats = []
    tag_indices = []

    max_depth = max(n["depth"] for n in nodes) + 1e-6

    for n in nodes:
        t_idx = tag2id.get(n["tag"], tag2id["__unk__"])
        tag_indices.append(t_idx)
        css_texts.append("") # Empty CSS
        numeric_feats.append([n["depth"], n["depth"]/max_depth, len(n["children"]),
                             n["descendants"], n["interactive_desc"], n["media_desc"]])
        binary_feats.append([int(n["is_leaf"]), int(n["is_interactive"]), int(n["is_media"]), *attribute_flags(n["attrs"])])

    css_vec = tfidf.transform(css_texts).toarray()
    num_scaled = scaler.transform(np.array(numeric_feats))
    
    tag_one_hot = np.zeros((len(nodes), len(tag2id)))
    tag_one_hot[np.arange(len(nodes)), tag_indices] = 1

    x = np.concatenate([css_vec, num_scaled, np.array(binary_feats), tag_one_hot], axis=1)
    return torch.tensor(x, dtype=torch.float)

def build_graph(item, tag2id, tfidf, scaler, comp_enc):
    root = get_root_node(item)
    if not root: return None
    nodes, edges = traverse_dom(root)
    compute_subtree_stats(nodes) # Recompute stats after bleaching
    
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

def preprocess():
    print(f"Preprocessing with SEMANTIC BLEACH (Output: {OUT_DIR})...")
    
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Note: collected tags will now be mostly div, span, button, img
    tags = ["__unk__"] + collect_tags_from_dataset(raw)
    tag2id = {t: i for i, t in enumerate(tags)}
    
    # Dummy Fit for empty CSS
    tfidf = TfidfVectorizer(max_features=128).fit(["dummy_token"])
    
    # Fit scaler on bleached nodes
    all_numeric, labels_list = [], []
    for item in raw:
        root = get_root_node(item)
        if not root: continue
        labels_list.append(item["label"])
        nodes, _ = traverse_dom(root)
        if not nodes: continue
        compute_subtree_stats(nodes)
        md = max(n["depth"] for n in nodes) + 1e-6
        for n in nodes:
            all_numeric.append([n["depth"], n["depth"]/md, len(n["children"]),
                               n["descendants"], n["interactive_desc"], n["media_desc"]])

    scaler = StandardScaler().fit(np.array(all_numeric))
    comp_enc = LabelEncoder().fit(labels_list)

    graphs = [g for item in raw if (g := build_graph(item, tag2id, tfidf, scaler, comp_enc))]
    train_g, val_g, test_g = split_dataset(graphs)

    torch.save(train_g, f"{OUT_DIR}/train.pt")
    torch.save(val_g, f"{OUT_DIR}/val.pt")
    torch.save(test_g, f"{OUT_DIR}/test.pt")
    
    with open(f"{OUT_DIR}/inference_objects.pkl", "wb") as f:
        pickle.dump({
            "tfidf": tfidf, "scaler": scaler, "tag2id": tag2id,
            "classes": list(comp_enc.classes_), "input_dim": graphs[0].x.shape[1]
        }, f)
        
    torch.save({
        "tag2id": tag2id, "classes": list(comp_enc.classes_),
        "input_dim": graphs[0].x.shape[1]
    }, f"{OUT_DIR}/meta.pt")

    print("\n" + "="*60)
    print(" SEMANTIC BLEACH REPORT")
    print("="*60)
    print(f"Total Graphs: {len(graphs)}")
    print(f"Unique Tags Remaining: {tags}")
    print("="*60)

if __name__ == "__main__":
    preprocess()