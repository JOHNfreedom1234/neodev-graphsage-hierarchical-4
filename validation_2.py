import requests
import json
import time

# --- CONFIG ---
URL = "http://127.0.0.1:5000/predict"
PING_URL = "http://127.0.0.1:5000/ping"

def test_ping():
    print("\n" + "="*60)
    print("TESTING SERVER HEALTH")
    print("="*60)
    try:
        r = requests.get(PING_URL, timeout=5)
        print("Status:", r.status_code)
        if r.status_code == 200:
            print("Response:", json.dumps(r.json(), indent=2))
            return True
        else:
            print("Server not healthy!")
            return False
    except Exception as e:
        print(f"ERROR: Cannot connect to server - {e}")
        return False

# ------------------------------
# PRODUCTION-MATCHING PAYLOADS
# ------------------------------

# 1. SEARCH BAR (Input + Button)
# Structure: Div container -> Input & Button children
search_payload = {
    "dom": {
        "tag": "div",
        "attributes": {"class": "search-container"},
        "children": [
            {"tag": "input", "attributes": {"type": "text", "placeholder": "Search..."}, "children": []},
            {"tag": "button", "attributes": {"type": "submit"}, "children": [
                {"tag": "i", "attributes": {"class": "fa fa-search"}, "children": []}
            ]}
        ]
    }
}

# 2. HEADER (Nav + Logo)
# Structure: Header container -> Logo Div & Nav children
header_payload = {
    "dom": {
        "tag": "header",
        "attributes": {"class": "main-header"},
        "children": [
            {"tag": "div", "attributes": {"class": "logo"}, "children": []},
            {"tag": "nav", "attributes": {}, "children": [
                {"tag": "ul", "children": [
                    {"tag": "li", "children": [{"tag": "a", "attributes": {"href": "/"}, "children": []}]},
                    {"tag": "li", "children": [{"tag": "a", "attributes": {"href": "/about"}, "children": []}]}
                ]}
            ]}
        ]
    }
}

# 3. FOOTER (Footer tag + Links)
# Structure: Footer container -> Copyright & Social Links
footer_payload = {
    "dom": {
        "tag": "footer",
        "attributes": {"class": "site-footer"},
        "children": [
            {"tag": "div", "attributes": {"class": "copyright"}, "children": []},
            {"tag": "div", "attributes": {"class": "social-links"}, "children": [
                {"tag": "a", "attributes": {"href": "#"}, "children": []},
                {"tag": "a", "attributes": {"href": "#"}, "children": []}
            ]}
        ]
    }
}

# 4. CARD (Div -> Image + Content)
# Structure: Div wrapper -> Img & Body Div
card_payload = {
    "dom": {
        "tag": "div",
        "attributes": {"class": "card"},
        "children": [
            {"tag": "img", "attributes": {"src": "image.jpg", "class": "card-img-top"}, "children": []},
            {"tag": "div", "attributes": {"class": "card-body"}, "children": [
                {"tag": "h5", "attributes": {"class": "card-title"}, "children": []},
                {"tag": "p", "attributes": {"class": "card-text"}, "children": []},
                {"tag": "a", "attributes": {"class": "btn btn-primary"}, "children": []}
            ]}
        ]
    }
}

# 5. TABLE (Table -> Thead/Tbody)
# Structure: Table container -> Head & Body
table_payload = {
    "dom": {
        "tag": "table",
        "attributes": {"class": "data-table"},
        "children": [
            {"tag": "thead", "children": [
                {"tag": "tr", "children": [
                    {"tag": "th", "children": []}, {"tag": "th", "children": []}
                ]}
            ]},
            {"tag": "tbody", "children": [
                {"tag": "tr", "children": [
                    {"tag": "td", "children": []}, {"tag": "td", "children": []}
                ]}
            ]}
        ]
    }
}

def test(payload, description):
    print(f"\n--- Testing: {description} ---")
    try:
        start = time.time()
        r = requests.post(URL, json=payload, timeout=10)
        latency = (time.time() - start) * 1000
        
        if r.status_code == 200:
            res = r.json()
            label = res.get("predicted_label", "Unknown")
            conf = res.get("confidence", 0.0)
            
            # Check if prediction roughly matches description string
            # e.g. "Search Bar" description contains "Search"
            is_match = label.lower() in description.lower().replace(" ", "")
            icon = "✅" if is_match else "⚠️"
            
            print(f"{icon} Status: 200 OK ({latency:.1f}ms)")
            print(f"🎯 Prediction: {label}")
            print(f"📊 Confidence: {conf:.2%}")
        else:
            print(f"❌ Error {r.status_code}: {r.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" NEW DATASET VALIDATION SUITE (PROD STRUCTURE)")
    print("="*60)
    
    if test_ping():
        test(search_payload, "SearchBar")
        test(header_payload, "Header")
        test(footer_payload, "Footer")
        test(card_payload, "Card")
        test(table_payload, "Table")
        
    print("\n" + "="*60)
    print(" SUITE COMPLETE ")
    print("="*60)