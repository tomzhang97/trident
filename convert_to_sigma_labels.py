#!/usr/bin/env python3
"""
Convert to Σ(p,f) labels.
UPDATES:
- Safe reading of support_label.
- Full text usage.
- Strict numeric proximity iteration.
"""

import json
import re
import sys
from pathlib import Path

DEFAULT_KEEP_TYPES = {"ENTITY", "NUMERIC", "BRIDGE_HOP1"}

BRIDGE_CUES = [
    r"guest appearance[s]?\b", r"featur(?:es|ing)\b", r"with (?:guest|appearances?)\b",
    r"appears? (?:on|in)\b", r"includes? (?:guest|appearances?)\b",
    r"is (?:a|an|the)\b", r"was (?:a|an|the)\b", r"born\b", r"released\b",
    r"founded\b", r"located in\b", r"capital of\b", r"directed by\b",
    r"written by\b", r"starr?ing\b",
]

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()

def _find_span(text: str, needle: str):
    n = _norm(needle)
    t = _norm(text)
    i = t.find(n)
    return None if i < 0 else (i, i + len(n))

def bridge_sigma_ok(passage: str, title: str, entity1: str, bridge_entity: str) -> bool:
    if not passage: return False
    t = _norm(passage)
    ttl = _norm(title or "")

    e1_in_title = (entity1 and _norm(entity1) in ttl)
    e1_span = _find_span(passage, entity1) if entity1 else None
    about_e1 = e1_in_title or (e1_span is not None and e1_span[0] < 250)
    
    if not about_e1: return False

    be_span = _find_span(passage, bridge_entity) if bridge_entity else None
    if be_span is None: return False

    left = max(0, be_span[0] - 160)
    right = min(len(t), be_span[1] + 160)
    window = t[left:right]

    for cue in BRIDGE_CUES:
        if re.search(cue, window): return True

    if e1_span is not None:
        dist = abs(be_span[0] - e1_span[0])
        if dist <= 120: return True

    return False

def check_entity_sigma(passage, supporting_text, mention, lexical_match):
    if not lexical_match: return False
    text = (supporting_text if supporting_text else passage).lower()
    mention = mention.lower()
    if mention not in text: return False
    if re.search(rf"\b{re.escape(mention)}\s*[\(,]", text): return True
    if re.search(rf"\b{re.escape(mention)}\s+is\s+(a|an|the)\s+\w+", text): return True
    if re.search(rf"\b{re.escape(mention)}('s|s')\s+\w+", text): return True
    if text.count(mention) == 1 and text.count(',') < 5: return True
    return False

def check_numeric_sigma(passage, supporting_text, value, unit, entity, attribute, lexical_match):
    if not lexical_match: return False
    text = (supporting_text if supporting_text else passage).lower()
    val_str = _norm(str(value))
    esc_val = re.escape(val_str)
    pattern = rf"(?<![\w.]){esc_val}(?![\w.])"
    
    matches = list(re.finditer(pattern, text))
    if not matches: return False
    
    cues = ["born", "died", "founded", "established", "released", "published",
            "population", "residents", "length", "distance", "height", "area", 
            "size", "duration", "aged", "age", "year"]
    
    for m in matches:
        start, end = m.span()
        if entity:
            idx_e = text.find(entity.lower())
            if idx_e != -1 and abs(idx_e - start) < 150: return True
        
        window_start = max(0, start - 60)
        window_end = min(len(text), end + 60)
        context = text[window_start:window_end]
        if any(cue in context for cue in cues): return True
            
    return False

def compute_sigma_label(item):
    meta = item.get("metadata", {})
    ft = meta.get("facet_type")
    tpl = meta.get("facet_template", {})
    
    # Safely get support_label (handling both raw and pre-processed files)
    support_label = int(item.get("support_label", item.get("label", 0)))
    lexical_match = meta.get("lexical_match")
    
    passage = meta.get("passage_full") or meta.get("passage_preview", "")
    sup_text = meta.get("supporting_text", "")
    title = meta.get("title", "")
    
    if ft == "ENTITY":
        mention = tpl.get("mention", "")
        schema_ok = check_entity_sigma(passage, sup_text, mention, lexical_match)
        return 1 if (support_label == 1 and schema_ok) else 0
        
    elif ft == "NUMERIC":
        schema_ok = check_numeric_sigma(
            passage, sup_text,
            tpl.get("value", ""), tpl.get("unit", ""),
            tpl.get("entity", ""), tpl.get("attribute", ""),
            lexical_match
        )
        return 1 if (support_label == 1 and schema_ok) else 0

    elif ft == "BRIDGE_HOP1":
        entity1 = tpl.get("entity1", "")
        bridge = tpl.get("bridge_entity", "")
        schema_ok = bridge_sigma_ok(passage, title, entity1, bridge)
        return 1 if (support_label == 1 and schema_ok) else 0

    return 0

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_to_sigma_labels.py input.jsonl output.jsonl [TYPES]")
        sys.exit(1)
        
    in_path, out_path = sys.argv[1], sys.argv[2]
    keep_types = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else DEFAULT_KEEP_TYPES

    print(f"Converting {in_path} -> {out_path}")
    print(f"Keeping Facet Types: {keep_types}")
    
    kept_count = 0
    with open(in_path) as f, open(out_path, "w") as g:
        for line in f:
            r = json.loads(line)
            ft = r["metadata"].get("facet_type", "UNKNOWN")
            
            if ft not in keep_types: continue
                
            # Set support_label explicitly before computation
            r["support_label"] = int(r.get("label", 0))
            r["metadata"]["label_semantics"] = "sigma"
            r["label"] = compute_sigma_label(r)
            
            g.write(json.dumps(r) + "\n")
            kept_count += 1
            
    print(f"Done. Kept {kept_count} samples.")

if __name__ == "__main__":
    main()