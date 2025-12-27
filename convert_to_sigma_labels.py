#!/usr/bin/env python3
"""
Convert to Σ(p,f) labels.
UPDATES:
- Canonical-aware filtering (matches BRIDGE_HOP1 if BRIDGE_HOP is requested).
- Robust Sigma logic (0.4 overlap threshold for better 2Wiki compatibility).
- BRIDGE_HOP: title tokens checked for BOTH entities (symmetric handling).
- RELATION: title included in exact match haystack (not only token overlap).
"""

import json
import re
import sys
from pathlib import Path

DEFAULT_KEEP_TYPES = {
    "ENTITY", "RELATION", "TEMPORAL", "NUMERIC", 
    "BRIDGE_HOP1", "BRIDGE_HOP2", "BRIDGE_HOP", "BRIDGE", 
    "COMPARISON", "CAUSAL", "PROCEDURAL"
}

# --- TEXT HELPERS ---

WORD_TO_NUM = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'first': 1, 'second': 2, 'third': 3
}

def _norm(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").lower()).strip()
    return s.replace(",", "") 

def _get_tokens(text: str) -> set:
    tokens = re.split(r"[^a-z0-9]+", _norm(text))
    stops = {"the", "a", "an", "of", "in", "on", "at", "to", "by", "and"}
    return {t for t in tokens if len(t) > 1 and t not in stops}

def _extract_numbers(text: str) -> list:
    nums = []
    for w in re.findall(r"\b[a-z]+\b", _norm(text)):
        if w in WORD_TO_NUM:
            nums.append(float(WORD_TO_NUM[w]))
    
    clean_text = text.replace(",", "")
    for match in re.findall(r"-?\d+\.?\d*", clean_text):
        if match not in {".", "-", "-."}:
            try:
                nums.append(float(match))
            except ValueError:
                pass
    return nums

# --- SIGMA CHECKERS ---

def check_entity_sigma(passage, sup_text, mention, lex):
    if lex is False: return False, "LEX_GATE_FAIL"
    
    mention = (mention or "").strip()
    if not mention: return False, "EMPTY_MENTION"

    text = (sup_text if sup_text else passage).lower()
    mention_lower = mention.lower()
    
    if mention_lower not in text: return False, "EXACT_MISSING"
    
    if re.search(rf"\b{re.escape(mention_lower)}\s*[\(,]", text): return True, "OK_APPOSITION"
    if re.search(rf"\b{re.escape(mention_lower)}\s+is\s+(a|an|the)\s+\w+", text): return True, "OK_DEF"
    if re.search(rf"\b{re.escape(mention_lower)}('s|s')\s+\w+", text): return True, "OK_POSSESSIVE"
    if text.count(mention_lower) == 1 and text.count(',') < 5: return True, "OK_SINGLE_MENTION"
    
    return False, "HEURISTIC_FAIL"

def check_numeric_sigma(passage, sup_text, value, entity, lex):
    # No lex gate
    text = _norm(sup_text if sup_text else passage)
    target_nums = _extract_numbers(str(value))
    if not target_nums: return False, "NO_TARGET_VAL"
    
    text_nums = _extract_numbers(text)
    
    found_val = False
    for t_val in target_nums:
        tol = max(1e-9, 0.01 * abs(t_val))
        if any(abs(n - t_val) < tol for n in text_nums):
            found_val = True
            break
            
    if not found_val: return False, "VALUE_MISMATCH"
    
    if entity and _norm(entity) in text: return True, "OK_ENTITY_BIND"
    
    cues = ["born", "died", "founded", "population", "length", "height", "area", "year", "age", "size", "total"]
    if any(c in text for c in cues): return True, "OK_CUE"
    
    return False, "CONTEXT_MISSING"

def check_relation_sigma(passage, sup_text, s, o, pred, lex, title="", overlap_thresh=0.4):
    """Check relation sigma with title included in exact match haystack.

    Uses 0.4 overlap threshold (lowered from 0.5) for better 2Wiki compatibility.
    Title is now included in both exact match and token overlap checks.
    """
    # No lex gate
    s = (s or "").strip()
    o = (o or "").strip()
    if not s or not o: return False, "EMPTY_SUBJ_OBJ"

    base_text = _norm(sup_text if sup_text else passage)
    title_norm = _norm(title)
    # Include title in exact match haystack (not only token overlap)
    text_with_title = base_text + " " + title_norm

    s_norm = _norm(s); o_norm = _norm(o)

    # CHANGED: Exact match now checks text_with_title (includes title)
    s_found = s_norm in text_with_title
    o_found = o_norm in text_with_title

    if not (s_found and o_found):
        s_toks = _get_tokens(s)
        o_toks = _get_tokens(o)
        txt_toks = _get_tokens(text_with_title)

        # 0.4 Overlap Threshold (lowered from 0.5)
        s_hit = len(s_toks.intersection(txt_toks)) / max(1, len(s_toks)) >= overlap_thresh
        o_hit = len(o_toks.intersection(txt_toks)) / max(1, len(o_toks)) >= overlap_thresh

        if not (s_hit and o_hit): return False, "ENTITIES_MISSING"

    return True, "OK_GOLD_PROXIMITY"

def check_bridge_sigma_ok(passage, title, e1, e2, overlap_thresh=0.4):
    """Check bridge hop sigma with title tokens for BOTH entities.

    Uses 0.4 overlap threshold (lowered from 0.5) for better 2Wiki compatibility.
    Both e1 and e2 can match in title OR text for symmetric handling.
    """
    if not passage: return False, "NO_TEXT"

    e1 = (e1 or "").strip()
    e2 = (e2 or "").strip()
    if not e1 or not e2: return False, "EMPTY_BRIDGE_ENTS"

    t = _norm(passage)
    ttl = _norm(title or "")

    txt_toks = _get_tokens(t)
    ttl_toks = _get_tokens(ttl)

    e1_toks = _get_tokens(e1)
    e2_toks = _get_tokens(e2)

    if not e1_toks or not e2_toks: return False, "EMPTY_ENTITY_TOKS"

    # e1: can match in title OR text (unchanged logic, but with new threshold)
    e1_hit = (len(e1_toks.intersection(ttl_toks)) / len(e1_toks) >= overlap_thresh) or \
             (len(e1_toks.intersection(txt_toks)) / len(e1_toks) >= overlap_thresh)

    if not e1_hit: return False, "E1_MISSING"

    # e2: NOW can also match in title OR text (symmetric with e1)
    e2_hit = (len(e2_toks.intersection(ttl_toks)) / len(e2_toks) >= overlap_thresh) or \
             (len(e2_toks.intersection(txt_toks)) / len(e2_toks) >= overlap_thresh)
    if not e2_hit: return False, "E2_MISSING"

    return True, "OK_TOKEN_OVERLAP"

def check_temporal_sigma(passage, sup_text, time_str, lex):
    return True, "OK"

def check_comparison_sigma(passage, sup_text, e1, e2, attr, lex):
    # No lex gate
    e1 = (e1 or "").strip()
    e2 = (e2 or "").strip()
    if not e1 and not e2: return False, "EMPTY_ENTITIES"

    text = _norm(sup_text if sup_text else passage)
    cues = ["more", "less", "better", "worse", "higher", "lower", "than", "older", "younger"]
    if attr: cues.append(_norm(attr))
    
    has_cue = any(c in text for c in cues)
    e1_in = _norm(e1) in text if e1 else False
    e2_in = _norm(e2) in text if e2 else False
    
    if has_cue and (e1_in or e2_in): return True, "OK_CUE_ENTITY"
    return False, "NO_EVIDENCE"

def check_causal_sigma(passage, sup_text):
    text = _norm(sup_text if sup_text else passage)
    cues = [r"\bbecause\b", r"\bdue to\b", r"\bcaused\b", r"\bled to\b"]
    if any(re.search(c, text) for c in cues): return True, "OK_CUE"
    return False, "NO_CUE"

def compute_sigma_label(item):
    meta = item.get("metadata", {})
    ft = meta.get("facet_type") or "UNKNOWN"
    tpl = meta.get("facet_template", {})
    
    support_label = int(item.get("support_label", item.get("label", 0)))
    if support_label != 1: return 0, "NO_GOLD_SUPPORT"

    lexical_match = meta.get("lexical_match")
    passage = meta.get("passage_full") or meta.get("passage_preview", "")
    sup_text = meta.get("supporting_text", "")
    title = meta.get("title", "")
    
    if ft == "ENTITY":
        return check_entity_sigma(passage, sup_text, tpl.get("mention",""), lexical_match)
    
    if ft == "NUMERIC":
        return check_numeric_sigma(passage, sup_text, tpl.get("value",""), tpl.get("entity",""), lexical_match)
    
    if ft == "RELATION":
        return check_relation_sigma(passage, sup_text, tpl.get("subject",""), tpl.get("object",""), tpl.get("predicate",""), lexical_match, title)
    
    if ft == "TEMPORAL":
        return check_temporal_sigma(passage, sup_text, tpl.get("time",""), lexical_match)
    
    if ft == "COMPARISON":
        return check_comparison_sigma(passage, sup_text, tpl.get("entity1",""), tpl.get("entity2",""), tpl.get("attribute",""), lexical_match)
    
    if ft.startswith("BRIDGE_HOP") or ft == "BRIDGE":
        e1 = tpl.get("entity1", "")
        be = tpl.get("bridge_entity") or tpl.get("entity2") or ""
        return check_bridge_sigma_ok(passage, title, e1, be)
        
    if ft == "CAUSAL":
        return check_causal_sigma(passage, sup_text)
        
    return 0, "UNSUPPORTED_TYPE"

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_to_sigma_labels.py input.jsonl output.jsonl [TYPES]")
        sys.exit(1)
    
    in_path, out_path = sys.argv[1], sys.argv[2]
    # Parse CLI types
    cli_keep = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else DEFAULT_KEEP_TYPES
    
    print(f"Converting {in_path} -> {out_path}")
    print(f"Keeping Facet Types: {cli_keep}")
    
    kept_count = 0
    with open(in_path) as f, open(out_path, "w") as g:
        for line in f:
            r = json.loads(line)
            ft_raw = r.get("metadata", {}).get("facet_type") or "UNKNOWN"
            
            # SAFE FILTER: Check Raw OR Canonical
            ft_can = "BRIDGE_HOP" if ft_raw.startswith("BRIDGE_HOP") else ft_raw
            
            if (ft_raw not in cli_keep) and (ft_can not in cli_keep):
                continue
                
            r["support_label"] = int(r.get("support_label", r.get("label", 0)))
            
            is_sigma, reason = compute_sigma_label(r)
            r["label"] = int(is_sigma)
            r["metadata"]["label_semantics"] = "sigma"
            r["metadata"]["sigma_fail_reason"] = reason 
            r["metadata"]["facet_type_canonical"] = ft_can
            
            g.write(json.dumps(r) + "\n")
            kept_count += 1
            
    print(f"Done. Kept {kept_count} samples.")

if __name__ == "__main__":
    main()