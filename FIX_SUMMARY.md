# Label-Hypothesis Mismatch Fix Summary

## ✅ Completed Fixes

### Blocker A: Diagnostics Wrong for Σ Labels

**Problem:** After implementing Σ(p,f) schema-bound labels, diagnostics were incorrectly flagging `lex=True + label=0` cases as errors.

**Root Cause:** Diagnostics assumed labels should match `lexical_match`, but Σ labels represent schema-bound sufficiency, not lexical containment.

**Fix (commit f94ea03):**
1. Added `label_semantics = "sigma"` tag to converted data in `convert_to_sigma_labels.py`
2. Updated `diagnose_calibration_detailed.py` to detect and handle three label semantics:
   - `"sigma"`: Σ(p,f) schema-bound labels (NEW - correct for Safe-Cover)
   - `"entail"`: Simple entailment labels (convert_to_entail_labels.py - debugging hack)
   - `"support"`: Hotpot supporting-fact labels (original - mismatched)

3. For Σ labels, diagnostics now correctly interpret:
   - ✅ `lex=True + label=0` → **EXPECTED** (critical negatives: phrase exists but schema check failed)
   - ⚠️  `lex=False + label=1` → **BUG** (phrase doesn't exist but labeled positive)

4. Added guidance for TPR=0 issue (hypothesis/verifier mismatch)

**Files Modified:**
- `convert_to_sigma_labels.py:382` - Add label_semantics tag
- `diagnose_calibration_detailed.py:40-91` - Detect and handle Σ semantics
- `diagnose_calibration_detailed.py:155-175` - Update high-confidence negative interpretation
- `diagnose_calibration_detailed.py:301-325` - Add Σ-specific summary guidance

---

### Blocker B: Hypotheses Ask Lexical Questions Instead of Σ

**Problem:** Calibrator selects nothing (TPR=0 at all alphas). Score distributions don't separate Σ positives from negatives.

**Root Cause:** Hypothesis templates asked lexical questions ("does phrase exist?") while labels represented Σ (schema-bound sufficiency). The NLI model scored hypotheses, not Σ.

**Fix (commit fc55ee4):**

Updated hypothesis templates in `trident/facets.py` to ask Σ questions:

#### ENTITY (line 196)
**Before:**
```python
return _ensure_sentence(f'The passage contains the exact phrase "{mention}".')
```

**After:**
```python
# Σ(p,f) hypothesis: schema-bound sufficiency, not mere lexical containment
return _ensure_sentence(f'The passage identifies "{mention}" unambiguously (not merely a mention).')
```

**Rationale:** Σ for ENTITY means unambiguous identification (apposition, definition, role), not mere phrase containment.

#### NUMERIC (lines 229-238)
**Before:**
```python
if entity and attr and value:
    v = f"{value} {unit}".strip()
    return _ensure_sentence(f"{entity} has {attr} of {v}")
if value:
    v = f"{value} {unit}".strip()
    return _ensure_sentence(f"The passage states the value {v}")
```

**After:**
```python
# Σ(p,f) hypothesis: value bound to property, not mere containment
if entity and attr and value:
    v = f"{value} {unit}".strip()
    return _ensure_sentence(f"The passage states that {entity}'s {attr} is {v}")
if entity and value:
    v = f"{value} {unit}".strip()
    return _ensure_sentence(f"The passage states that {entity} has the value {v}")
if value:
    v = f"{value} {unit}".strip()
    return _ensure_sentence(f"The passage binds the value {v} to a specific property")
```

**Rationale:** Σ for NUMERIC means value bound to property (with property cues), not just value containment.

**Files Modified:**
- `trident/facets.py:191-196` - ENTITY hypothesis → Σ question
- `trident/facets.py:223-238` - NUMERIC hypothesis → Σ question

---

## 📊 Impact

### Before Fixes
- **Diagnostics:** Incorrectly flagging 360+ critical negatives as errors
- **Hypotheses:** Asking "does phrase exist?" (lexical)
- **Labels:** Answering "does passage satisfy Σ(p,f)?" (schema-bound)
- **Result:** Hypothesis/verifier mismatch → TPR=0, calibrator selects nothing

### After Fixes
- **Diagnostics:** ✅ Correctly interpret Σ semantics, flag only real bugs
- **Hypotheses:** ✅ Asking "does passage satisfy Σ(p,f)?" (schema-bound)
- **Labels:** ✅ Answering "does passage satisfy Σ(p,f)?" (schema-bound)
- **Expected Result:** Hypothesis/verifier alignment → TPR > 0, calibrator works

---

## 🔧 Next Steps

### 1. Re-extract Calibration Data
The new hypothesis templates are now in `trident/facets.py`. Re-extract calibration data to use them:

```bash
# Extract from MuSiQue dev (expand to 500+ samples)
python extract_calibration_data.py \
  --dataset musique \
  --split dev \
  --output_path calibration_musique_raw.jsonl \
  --max_samples 500
```

**Why:** Old data has lexical hypotheses ("contains exact phrase"). Need new data with Σ hypotheses ("identifies unambiguously").

### 2. Convert to Σ Labels
```bash
# Convert Hotpot/MuSiQue supporting-fact labels to Σ(p,f) labels
python convert_to_sigma_labels.py \
  calibration_musique_raw.jsonl \
  calibration_musique_sigma.jsonl
```

**This will:**
- Add schema-specific checks (ENTITY: apposition/definition, NUMERIC: property cues)
- Set `label = 1` iff `support_label==1 AND schema_ok==True`
- Create critical negatives: `lex=True but label=0` (phrase exists but schema fails)
- Tag data with `label_semantics = "sigma"`

### 3. Diagnose Quality
```bash
python diagnose_calibration_detailed.py calibration_musique_sigma.jsonl
```

**Expected output:**
```
✅ Detected Σ(p,f) schema-bound labels
   For Σ labels, lex=True + label=0 is EXPECTED:
   - These are the CRITICAL NEGATIVES for calibration!

📊 Found:
   lex=False + label=1: 0 (BUGS - phrase doesn't exist!)
   lex=True + label=0:  X (EXPECTED - schema check failed, CRITICAL NEGATIVES! ✅)
```

**Check for:**
- ✅ Zero `lex=False + label=1` bugs
- ✅ Many `lex=True + label=0` critical negatives (good!)
- ⚠️  ENTITY has 20+ positives (needed for Mondrian calibration)
- ⚠️  NUMERIC has 20+ positives (may need more data or relaxed schema_ok)

### 4. Train Calibrator
```bash
# ENTITY only (safest, if NUMERIC has too few positives)
python train_calibration.py \
  --data_path calibration_musique_sigma.jsonl \
  --output_path calibrator_entity.json \
  --use_mondrian \
  --facet_types ENTITY \
  --filter_lexical_false \
  --use_entail_prob

# Or ENTITY+NUMERIC (if both have 20+ positives)
python train_calibration.py \
  --data_path calibration_musique_sigma.jsonl \
  --output_path calibrator.json \
  --use_mondrian \
  --filter_lexical_false \
  --use_entail_prob
```

**Critical flags:**
- `--filter_lexical_false`: Remove gate-dominated zeros from training
- `--use_entail_prob`: Use `probs.entail` instead of derived score
- `--use_mondrian`: Stratified calibration by facet type

### 5. Check Calibrator Quality
Look for in the training output:

**Good signs:**
```
TPR at alpha=0.1: 0.XX (> 0!)
FPR at alpha=0.1: 0.0X (< 0.1)
ENTITY: X positives, Y negatives (both > 20)
```

**Bad signs (hypothesis still mismatched):**
```
TPR at alpha=0.1: 0.0  ← Still selecting nothing!
Selection rate: 0.0
```

If TPR=0 persists:
- Check hypothesis examples in JSONL (should say "identifies unambiguously", not "contains phrase")
- Check if NLI model scores align with Σ labels (plot score distributions by label)
- May need to strengthen schema_ok checks or adjust NLI prompt

### 6. End-to-End Test
```bash
# Run on a few test questions
python -m trident.pipeline \
  --query "Which magazine was started first, Arthur's Magazine or First for Women?" \
  --config configs/safe_cover.json \
  --calibrator_path calibrator.json \
  --alpha 0.1
```

**Expected:**
- Lexical gate filters ENTITY facets where phrase doesn't exist
- NLI scores Σ hypotheses ("identifies unambiguously")
- Calibrator maps scores to p-values using Σ-trained mapping
- Safe-Cover selects facets with sufficient Σ support

---

## 📝 Technical Details

### What is Σ(p,f)?

Σ(p,f) = "passage p sufficiently supports facet f under schema constraints"

**Not the same as:**
- ❌ Lexical containment: "phrase/value exists in passage"
- ❌ NLI entailment: "passage entails hypothesis" (depends on hypothesis!)
- ❌ Hotpot supporting fact: "is this a supporting fact for the question?"

**For TRIDENT:**
- **ENTITY Σ**: Passage identifies entity unambiguously (apposition, definition, role)
  - ✅ "Leonid Levin (born 1948, Soviet-American computer scientist)"
  - ✅ "Arthur's Magazine was an American literary periodical"
  - ❌ "...Levin's algorithm..." (mere mention, ambiguous)
  - ❌ "Arthur's work..." (wrong binding)

- **NUMERIC Σ**: Passage binds value to property (property cues near value)
  - ✅ "Bathurst 12 Hour: a 12-hour endurance race" (property: "duration")
  - ✅ "Founded in 1844" (property: "founding year")
  - ❌ "Bathurst 12 Hour" (name contains 12, not a property)
  - ❌ "12 teams participated" (wrong binding)

### Label Semantics Comparison

| Semantics | Question | Use Case | Mismatches |
|-----------|----------|----------|------------|
| `support` | Is this a Hotpot supporting fact? | Original Hotpot labels | Many `lex=True + label=0` (phrase exists but not supporting) |
| `entail` | Does passage entail hypothesis? | Debugging hack (convert_to_entail_labels.py) | Few mismatches, but destroys Σ semantics |
| `sigma` | Does passage satisfy Σ(p,f)? | Correct for Safe-Cover calibration | `lex=True + label=0` are critical negatives (expected!) |

### Why Critical Negatives Matter

For calibration to work, we need:
1. **Positives** where NLI scores HIGH (passage satisfies Σ)
2. **Hard negatives** where NLI scores HIGH but label=0 (lexical match but schema fails)

Without (2), calibrator learns "flat" mapping because all negatives have score=0 (gate-dominated).

The `lex=True + label=0` cases ARE the hard negatives:
- Lexical gate doesn't fire (phrase exists)
- NLI must score the Σ question properly
- If NLI scores high but label=0 → calibrator learns to down-weight these scores

This is why Σ labels are critical: they create hard negatives that lexical_match labels destroy.

---

## 🔗 Files Modified

1. `trident/facets.py` - Hypothesis templates (ENTITY, NUMERIC)
2. `convert_to_sigma_labels.py` - Add label_semantics tag
3. `diagnose_calibration_detailed.py` - Detect and handle Σ semantics
4. `test_sigma_hypotheses.py` - Test suite for Σ hypotheses

---

## 🎯 Success Criteria

After completing next steps, you should see:

✅ **Diagnostics:** No `lex=False + label=1` bugs, many `lex=True + label=0` critical negatives
✅ **Hypotheses:** Ask Σ questions ("identifies unambiguously", "states that X's property is Y")
✅ **Training:** TPR > 0 at reasonable alphas (0.1, 0.2, etc.)
✅ **Calibrator:** Non-flat mapping, good separation between pos/neg scores
✅ **End-to-end:** Selects facets with true Σ support, rejects hard negatives

---

## 📚 References

- Σ(p,f) schema-bound labeling: `convert_to_sigma_labels.py`
- Hypothesis templates: `trident/facets.py:188-294`
- Diagnostics: `diagnose_calibration_detailed.py`
- Training: `train_calibration.py` (with FPR fix, filtering, entail probs)
- Previous fixes: commits d72ab97, 2f2dcdb, 443cd40, 128f589, c75736f

---

**Generated:** 2025-12-25 (after fixing Blockers A and B)
