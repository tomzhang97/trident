# KET-RAG Baseline Fixes for Improved EM/F1 Scores

## Problem Diagnosis

The KET-RAG framework was producing poor results despite having the correct information in the context. Analysis revealed:

1. **Empty `[]` Prefix**: Context started with literal `"[]"` which signals to the LLM that primary passages are empty
2. **Poor Prompt Structure**: Original prompt didn't emphasize the importance of the Entities table
3. **"Woodson" Noise**: Irrelevant mentions of "Woodson, Arkansas" confused the retrieval

### Example Analysis
For question: "Were Scott Derrickson and Ed Wood of the same nationality?"

- ✅ **Answer present**: "American" mentioned 9 times in context
- ✅ **Entities correct**: Both described as "American" in entity descriptions
- ❌ **Format confusing**: Context started with `[]` followed by structured data
- ❌ **Prompt weak**: Didn't guide LLM to use entity descriptions

## Fixes Applied

### Fix 1: Remove Empty `[]` Prefix
**File**: `KET-RAG/indexing_sket/create_context.py`

Added cleanup logic in `generate_text_unit_context()` function (lines 338-341):
```python
# Fix: Remove empty list representation "[]" that sometimes appears
# when context_chunks is empty or improperly formatted
if isinstance(graph_context, str) and graph_context.strip().startswith("[]"):
    graph_context = graph_context.strip()[2:].lstrip()
```

This removes the confusing `[]` prefix that was causing the LLM to think there were no relevant passages.

### Fix 2: Enhanced Prompt with Entity Emphasis
**Files**:
- `KET-RAG/indexing_sket/util_v1.py` (LOCAL_SEARCH_EXACT_SYSTEM_PROMPT)
- `baselines/prompt_utils.py` (KETRAG_SYSTEM_PROMPT)

Added new "Important Instructions" section to guide the LLM:
```
---Important Instructions---

1. Pay special attention to the "Entities" table - entity descriptions often contain key attributes needed to answer the question
2. Use the "Relationships" table to understand connections between entities
3. Cross-reference information across all sections (Entities, Relationships, Sources, Text sources) to find the answer
4. If multiple sources contain the same information, that increases confidence in the answer
```

This helps the LLM:
- Focus on entity descriptions (which often contain the answer)
- Use the structured knowledge graph data effectively
- Cross-reference information for higher confidence

## Expected Impact

### Before Fixes:
- LLM sees `[]` at start → thinks no passages retrieved
- Doesn't know to check entity descriptions
- Gets confused by irrelevant mentions
- Lower EM/F1 scores

### After Fixes:
- Clean context without `[]` prefix
- LLM explicitly told to check entity descriptions
- Better utilization of structured knowledge graph
- **Expected: Significant improvement in EM/F1 scores**

## How to Re-run Experiments

After these fixes, regenerate the contexts and re-run evaluation:

```bash
# Step 1: Regenerate contexts with the fix
cd KET-RAG
poetry run python indexing_sket/create_context.py <root_path> keyword 0.5

# Step 2: Re-run evaluation
cd ..
python experiments/eval_full_baselines.py \
  --model gpt-4o-mini \
  --dataset data/hotpotqa_dev.json \
  --ketrag_context_file KET-RAG/<root_path>/output/<name>-keyword-0.5.json \
  --output results/ketrag_fixed.json
```

## Test the Fix

You can test the fix immediately on the debug example:

```bash
python experiments/debug_ketrag_prompts.py
```

The output should now show clean context without the `[]` prefix.

## Additional Optimizations (Future Work)

1. **Filter "Woodson" mentions**: Add preprocessing to remove references to "Woodson, Arkansas" when the question is about "Wood" and "Derrickson"
2. **Adaptive entity emphasis**: Dynamically adjust prompt based on question type (attribute query vs. relationship query)
3. **Confidence scoring**: Add explicit confidence indicators when multiple sources agree

## Related Files Modified

1. `KET-RAG/indexing_sket/create_context.py` - Context generation fix
2. `KET-RAG/indexing_sket/util_v1.py` - Prompt enhancement
3. `baselines/prompt_utils.py` - Matching prompt enhancement for consistency
4. `experiments/test_ketrag_context_variations.py` - Analysis tool (already created)
5. `experiments/KETRAG_FIX_SUMMARY.md` - This file
