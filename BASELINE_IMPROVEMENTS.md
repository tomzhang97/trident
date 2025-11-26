# Baseline Improvements Summary

This document summarizes the improvements made to GraphRAG and Self-RAG baselines to make them valid for paper-level comparison with TRIDENT.

## Changes Made

### 1. GraphRAG System Improvements

#### Issue: Graph expansion ignored `max_hops` parameter
**Before:**
```python
def expand_subgraph(self, seed_node_ids, max_hops=2, max_nodes=64):
    subgraph = {}
    for seed_id in seed_node_ids:
        # Only looked at direct edges, ignoring max_hops
        for s, r, o in self.edges:
            if s == seed_id or o == seed_id:
                triples.append((s, r, o))
```

**After:**
```python
def expand_subgraph(self, seed_node_ids, max_hops=2, max_nodes=64):
    # Build adjacency list
    adj = defaultdict(list)
    for s, r, o in self.edges:
        adj[s].append((s, r, o))
        adj[o].append((s, r, o))

    # BFS-based expansion respecting max_hops
    queue = deque([(seed, 0) for seed in seed_node_ids])
    while queue and len(visited) < max_nodes:
        node_id, hop = queue.popleft()
        if hop > max_hops:
            continue
        # ... expand neighbors at hop+1
```

✅ **Impact**: GraphRAG now genuinely performs multi-hop graph reasoning.

#### Issue: Seed IDs not validated
**Before:**
```python
seed_output = self._call_llm(seed_prompt, qstats, max_new_tokens=32)
seed_ids = [s.strip() for s in seed_output.split(",")]
# If LLM outputs "1, 2" instead of "n1, n2", silently fails
```

**After:**
```python
seed_ids = [s.strip() for s in seed_output.split(",") if s.strip()]
seed_ids = [s for s in seed_ids if s in graph_index.nodes]

if not seed_ids:
    # Fallback: use top candidates directly
    seed_ids = [c["id"] for c in candidates[:min(self.max_seeds, len(candidates))]]
```

✅ **Impact**: Prevents silent failures from malformed LLM outputs.

---

### 2. Self-RAG System Improvements

#### Issue: Oracle context bypass in retrieval comparison
**Before:**
```python
if context is not None:
    # Use provided context, bypassing retrieval gate entirely
    gate = "PROVIDED_CONTEXT"
    for title, sentences in context:
        docs.append({"text": text, "title": title})
else:
    # Use retrieval gate
    gate_output = self._call_llm(retrieve_prompt, qstats)
```

**After:**
```python
def __init__(self, ..., allow_oracle_context=False):
    self.allow_oracle_context = allow_oracle_context

def answer(self, question, context=None, ...):
    if context is not None and not self.allow_oracle_context:
        raise ValueError(
            "Self-RAG in retrieval mode does not accept oracle context. "
            "Set allow_oracle_context=True to use provided context."
        )

    if context is not None and self.allow_oracle_context:
        # Oracle-context regime (clearly separated)
        gate = "PROVIDED_CONTEXT"
        ...
    else:
        # Retrieval regime
        gate_output = self._call_llm(retrieve_prompt, qstats)
```

✅ **Impact**: Clearly separates retrieval vs oracle-context experiments.

#### Issue: Critic computed but never used
**Before:**
```python
if self.use_critic and docs:
    critic_output = self._call_llm(critic_prompt, qstats)
    critic_label = critic_output.strip().split()[0].upper()
    # Label stored but never acted upon
```

**After:**
```python
if self.use_critic and docs:
    critic_output = self._call_llm(critic_prompt, qstats)
    critic_label = critic_output.strip().split()[0].upper()

    # If critic finds issues, generate fallback answer
    if critic_label in ("CONTRADICTS", "INSUFFICIENT"):
        fallback_prompt = SELF_RAG_ANSWER_PROMPT.format(
            question=question,
            context="(none)",
        )
        raw_answer2 = self._call_llm(fallback_prompt, qstats)
        answer = raw_answer2.strip()
```

✅ **Impact**: Critic now actually affects the output, justifying its token cost.

---

### 3. Harmonized Retrieval Parameters

#### Issue: Different k values across systems
**Before:**
```python
# In config
"baselines": {
    "selfrag_k": 8,
    "graphrag_k": 20,  # Different!
}
```

**After:**
```python
# In config
"baselines": {
    "common_k": 8,
    "selfrag_k": 8,
    "graphrag_k": 8,  # Harmonized
}

# Command-line control
parser.add_argument("--common_k", type=int, default=8,
                    help="Common k for retrieval across all baselines")
```

✅ **Impact**: Fair token and quality comparison across all systems.

---

### 4. Evaluation Script Improvements

#### New command-line arguments:
```python
--common_k 8                         # Shared retrieval k
--selfrag_allow_oracle_context       # Enable oracle-context mode
--selfrag_use_critic                 # Enable critic
```

#### Updated system initialization:
```python
# Self-RAG
systems['self_rag'] = SelfRAGSystem(
    llm=self.instrumented_llm,
    retriever=self.retriever,
    k=self.config.baselines.selfrag_k,
    use_critic=self.config.baselines.selfrag_use_critic,
    allow_oracle_context=self.config.baselines.selfrag_allow_oracle_context
)

# GraphRAG
systems['graphrag'] = GraphRAGSystem(
    llm=self.instrumented_llm,
    retriever=self.retriever,
    k=self.config.baselines.graphrag_k,
    topk_nodes=self.config.baselines.graphrag_topk_nodes,
    max_seeds=self.config.baselines.graphrag_max_seeds,
    max_hops=self.config.baselines.graphrag_max_hops
)
```

---

## Validation Checklist

### For Reviewers

- [x] **Same LLM**: All systems use identical model, temperature, max_new_tokens
- [x] **Same retrieval**: All systems use same retriever, index, and k
- [x] **Same regime**: Can enforce either retrieval-only or oracle-context for all
- [x] **Token tracking**: All LLM calls tracked through InstrumentedLLM
- [x] **Honest naming**: Documentation clearly states simplifications

### Remaining Simplifications (to mention in paper)

**Self-RAG:**
- Single-step critic (no iterative refinement)
- Simple fallback strategy (regenerate without context)
- Should be called "Self-RAG-style baseline" or "Simplified Self-RAG"

**GraphRAG:**
- Per-query graphs (not persistent global KG)
- Simple chain-based edges (not rich entity extraction)
- Should be called "Graph-style RAG baseline" or "Local GraphRAG-lite"

---

## Usage Examples

### Retrieval-only comparison (recommended):
```bash
python eval_compare_baselines.py \
    --systems all \
    --common_k 8 \
    --data_path data/hotpotqa_dev.json \
    --output_dir results/retrieval_comparison
```

### Oracle-context ablation:
```bash
python eval_compare_baselines.py \
    --systems all \
    --common_k 8 \
    --selfrag_allow_oracle_context \
    --data_path data/hotpotqa_dev.json \
    --output_dir results/oracle_comparison
```

### With critic enabled:
```bash
python eval_compare_baselines.py \
    --systems self_rag \
    --common_k 8 \
    --selfrag_use_critic \
    --data_path data/hotpotqa_dev.json \
    --output_dir results/selfrag_critic
```

---

## Files Modified

1. `baselines/graphrag_system.py`
   - Updated `expand_subgraph()` with BFS-based expansion
   - Added seed ID validation with fallback

2. `baselines/self_rag_system.py`
   - Added `allow_oracle_context` parameter
   - Added critic fallback generation
   - Enforced retrieval regime when appropriate

3. `eval_compare_baselines.py`
   - Added `--common_k` argument
   - Added `--selfrag_allow_oracle_context` argument
   - Harmonized config defaults
   - Updated system initialization

4. `configs/baseline_comparison.json`
   - Set `common_k: 8`
   - Set `graphrag_k: 8` (was 20)
   - Added `selfrag_allow_oracle_context: false`
   - Added comments for clarity

5. **New files:**
   - `BASELINES.md`: Comprehensive documentation
   - `BASELINE_IMPROVEMENTS.md`: This file

---

## Impact on Results

### Expected changes:

1. **GraphRAG accuracy may improve slightly**
   - BFS expansion captures more relevant context
   - Seed validation prevents crashes

2. **GraphRAG tokens may increase slightly**
   - More subgraph nodes → more summaries
   - But now correctly reflects graph reasoning cost

3. **Self-RAG tokens may decrease**
   - GraphRAG now uses k=8 instead of k=20
   - Fairer comparison to Self-RAG

4. **Self-RAG with critic tokens increase**
   - Critic + potential fallback generation
   - But accuracy may improve

### No changes expected:
- TRIDENT results (unchanged)
- Relative rankings (unless GraphRAG was previously failing)

---

## Recommendation for Paper

### In the Experimental Setup section:

> **Baselines.** We compare against simplified but valid implementations of Self-RAG (Asai et al., 2023) and GraphRAG (Edge et al., 2024). Our Self-RAG baseline includes retrieval gating and single-step critic verification without iterative refinement. Our GraphRAG baseline builds per-query graphs with BFS-based expansion (max_hops=2) rather than persistent global knowledge graphs. For fair comparison, all systems use the same retriever (Contriever), the same k=8 retrieved documents, and identical LLM configurations (Llama-2-7b, temperature=0).

### In a footnote or appendix:

> We validated our baseline implementations by ensuring: (1) all systems use identical LLM and retrieval configurations, (2) token accounting includes all LLM calls (gates, critics, summaries), and (3) evaluation regimes are consistent (either retrieval-only or oracle-context, applied uniformly). See BASELINES.md for implementation details.

---

## Questions?

For technical details, see:
- `BASELINES.md` - Full documentation
- `baselines/self_rag_system.py` - Implementation
- `baselines/graphrag_system.py` - Implementation
- `eval_compare_baselines.py` - Evaluation script
