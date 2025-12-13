"""Test KET-RAG with different context configurations to isolate the issue."""
import json
from pathlib import Path

# Load the problematic context
context_file = Path("data/ketrag_debug/context_keyword_0.5.json")
with open(context_file) as f:
    data = json.load(f)

original_context = data[0]["context"]
question = "Were Scott Derrickson and Ed Wood of the same nationality?"

print("="*80)
print("ORIGINAL CONTEXT (first 1500 chars):")
print("="*80)
print(original_context[:1500])
print("\n" + "="*80)
print("ANALYSIS:")
print("="*80)

# Count occurrences of key terms
print(f"\n'American' mentions: {original_context.count('American')}")
print(f"'Woodson' mentions: {original_context.count('Woodson')}")
print(f"Empty passages at start: {original_context.startswith('[]')}")

# Check if answer is obvious from entities alone
entities_section = original_context.split("-----Relationships-----")[0]
if "SCOTT DERRICKSON" in entities_section and "American" in entities_section:
    print("\n✓ Scott Derrickson nationality (American) is in entities")
if "ED WOOD" in entities_section and "American" in entities_section:
    print("✓ Ed Wood nationality (American) is in entities")

# Check text chunks
if "-----Text source that may be relevant-----" in original_context:
    chunks_section = original_context.split("-----Text source that may be relevant-----")[1]
    print(f"\nChunks section length: {len(chunks_section)} chars")
    print(f"Chunk count: {chunks_section.count('chunk_')}")

print("\n" + "="*80)
print("PROPOSED FIX:")
print("="*80)
print("""
1. Remove the '[]' prefix from empty graph passages
2. Emphasize the Entities section in the prompt
3. Filter out irrelevant mentions (e.g., 'Woodson, Arkansas')
4. Simplify the context structure to avoid confusion
""")

# Create a simplified context for comparison
simplified_context = original_context.replace("[]\\n\\n", "")
print("\n" + "="*80)
print("SIMPLIFIED CONTEXT (removed empty '[]'):")
print("="*80)
print(simplified_context[:1500])