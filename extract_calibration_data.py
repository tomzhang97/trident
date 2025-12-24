#!/usr/bin/env python3
"""
Extract calibration data from HotpotQA dataset.

This script:
1. Loads HotpotQA data with context and supporting_facts
2. Mines facets from questions
3. Scores (facet, passage) pairs with NLI
4. Labels passages as sufficient/insufficient based on supporting_facts
5. Outputs calibration data in JSONL format

Usage:
    python extract_calibration_data.py \
        --data_path hotpotqa/data/hotpot_train_v1.json \
        --output_path calibration_data.jsonl \
        --num_samples 500 \
        --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# Add trident to path
sys.path.insert(0, str(Path(__file__).parent))

from trident.facets import FacetMiner
from trident.nli_scorer import NLIScorer
from trident.candidates import Passage
from trident.config import TridentConfig, NLIConfig


def load_hotpotqa_data(path: str, limit: int = None) -> List[Dict[str, Any]]:
    """Load HotpotQA data."""
    with open(path) as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    return data


def extract_calibration_samples(
    data: List[Dict[str, Any]],
    facet_miner: FacetMiner,
    nli_scorer: NLIScorer,
    device: str = "cuda:0"
) -> List[Dict[str, Any]]:
    """
    Extract calibration samples from HotpotQA data.

    For each question:
    - Mine facets
    - Score all (facet, passage) pairs with NLI
    - Label based on supporting_facts (gold labels)
    """
    samples = []

    for example in tqdm(data, desc="Extracting calibration samples"):
        question = example['question']
        context = example.get('context', [])
        supporting_facts = example.get('supporting_facts', [])

        # Skip if no context or supporting facts
        if not context or not supporting_facts:
            continue

        # Mine facets
        facets = facet_miner.extract_facets(question, supporting_facts)
        if not facets:
            continue

        # Build supporting facts lookup
        supporting_titles = {title for title, _ in supporting_facts}

        # Process each passage
        for i, (title, sentences) in enumerate(context):
            passage_text = ' '.join(sentences)
            passage_id = f"context_{i}"

            # Determine if this passage is in supporting facts
            is_supporting = title in supporting_titles

            # Create Passage object for NLI scoring
            passage = Passage(
                pid=passage_id,
                text=passage_text,
                cost=len(passage_text.split())  # Approximate token count
            )

            # Score each facet against this passage
            for facet in facets:
                # Get NLI score using the correct API
                score = nli_scorer.score(passage, facet)

                # Create calibration sample
                sample = {
                    'id': f"{example['_id']}_{passage_id}_{facet.facet_id}",
                    'query_id': example['_id'],
                    'score': float(score),
                    'label': 1 if is_supporting else 0,  # Binary label
                    'metadata': {
                        'facet_type': facet.facet_type.value,
                        'text_length': len(passage_text),
                        'retriever_score': 1.0,  # Perfect retrieval (oracle context)
                        'passage_id': passage_id,
                        'facet_id': facet.facet_id,
                        'question': question,
                        'facet_hypothesis': facet.to_hypothesis(),  # Use to_hypothesis() method
                    }
                }
                samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser(description="Extract calibration data from HotpotQA")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to HotpotQA data (train set recommended)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="calibration_data.jsonl",
        help="Output path for calibration data"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of questions to process (default: 500)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for NLI scorer"
    )

    args = parser.parse_args()

    # Initialize components
    print("Initializing NLI scorer and facet miner...")
    config = TridentConfig()
    facet_miner = FacetMiner(config)
    nli_scorer = NLIScorer(config.nli, device=args.device)

    # Load data
    print(f"Loading data from {args.data_path}...")
    data = load_hotpotqa_data(args.data_path, limit=args.num_samples)
    print(f"Loaded {len(data)} examples")

    # Extract calibration samples
    print("Extracting calibration samples...")
    samples = extract_calibration_samples(data, facet_miner, nli_scorer, args.device)

    # Save to JSONL
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"\n✅ Extracted {len(samples)} calibration samples")
    print(f"📁 Saved to: {output_path}")

    # Print statistics
    positive = sum(1 for s in samples if s['label'] == 1)
    negative = len(samples) - positive
    print(f"\n📊 Statistics:")
    print(f"  Positive samples: {positive} ({positive/len(samples)*100:.1f}%)")
    print(f"  Negative samples: {negative} ({negative/len(samples)*100:.1f}%)")
    print(f"  Avg NLI score: {sum(s['score'] for s in samples) / len(samples):.3f}")

    print(f"\n🔧 Next step: Train calibrator")
    print(f"  python train_calibration.py \\")
    print(f"    --data_path {output_path} \\")
    print(f"    --output_path calibrator.json \\")
    print(f"    --use_mondrian")


if __name__ == "__main__":
    main()
