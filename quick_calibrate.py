#!/usr/bin/env python3
"""Quick script to generate synthetic calibration for testing."""

import json
import pickle
import numpy as np
from pathlib import Path

# Generate synthetic calibration data
np.random.seed(42)
n_samples = 2000

# Use canonical facet types (BRIDGE_HOP not BRIDGE_HOP1/2)
facet_types = ["ENTITY", "RELATION", "TEMPORAL", "NUMERIC", "BRIDGE_HOP"]
data = []

for i in range(n_samples):
    # Generate NLI score (higher = more relevant)
    score = np.random.beta(3, 2)  # Skewed toward higher scores

    # Label: 1 if passage is sufficient, 0 otherwise
    # Make it correlated with score but add noise
    prob_sufficient = (score ** 1.5)  # Non-linear relationship
    label = 1 if np.random.random() < prob_sufficient else 0

    # Metadata for Mondrian binning
    facet_type = np.random.choice(facet_types)
    text_length = int(np.random.uniform(20, 400))
    retriever_score = float(np.random.beta(2, 2))

    data.append({
        'id': f'sample_{i}',
        'score': float(score),
        'label': int(label),
        'metadata': {
            'facet_type': facet_type,
            'text_length': text_length,
            'retriever_score': retriever_score,
            'passage_id': f'p_{i}',
            'facet_id': f'f_{i % 100}'
        }
    })

# Save to JSONL
output_path = Path(__file__).parent / 'synthetic_calibration.jsonl'
with open(output_path, 'w') as f:
    for item in data:
        f.write(json.dumps(item) + '\n')

print(f"✅ Generated {n_samples} synthetic calibration samples")
print(f"📁 Saved to: {output_path}")
print(f"\nNext: Run train_calibration.py to build the calibrator")
print(f"  python train_calibration.py --data_path {output_path} --output_path calibrator.json")