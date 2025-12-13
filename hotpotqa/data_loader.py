#!/usr/bin/env python3
"""HotpotQA dataset loader and converter for TRIDENT pipeline."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


class HotpotQADataLoader:
    """Load and convert HotpotQA data to TRIDENT's standard format."""

    # HotpotQA data files (local)
    DATA_FILES = {
        'dev_distractor': 'hotpot_dev_distractor_v1.json',
        'dev_fullwiki': 'hotpot_dev_fullwiki_v1.json',
        'train': 'hotpot_train_v1.1.json',
        'test_fullwiki': 'hotpot_test_fullwiki_v1.json'
    }

    def __init__(self, data_dir: str = None):
        """Initialize the data loader.

        Args:
            data_dir: Path to HotpotQA data directory. Defaults to hotpotqa/data.
        """
        if data_dir is None:
            # Default to the data directory relative to this file
            data_dir = Path(__file__).parent / "data"
        self.data_dir = Path(data_dir)

    def load_raw(
        self,
        split: str = 'dev_distractor',
        limit: Optional[int] = None,
        use_huggingface: bool = False
    ) -> List[Dict[str, Any]]:
        """Load raw HotpotQA data.

        Args:
            split: One of 'dev_distractor', 'dev_fullwiki', 'train', 'test_fullwiki'
                   or HuggingFace splits: 'validation', 'train'
            limit: Maximum number of examples to load
            use_huggingface: If True, load from HuggingFace datasets

        Returns:
            List of raw HotpotQA examples
        """
        if use_huggingface:
            return self._load_from_huggingface(split, limit)

        return self._load_from_local(split, limit)

    def _load_from_local(
        self,
        split: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Load data from local JSON files."""
        if split not in self.DATA_FILES:
            raise ValueError(f"Unknown split: {split}. Available: {list(self.DATA_FILES.keys())}")

        file_path = self.data_dir / self.DATA_FILES[split]

        if not file_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {file_path}\n"
                f"Please download HotpotQA data from: https://hotpotqa.github.io/\n"
                f"Or use use_huggingface=True to load from HuggingFace datasets."
            )

        with open(file_path, 'r') as f:
            data = json.load(f)

        if limit:
            data = data[:limit]

        return data

    def _load_from_huggingface(
        self,
        split: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Load data from HuggingFace datasets."""
        if not HAS_DATASETS:
            raise ImportError(
                "datasets library not installed. "
                "Install with: pip install datasets"
            )

        # Map split names
        hf_split_map = {
            'dev_distractor': 'validation',
            'dev_fullwiki': 'validation',
            'train': 'train',
            'validation': 'validation'
        }
        hf_split = hf_split_map.get(split, split)

        # Determine which config to use
        if 'fullwiki' in split:
            config = 'fullwiki'
        else:
            config = 'distractor'

        dataset = load_dataset('hotpot_qa', config, split=hf_split)

        examples = []
        for idx, item in enumerate(dataset):
            if limit and idx >= limit:
                break

            # Convert HuggingFace format to standard format
            example = {
                '_id': item['id'],
                'question': item['question'],
                'answer': item['answer'],
                'type': item['type'],
                'level': item['level'],
                'supporting_facts': {
                    'title': item['supporting_facts']['title'],
                    'sent_id': item['supporting_facts']['sent_id']
                },
                'context': {
                    'title': item['context']['title'],
                    'sentences': item['context']['sentences']
                }
            }
            examples.append(example)

        return examples

    def convert_to_standard_format(
        self,
        examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert HotpotQA examples to TRIDENT's standard format.

        Standard format:
        {
            '_id': str,
            'question': str,
            'answer': str,
            'context': List[Tuple[str, List[str]]],  # [(title, [sentences]), ...]
            'supporting_facts': List[Tuple[str, int]],  # [(title, sent_idx), ...]
            'type': str,  # bridge or comparison
            'level': str,  # easy, medium, hard
        }
        """
        converted = []

        for ex in examples:
            # Handle different input formats
            # Format 1: Raw HotpotQA JSON
            if 'context' in ex and isinstance(ex['context'], list):
                # Raw format: context is list of [title, sentences]
                context = [(c[0], c[1]) for c in ex['context']]

                # Supporting facts: list of [title, sent_idx]
                supporting_facts = [
                    (sf[0], sf[1]) for sf in ex.get('supporting_facts', [])
                ]

            # Format 2: HuggingFace format with dict
            elif 'context' in ex and isinstance(ex['context'], dict):
                titles = ex['context'].get('title', [])
                sentences_list = ex['context'].get('sentences', [])
                context = list(zip(titles, sentences_list))

                sf_titles = ex['supporting_facts'].get('title', [])
                sf_sents = ex['supporting_facts'].get('sent_id', [])
                supporting_facts = list(zip(sf_titles, sf_sents))

            else:
                context = []
                supporting_facts = []

            converted.append({
                '_id': ex.get('_id', ex.get('id', '')),
                'question': ex.get('question', ''),
                'answer': ex.get('answer', ''),
                'context': context,
                'supporting_facts': supporting_facts,
                'type': ex.get('type', 'unknown'),
                'level': ex.get('level', 'unknown'),
                # Keep original data for evaluation
                '_original': ex
            })

        return converted

    def load_and_convert(
        self,
        split: str = 'dev_distractor',
        limit: Optional[int] = None,
        use_huggingface: bool = False
    ) -> List[Dict[str, Any]]:
        """Load and convert HotpotQA data in one step.

        Args:
            split: Data split to load
            limit: Maximum number of examples
            use_huggingface: If True, load from HuggingFace

        Returns:
            List of examples in TRIDENT's standard format
        """
        raw = self.load_raw(split, limit, use_huggingface)
        return self.convert_to_standard_format(raw)

    def save_shards(
        self,
        examples: List[Dict[str, Any]],
        output_dir: str,
        shard_size: int = 100,
        split_name: str = 'dev'
    ) -> List[str]:
        """Save examples as shards for parallel processing.

        Args:
            examples: List of converted examples
            output_dir: Output directory for shards
            shard_size: Number of examples per shard
            split_name: Name prefix for shard files

        Returns:
            List of shard file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        shard_paths = []

        for i in range(0, len(examples), shard_size):
            shard = examples[i:i + shard_size]
            start_idx = i
            end_idx = min(i + shard_size, len(examples)) - 1

            shard_name = f"{split_name}_{start_idx}_{end_idx}.json"
            shard_path = output_path / shard_name

            with open(shard_path, 'w') as f:
                json.dump(shard, f, indent=2)

            shard_paths.append(str(shard_path))

        # Create manifest
        manifest = {
            'dataset': 'hotpotqa',
            'split': split_name,
            'num_examples': len(examples),
            'num_shards': len(shard_paths),
            'shard_size': shard_size,
            'shards': shard_paths
        }

        manifest_path = output_path / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        return shard_paths

    def get_stats(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the loaded data."""
        question_types = {}
        difficulty_levels = {}
        total_paragraphs = 0
        total_supporting = 0

        for ex in examples:
            q_type = ex.get('type', 'unknown')
            question_types[q_type] = question_types.get(q_type, 0) + 1

            level = ex.get('level', 'unknown')
            difficulty_levels[level] = difficulty_levels.get(level, 0) + 1

            total_paragraphs += len(ex.get('context', []))

            # Count unique supporting fact titles
            sf = ex.get('supporting_facts', [])
            if sf:
                unique_titles = set(s[0] if isinstance(s, (list, tuple)) else s for s in sf)
                total_supporting += len(unique_titles)

        return {
            'total_examples': len(examples),
            'question_types': question_types,
            'difficulty_levels': difficulty_levels,
            'avg_paragraphs': total_paragraphs / len(examples) if examples else 0,
            'avg_supporting': total_supporting / len(examples) if examples else 0
        }


def main():
    """CLI for HotpotQA data loading."""
    import argparse

    parser = argparse.ArgumentParser(description="Load and convert HotpotQA data")
    parser.add_argument(
        "--split",
        type=str,
        default="dev_distractor",
        choices=['dev_distractor', 'dev_fullwiki', 'train', 'test_fullwiki', 'validation'],
        help="Data split to load"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/hotpotqa_shards",
        help="Output directory for shards"
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=100,
        help="Number of examples per shard"
    )
    parser.add_argument(
        "--stats_only",
        action="store_true",
        help="Only print statistics, don't create shards"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to HotpotQA data directory"
    )
    parser.add_argument(
        "--use_huggingface",
        action="store_true",
        help="Load data from HuggingFace datasets instead of local files"
    )

    args = parser.parse_args()

    loader = HotpotQADataLoader(args.data_dir)

    print(f"Loading HotpotQA data (split: {args.split})...")
    try:
        examples = loader.load_and_convert(
            args.split,
            args.limit,
            use_huggingface=args.use_huggingface
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTrying to load from HuggingFace...")
        examples = loader.load_and_convert(
            args.split,
            args.limit,
            use_huggingface=True
        )

    stats = loader.get_stats(examples)
    print(f"\nDataset Statistics:")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Question types: {stats['question_types']}")
    print(f"  Difficulty levels: {stats['difficulty_levels']}")
    print(f"  Avg paragraphs: {stats['avg_paragraphs']:.1f}")
    print(f"  Avg supporting: {stats['avg_supporting']:.1f}")

    if not args.stats_only:
        print(f"\nCreating shards in {args.output_dir}...")
        shard_paths = loader.save_shards(
            examples,
            args.output_dir,
            args.shard_size,
            args.split
        )
        print(f"Created {len(shard_paths)} shards")
        for path in shard_paths:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
