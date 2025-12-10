#!/usr/bin/env python3
"""MuSiQue dataset loader and converter for TRIDENT pipeline."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any


class MuSiQueDataLoader:
    """Load and convert MuSiQue data to TRIDENT's standard format."""

    # MuSiQue data files
    DATA_FILES = {
        'ans_dev': 'musique_ans_v1.0_dev.jsonl',
        'ans_test': 'musique_ans_v1.0_test.jsonl',
        'full_dev': 'musique_full_v1.0_dev.jsonl',
        'full_test': 'musique_full_v1.0_test.jsonl',
        'singlehop': 'dev_test_singlehop_questions_v1.0.json'
    }

    def __init__(self, data_dir: str = None):
        """Initialize the data loader.

        Args:
            data_dir: Path to MuSiQue data directory. Defaults to musique/data.
        """
        if data_dir is None:
            # Default to the data directory relative to this file
            data_dir = Path(__file__).parent / "data"
        self.data_dir = Path(data_dir)

    def load_raw(
        self,
        split: str = 'ans_dev',
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Load raw MuSiQue data.

        Args:
            split: One of 'ans_dev', 'ans_test', 'full_dev', 'full_test', 'singlehop'
            limit: Maximum number of examples to load

        Returns:
            List of raw MuSiQue examples
        """
        if split not in self.DATA_FILES:
            raise ValueError(f"Unknown split: {split}. Available: {list(self.DATA_FILES.keys())}")

        file_path = self.data_dir / self.DATA_FILES[split]

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        examples = []

        if file_path.suffix == '.jsonl':
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        examples.append(json.loads(line))
                        if limit and len(examples) >= limit:
                            break
        else:
            # JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    examples = data[:limit] if limit else data
                else:
                    examples = [data]

        return examples

    def convert_to_standard_format(
        self,
        examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert MuSiQue examples to TRIDENT's standard format.

        Standard format:
        {
            '_id': str,
            'question': str,
            'answer': str,
            'answer_aliases': List[str],
            'context': List[Tuple[str, List[str]]],  # [(title, [sentences]), ...]
            'supporting_facts': List[Tuple[str, int]],  # [(title, sent_idx), ...]
            'type': str,  # hop type
            'answerable': bool,
            'question_decomposition': List[Dict]
        }
        """
        converted = []

        for ex in examples:
            # Extract supporting fact indices
            supporting_facts = []
            context = []

            for para in ex.get('paragraphs', []):
                title = para.get('title', f"para_{para['idx']}")
                text = para.get('paragraph_text', '')

                # Split into sentences (simple heuristic)
                sentences = self._split_sentences(text)
                context.append((title, sentences))

                # Track supporting paragraphs
                if para.get('is_supporting', False):
                    # Add all sentences from supporting paragraph
                    for sent_idx in range(len(sentences)):
                        supporting_facts.append((title, sent_idx))

            # Determine hop type from ID
            hop_type = self._extract_hop_type(ex.get('id', ''))

            converted.append({
                '_id': ex.get('id', ''),
                'question': ex.get('question', ''),
                'answer': ex.get('answer', ''),
                'answer_aliases': ex.get('answer_aliases', []),
                'context': context,
                'supporting_facts': supporting_facts,
                'type': hop_type,
                'level': self._get_level_from_hop(hop_type),
                'answerable': ex.get('answerable', True),
                'question_decomposition': ex.get('question_decomposition', []),
                # Keep original data for evaluation
                '_original': {
                    'paragraphs': ex.get('paragraphs', [])
                }
            })

        return converted

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _extract_hop_type(self, example_id: str) -> str:
        """Extract hop type from MuSiQue example ID."""
        # IDs like "2hop__123_456", "3hop1__123_456_789", etc.
        if example_id.startswith('2hop'):
            return '2hop'
        elif example_id.startswith('3hop'):
            return '3hop'
        elif example_id.startswith('4hop'):
            return '4hop'
        else:
            return 'unknown'

    def _get_level_from_hop(self, hop_type: str) -> str:
        """Map hop type to difficulty level."""
        mapping = {
            '2hop': 'medium',
            '3hop': 'hard',
            '4hop': 'hard',
            'unknown': 'unknown'
        }
        return mapping.get(hop_type, 'unknown')

    def load_and_convert(
        self,
        split: str = 'ans_dev',
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Load and convert MuSiQue data in one step.

        Args:
            split: Data split to load
            limit: Maximum number of examples

        Returns:
            List of examples in TRIDENT's standard format
        """
        raw = self.load_raw(split, limit)
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
            'dataset': 'musique',
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
        hop_types = {}
        answerable_count = 0
        total_paragraphs = 0
        total_supporting = 0

        for ex in examples:
            hop_type = ex.get('type', 'unknown')
            hop_types[hop_type] = hop_types.get(hop_type, 0) + 1

            if ex.get('answerable', True):
                answerable_count += 1

            if '_original' in ex:
                paras = ex['_original'].get('paragraphs', [])
                total_paragraphs += len(paras)
                total_supporting += sum(1 for p in paras if p.get('is_supporting', False))
            else:
                total_paragraphs += len(ex.get('context', []))
                total_supporting += len(set(t for t, _ in ex.get('supporting_facts', [])))

        return {
            'total_examples': len(examples),
            'answerable': answerable_count,
            'unanswerable': len(examples) - answerable_count,
            'hop_types': hop_types,
            'avg_paragraphs': total_paragraphs / len(examples) if examples else 0,
            'avg_supporting': total_supporting / len(examples) if examples else 0
        }


def main():
    """CLI for MuSiQue data loading."""
    import argparse

    parser = argparse.ArgumentParser(description="Load and convert MuSiQue data")
    parser.add_argument(
        "--split",
        type=str,
        default="ans_dev",
        choices=['ans_dev', 'ans_test', 'full_dev', 'full_test', 'singlehop'],
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
        default="runs/musique_shards",
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
        help="Path to MuSiQue data directory"
    )

    args = parser.parse_args()

    loader = MuSiQueDataLoader(args.data_dir)

    print(f"Loading MuSiQue data (split: {args.split})...")
    examples = loader.load_and_convert(args.split, args.limit)

    stats = loader.get_stats(examples)
    print(f"\nDataset Statistics:")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Answerable: {stats['answerable']}")
    print(f"  Unanswerable: {stats['unanswerable']}")
    print(f"  Hop types: {stats['hop_types']}")
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
