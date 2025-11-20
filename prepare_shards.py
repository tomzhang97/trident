#!/usr/bin/env python3
"""Script to create data shards for parallel TRIDENT evaluation."""

import argparse
import json
import os
from pathlib import Path

from trident.evaluation import DatasetLoader


def create_shards(args):
    """Create data shards for parallel processing."""
    
    print(f"Loading {args.dataset} dataset...")
    
    if args.local_file:
        # Load from local file
        with open(args.local_file, 'r') as f:
            examples = json.load(f)
        
        # If it's a dictionary with 'data' key (like HotpotQA format)
        if isinstance(examples, dict) and 'data' in examples:
            examples = examples['data']
            
        print(f"Loaded {len(examples)} examples from local file")
    else:
        # Load using DatasetLoader (original method)
        examples = DatasetLoader.load_dataset(
            args.dataset,
            split=args.split,
            limit=args.limit,
            cache_dir=args.cache_dir
        )
    
    print(f"Loaded {len(examples)} examples")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create shards
    shard_paths = []
    for i in range(0, len(examples), args.shard_size):
        shard = examples[i:i + args.shard_size]
        start_idx = i
        end_idx = min(i + args.shard_size, len(examples)) - 1
        
        shard_name = f"{args.split}_{start_idx}_{end_idx}.json"
        shard_path = output_dir / shard_name
        
        with open(shard_path, 'w') as f:
            json.dump(shard, f, indent=2)
        
        shard_paths.append(str(shard_path))
        print(f"Created shard: {shard_path}")
    
    # Create manifest file
    manifest = {
        'dataset': args.dataset,
        'split': args.split,
        'num_examples': len(examples),
        'num_shards': len(shard_paths),
        'shard_size': args.shard_size,
        'shards': shard_paths
    }
    
    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nCreated {len(shard_paths)} shards")
    print(f"Manifest saved to: {manifest_path}")
    
    # Generate run commands
    if args.generate_commands:
        commands_path = output_dir / 'run_commands.sh'
        with open(commands_path, 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write("# Commands to run TRIDENT evaluation on all shards\n\n")
            
            for i, shard_path in enumerate(shard_paths):
                shard_name = Path(shard_path).stem
                output_path = f"results/{args.dataset}_{args.split}/{shard_name}"
                
                cmd = (
                    f"python experiments/eval_complete_runnable.py "
                    f"--worker "
                    f"--data_path {shard_path} "
                    f"--output_dir {output_path} "
                    f"--device {i % args.num_gpus} "
                    f"--budget_tokens {args.budget_tokens} "
                    f"--mode {args.mode} "
                    f"--seed {args.seed}"
                )
                
                f.write(f"# Shard {i+1}/{len(shard_paths)}\n")
                f.write(f"{cmd}\n\n")
        
        print(f"Run commands saved to: {commands_path}")
        
        # Also create a parallel run script
        parallel_path = output_dir / 'run_parallel.sh'
        with open(parallel_path, 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write("# Run TRIDENT evaluation in parallel\n\n")
            f.write(f"GPUS={args.num_gpus}\n")
            f.write("PIDS=()\n\n")
            
            for i, shard_path in enumerate(shard_paths):
                shard_name = Path(shard_path).stem
                output_path = f"results/{args.dataset}_{args.split}/{shard_name}"
                gpu_id = i % args.num_gpus
                
                cmd = (
                    f"python experiments/eval_complete_runnable.py "
                    f"--worker "
                    f"--data_path {shard_path} "
                    f"--output_dir {output_path} "
                    f"--device {gpu_id} "
                    f"--budget_tokens {args.budget_tokens} "
                    f"--mode {args.mode} "
                    f"--seed {args.seed}"
                )
                
                f.write(f"echo 'Starting shard {i+1}/{len(shard_paths)} on GPU {gpu_id}'\n")
                f.write(f"{cmd} &\n")
                f.write("PIDS+=($!)\n")
                
                # Add wait after each batch of GPUs
                if (i + 1) % args.num_gpus == 0 and i < len(shard_paths) - 1:
                    f.write("\n# Wait for current batch to complete\n")
                    f.write("for pid in ${PIDS[@]}; do\n")
                    f.write("    wait $pid\n")
                    f.write("done\n")
                    f.write("PIDS=()\n\n")
            
            f.write("\n# Wait for final batch\n")
            f.write("for pid in ${PIDS[@]}; do\n")
            f.write("    wait $pid\n")
            f.write("done\n\n")
            f.write("echo 'All shards completed!'\n")
            f.write(f"python experiments/aggregate_results.py --results_dir results/{args.dataset}_{args.split}\n")
        
        os.chmod(parallel_path, 0o755)
        print(f"Parallel run script saved to: {parallel_path}")


def main():
    parser = argparse.ArgumentParser(description="Create data shards for TRIDENT evaluation")
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="hotpotqa",
        choices=["hotpotqa", "musique", "2wikimultihop", "nq", "triviaqa", "squad", "squad_v2"],
        help="Dataset to use"
    )
    parser.add_argument(
    "--local_file",
    type=str,
    default=None,
    help="Path to local dataset file (for hotpotqa, etc.)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/_shards",
        help="Output directory for shards"
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=100,
        help="Number of examples per shard"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit total number of examples"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=".cache/datasets",
        help="Cache directory for datasets"
    )
    parser.add_argument(
        "--generate_commands",
        action="store_true",
        help="Generate run commands for each shard"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs available for parallel processing"
    )
    parser.add_argument(
        "--budget_tokens",
        type=int,
        default=2000,
        help="Token budget for experiments"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="safe_cover",
        choices=["safe_cover", "pareto", "both"],
        help="TRIDENT mode to use"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    create_shards(args)


if __name__ == "__main__":
    main()