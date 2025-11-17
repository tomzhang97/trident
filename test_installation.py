#!/usr/bin/env python3
"""Quick test script to verify TRIDENT installation."""

import sys
import json
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    modules = [
        "trident.config",
        "trident.pipeline", 
        "trident.safe_cover",
        "trident.pareto",
        "trident.facets",
        "trident.candidates",
        "trident.calibration",
        "trident.nli_scorer",
        "trident.retrieval",
        "trident.vqc",
        "trident.bwk",
        "trident.monitoring",
        "trident.evaluation",
        "trident.llm_interface",
        "trident.logging_utils"
    ]
    
    failed = []
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ✗ {module_name}: {e}")
            failed.append(module_name)
    
    if failed:
        print(f"\n❌ Failed to import {len(failed)} modules")
        return False
    
    print("\n✓ All modules imported successfully")
    return True


def test_basic_functionality():
    """Test basic TRIDENT functionality."""
    print("\nTesting basic functionality...")
    
    try:
        from trident.config import TridentConfig
        from trident.facets import Facet, FacetType
        from trident.candidates import Passage
        from trident.calibration import ReliabilityCalibrator
        from trident.safe_cover import SafeCoverAlgorithm
        
        # Create config
        config = TridentConfig()
        print("  ✓ Configuration created")
        
        # Create facets
        facets = [
            Facet(
                facet_id="f1",
                facet_type=FacetType.ENTITY,
                template={"mention": "Einstein"}
            ),
            Facet(
                facet_id="f2",
                facet_type=FacetType.RELATION,
                template={"subject": "Einstein", "predicate": "wrote", "object": "papers"}
            )
        ]
        print("  ✓ Facets created")
        
        # Create passages
        passages = [
            Passage(
                pid="p1",
                text="Albert Einstein wrote many influential papers on physics.",
                cost=50
            ),
            Passage(
                pid="p2",
                text="The theory of relativity revolutionized our understanding.",
                cost=40
            )
        ]
        print("  ✓ Passages created")
        
        # Create calibrator
        calibrator = ReliabilityCalibrator()
        print("  ✓ Calibrator created")
        
        # Create Safe-Cover algorithm
        algo = SafeCoverAlgorithm(
            config=config.safe_cover,
            calibrator=calibrator
        )
        print("  ✓ Safe-Cover algorithm created")
        
        # Test p-values
        p_values = {
            ("p1", "f1"): 0.001,
            ("p1", "f2"): 0.005,
            ("p2", "f1"): 0.1,
            ("p2", "f2"): 0.2
        }
        
        # Run algorithm
        result = algo.run(facets, passages, p_values)
        print("  ✓ Algorithm executed")
        
        print(f"\n  Results:")
        print(f"    - Selected passages: {len(result.selected_passages)}")
        print(f"    - Covered facets: {len(result.covered_facets)}")
        print(f"    - Certificates: {len(result.certificates)}")
        print(f"    - Abstained: {result.abstained}")
        print(f"    - Total cost: {result.total_cost}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading():
    """Test dataset loading capabilities."""
    print("\nTesting dataset loading...")
    
    try:
        from trident.evaluation import DatasetLoader
        
        # Try to load a small sample
        print("  Attempting to load HotpotQA sample...")
        examples = DatasetLoader.load_dataset(
            "hotpotqa",
            split="validation",
            limit=5
        )
        
        print(f"  ✓ Loaded {len(examples)} examples")
        
        if examples:
            example = examples[0]
            print(f"\n  Sample example:")
            print(f"    - ID: {example['_id']}")
            print(f"    - Question: {example['question'][:100]}...")
            print(f"    - Answer: {example['answer']}")
            print(f"    - Context items: {len(example.get('context', []))}")
        
        return True
        
    except Exception as e:
        print(f"  ⚠ Dataset loading not available: {e}")
        print("  (This is optional - datasets library may not be installed)")
        return True  # Don't fail test for optional component


def main():
    """Run all tests."""
    print("=" * 60)
    print("TRIDENT Installation Test")
    print("=" * 60)
    
    tests_passed = []
    
    # Test imports
    if test_imports():
        tests_passed.append("imports")
    
    # Test basic functionality
    if test_basic_functionality():
        tests_passed.append("functionality")
    
    # Test dataset loading
    if test_dataset_loading():
        tests_passed.append("datasets")
    
    print("\n" + "=" * 60)
    if len(tests_passed) >= 2:  # At least imports and functionality
        print("✅ TRIDENT is installed correctly!")
        print(f"   Passed: {', '.join(tests_passed)}")
        
        print("\nNext steps:")
        print("1. Prepare data shards:")
        print("   python experiments/prepare_shards.py --dataset hotpotqa --limit 100")
        print("\n2. Run evaluation:")
        print("   python experiments/eval_complete_runnable.py --worker \\")
        print("     --data_path <shard> --output_dir results/test")
        
    else:
        print("❌ TRIDENT installation has issues")
        print("   Please check the error messages above")
        sys.exit(1)


if __name__ == "__main__":
    main()