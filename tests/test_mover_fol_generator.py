import sys
import os
import json
from pathlib import Path
import difflib

from dataset_scene_graphs import gen_data_all
from mover.nlg.mover_fol_generator import MoVerFOLGenerator


def load_ground_truth_programs():
    """Load ground truth programs from mover_dataset JSON files."""
    ground_truth = {}
    
    # Define dataset files to load
    dataset_files = [
        "../mover_dataset/mover_prompts_atomic_12_100.json",
        "../mover_dataset/mover_prompts_spatial_14_100.json", 
        "../mover_dataset/mover_prompts_temporal_12_100.json",
        "../mover_dataset/mover_prompts_spatial_temporal_18_100.json"
    ]
    
    for dataset_file in dataset_files:
        if Path(dataset_file).exists():
            with open(dataset_file, 'r') as f:
                data = json.load(f)
                for entry in data:
                    # Extract base name from chat_id_name (remove numeric suffix)
                    chat_id = entry["chat_id_name"]
                    base_name = "_".join(chat_id.split("_")[:-1])  # Remove last part (numeric suffix)
                    
                    if base_name not in ground_truth:
                        ground_truth[base_name] = entry["ground_truth_program"]
        else:
            print(f"Warning: Dataset file {dataset_file} not found")
    
    return ground_truth


def extract_file_base_name(file_name):
    """Extract base name from file_name (remove .json suffix)."""
    return file_name.replace('.json', '')


def normalize_program(program):
    """Normalize a FOL program for comparison."""
    # Remove extra whitespace and normalize line endings
    lines = [line.strip() for line in program.strip().split('\n') if line.strip()]
    return '\n'.join(lines)


def compare_programs(generated, expected, test_name):
    """Compare generated program with expected ground truth."""
    gen_norm = normalize_program(generated)
    exp_norm = normalize_program(expected)
    
    if gen_norm == exp_norm:
        print(f"PASS: {test_name}")
        return True
    else:
        print(f"FAIL: {test_name}")
        print(f"Expected:\n{exp_norm}\n")
        print(f"Generated:\n{gen_norm}\n")
        
        # Show unified diff
        expected_lines = exp_norm.splitlines(keepends=True)
        generated_lines = gen_norm.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            expected_lines, 
            generated_lines, 
            fromfile='expected', 
            tofile='generated', 
            lineterm=''
        )
        
        print("Unified diff:")
        for line in diff:
            if line.startswith('+++') or line.startswith('---'):
                print(line)
            elif line.startswith('@@'):
                print(line)
            elif line.startswith('+'):
                print(f"+ {line[1:]}")
            elif line.startswith('-'):
                print(f"- {line[1:]}")
            else:
                print(f"  {line}")
        
        print("-" * 50)
        return False


def test_scene_graph(gen_data, ground_truth_programs, test_name):
    """Test a single scene graph against ground truth."""
    print(f"\nTesting: {test_name}")
    print(f"File: {gen_data['file_name']}")
    
    # Generate program using our FOLGenerator
    generator = MoVerFOLGenerator()
    generated_program = generator.generate_program(gen_data)
    
    # Extract base name and find ground truth
    base_name = extract_file_base_name(gen_data['file_name'])
    
    if base_name in ground_truth_programs:
        expected_program = ground_truth_programs[base_name]
        return compare_programs(generated_program, expected_program, test_name)
    else:
        print(f"WARNING: No ground truth found for {base_name}")
        print(f"Generated program:\n{generated_program}")
        return False


def run_all_tests():
    """Run all test cases using gen_data_all."""
    print("Starting Comprehensive FOL Program Generation Tests")
    print("Testing against ground truth from mover_dataset")
    print("="*60)
    
    ground_truth = load_ground_truth_programs()
    passed = 0
    total = len(gen_data_all)
    
    for i, gen_data in enumerate(gen_data_all):
        test_name = f"test_{i+1:03d}"
        if test_scene_graph(gen_data, ground_truth, test_name):
            passed += 1
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"TOTAL:                 {passed:2d}/{total:2d} passed ({100*passed/total:.1f}%)")
    
    if passed == total:
        print("\nALL TESTS PASSED!")
    else:
        print(f"\n{total - passed} tests failed")


if __name__ == "__main__":
    run_all_tests()