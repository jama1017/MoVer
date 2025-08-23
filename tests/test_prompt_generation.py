"""
Test the refactored sentence generation implementation.
"""

import random
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mover.nlg.prompt_generator import PromptGenerator
from mover_dataset.dataset_scene_graphs import gen_data_all

def test_basic_functionality():
    """Test basic sentence generation functionality."""
    generator = PromptGenerator()
    
    for gen_data in gen_data_all:
        results = generator.generate(gen_data)
        print(gen_data["file_name"])
        print(f"Generated {len(results)} sentences:")
        sample_results = random.sample(results, min(10, len(results)))
        for i, result in enumerate(sample_results):
            print(f"{i+1}. {result['sentence']}")
            print(f"   Structure: {result['structure']}")
            print(f"   Ordering: {result['ordering']}")
            print(f"   Unseen synonyms: {result['unseen synonyms']}")
            print()
    
    

if __name__ == "__main__":
    test_basic_functionality()
