import json
import random
from typing import List, Dict, Any, Optional
from pathlib import Path

from mover.nlg.mover_fol_generator import MoVerFOLGenerator
from mover.nlg.prompt_generator import PromptGenerator
from mover.nlg.data_classes import Object, Motion
from mover.nlg.sentence_generation.vocab import Vocab
from mover.synthesizers.prompt_rewriter import PromptRewriter
from mover.synthesizers.utils import extract_code_block

## Import MoVer dataset scene graphs as an example
## You can replace this with your own scene graphs or define directly below
from dataset_scene_graphs import gen_data_all


##############################################################################
## CONFIGURATION - EDIT THESE PARAMETERS TO CUSTOMIZE DATASET GENERATION
##############################################################################
##
## This section contains all the configurable parameters for dataset creation.
## Simply edit the values below and run: python create_dataset.py
##

## Basic configuration
SVG_NAME = "four_objects.svg"       ## The SVG file to use for the dataset
OUTPUT_DIR = Path(__file__).parent  ## Output directory for the dataset
DATASET_NAME = "example_dataset"  ## Custom name for your dataset

## Prompt sampling configuration
## These control how many prompts are sampled from generated vs paraphrased
PROMPTS_FROM_GENERATOR = 80  ## Number of prompts to sample from PromptGenerator, which uses a template-based approach
PROMPTS_FROM_REWRITER = 20  ## Number of rewrites to generate and include from PromptRewriter, which uses an LLM
TOTAL_PROMPTS_PER_SCENE = PROMPTS_FROM_GENERATOR + PROMPTS_FROM_REWRITER

## PromptGenerator configuration
## Vocab configuration - set to None to use default, or provide path to custom vocab.json
## The default vocab.json is src/mover/nlg/assets/vocab.json. The synonym_labels are used to mark unseen synonyms (1 for unseen, 0 for seen)
VOCAB_PATH = None  ## or "path/to/vocab.json" for custom vocabulary

## Sampling configuration for PromptGenerator
SAMPLING_CONFIG = {
    "enabled": True,
    "max_per_group": 2,  ## Maximum sentences per (template, unit) group
    "multi_motion_downsample_num": 50,
}

## Specify your scene graphs
example_object = Object(
    shape="circle",
    fill="blue",
)

example_motion = Motion(
    type = "translate",
    agent = [example_object],
    direction = [1.0, 0.0],
)

example_motion_2 = Motion(
    type = "rotate",
    agent = [example_object],
    direction = 1.0,
)

example_scene_graph = {
    "motions": [example_motion],
    "relations": [None],
    "file_name": "example_scene_graph.json"
}

example_scene_graph_2 = {
    "motions": [example_motion_2],
    "relations": [None],
    "file_name": "example_scene_graph_2.json"
}

## To generate all scene graphs in the MoVer dataset, use gen_data_all
SCENE_GRAPHS = [example_scene_graph, example_scene_graph_2]  


##############################################################################
## MAIN SCRIPT FUNCTIONS
##############################################################################

def generate_rewrites(prompt_rewriter: Optional[PromptRewriter], base_prompt: str, num_rewrites: int = 20) -> List[str]:
    """Generate rewrites using PromptRewriter."""
        
    chat_history = [
        prompt_rewriter.compose_sys_msg(),
        prompt_rewriter.compose_initial_user_prompt(base_prompt, num_rewrites)
    ]
    
    error = ""
    while error is not None:
        response, error = prompt_rewriter.generate(chat_history)
        
    ## Extract JSON from response
    json_code = extract_code_block(response, '```json', '```')
    response_json = json.loads(json_code)
    return response_json.get("variations", [])


def sample_and_combine_prompts(generated_prompts: List[Dict], rewrites: List[str], 
                              num_from_generator: int, num_from_rewriter: int) -> List[Dict[str, Any]]:
    """Sample prompts from generated and combine with rewrites."""
    
    ## Sample from generated prompts
    num_to_sample = min(num_from_generator, len(generated_prompts))
    sampled_generated = random.sample(generated_prompts, num_to_sample) if generated_prompts else []
    
    ## Convert rewrites to prompt format
    rewrite_prompts = []
    for rewrite in rewrites[:num_from_rewriter]:
        rewrite_prompts.append({
            "sentence": rewrite,
            "structure": "llm",
            "unseen synonyms": {}
        })
    
    ## Combine prompts
    combined_prompts = sampled_generated + rewrite_prompts
    
    ## If we have fewer than target, pad with more generated prompts if available
    target_total = num_from_generator + num_from_rewriter
    if len(combined_prompts) < target_total and len(generated_prompts) > num_to_sample:
        remaining_generated = [p for p in generated_prompts if p not in sampled_generated]
        additional_needed = min(target_total - len(combined_prompts), len(remaining_generated))
        if additional_needed > 0:
            additional_prompts = random.sample(remaining_generated, additional_needed)
            combined_prompts.extend(additional_prompts)
    
    ## Shuffle and return
    random.shuffle(combined_prompts)
    return combined_prompts


def format_dataset_entry(prompt_data: Dict[str, Any], chat_id_name: str, ground_truth_program: str, svg_name: str) -> Dict[str, Any]:
    """Format a single dataset entry according to the required JSON structure."""
    return {
        "svg_name": svg_name,
        "animation_prompt": prompt_data["sentence"],
        "chat_id_name": chat_id_name,
        "unseen synonyms": prompt_data.get("unseen synonyms", {}),
        "structure": prompt_data.get("structure", []),
        "ground_truth_program": ground_truth_program,
        "has_run": False
    }


def process_scene_graph(scene_graph: Dict[str, Any], fol_generator: MoVerFOLGenerator, 
                       prompt_generator: PromptGenerator, prompt_rewriter: Optional[PromptRewriter]) -> List[Dict[str, Any]]:
    """Process a single scene graph through the complete pipeline."""
    
    print(f"Processing scene graph: {scene_graph.get('file_name', 'unknown')}")
    
    ## Step 0: Generate ground truth MoVer program
    ground_truth_program = fol_generator.generate_program(scene_graph)
    
    ## Step 1: Generate prompts in a template-based manner using PromptGenerator
    generated_prompts = prompt_generator.generate(scene_graph)
    print(f"  Generated {len(generated_prompts)} prompts")
    
    ## Step 2: Generate LLM rewrites using PromptRewriter
    rewrites = []
    if PROMPTS_FROM_REWRITER > 0 and generated_prompts:
        ## Use the first generated prompt as base for rewriting
        base_prompt = generated_prompts[0]
        rewrites = generate_rewrites(prompt_rewriter, base_prompt, PROMPTS_FROM_REWRITER)
        print(f"  Generated {len(rewrites)} rewrites")
    
    ## Sample and combine prompts
    sampled_prompts = sample_and_combine_prompts(
        generated_prompts, 
        rewrites,
        PROMPTS_FROM_GENERATOR,
        PROMPTS_FROM_REWRITER
    )
    print(f"  Sampled {len(sampled_prompts)} total prompts")
    
    ## Format dataset entries
    dataset_entries = []
    base_name = scene_graph.get("file_name", "unknown").replace(".json", "")
    
    for i, prompt_data in enumerate(sampled_prompts):
        entry = format_dataset_entry(
            prompt_data=prompt_data,
            chat_id_name=f"{base_name}_{i:03d}",
            ground_truth_program=ground_truth_program,
            svg_name=SVG_NAME
        )
        dataset_entries.append(entry)
        
    return dataset_entries


def create_dataset():
    """Main function to create the dataset."""
    print("=== MoVer Dataset Creation ===")
    print(f"Dataset name: {DATASET_NAME}")
    print(f"SVG name: {SVG_NAME}")
    print(f"Prompts per scene: {PROMPTS_FROM_GENERATOR} generated + {PROMPTS_FROM_REWRITER} rewrites = {TOTAL_PROMPTS_PER_SCENE} total")
    print(f"Use rewriting: {PROMPTS_FROM_REWRITER > 0}")
    print()
    
    ## Get scene graphs
    scene_graphs = SCENE_GRAPHS
    print(f"Processing {len(scene_graphs)} scene graphs...")
    
    ## Initialize the ground truth program generator
    fol_generator = MoVerFOLGenerator()
    
    ## Initialize the prompt generator
    vocab = Vocab(VOCAB_PATH)
    prompt_generator = PromptGenerator(vocab=vocab, sampling_config=SAMPLING_CONFIG)
    
    ## Initialize the prompt rewriter
    prompt_rewriter = PromptRewriter()
    
    ## Process all scene graphs
    all_dataset_entries = []
    for scene_graph in scene_graphs:
        try:
            scene_entries = process_scene_graph(scene_graph, fol_generator, prompt_generator, prompt_rewriter)
            all_dataset_entries.extend(scene_entries)
        except Exception as e:
            print(f"Error processing scene graph {scene_graph.get('file_name', 'unknown')}: {e}")
            continue
    
    ## Save dataset
    output_file = OUTPUT_DIR / f"{DATASET_NAME}_{len(scene_graphs)}_{TOTAL_PROMPTS_PER_SCENE}.json"
    print(f"\nSaving {len(all_dataset_entries)} entries to {output_file}")
    
    with open(output_file, 'w') as f:
        json.dump(all_dataset_entries, f, indent=4)
    
    print("Dataset creation completed!")
    print(f"Total entries: {len(all_dataset_entries)}")
    print(f"Average per scene graph: {len(all_dataset_entries) / len(scene_graphs):.1f}")
    

if __name__ == "__main__":    
    create_dataset()
