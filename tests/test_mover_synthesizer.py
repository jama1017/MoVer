import json
import traceback
import argparse
import yaml
from difflib import unified_diff
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from argparse import Namespace
from mover.synthesizers.mover_synthesizer import MoverSynthesizer


def parse_args() -> Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Test MoVer program synthesis on example prompts.')
    parser.add_argument('config_file', type=str, help='Path to the YAML config file')
    return parser.parse_args()


def init_synthesizer(config: Dict[str, Any]) -> MoverSynthesizer:
    """Initialize MoverSynthesizer with configuration.
    
    Args:
        config: Configuration dictionary for the synthesizer
        
    Returns:
        Initialized synthesizer instance
    """
    params = {k: v for k, v in config.get('params', {}).items() if v is not None}
    if config.get('vllm_serve_port') is not None:
        params['vllm_serve_port'] = config['vllm_serve_port']
    
    return MoverSynthesizer(
        model_name=config['name'],
        provider=config['provider'],
        num_ctx=config.get('num_ctx', 128000),
        params=params
    )


class TestProgramSynthesis:
    def __init__(self, config_path: str) -> None:        
        # Get project root directory (2 levels up from this file)
        self.project_root = Path(__file__).parent.parent
        
        # Load config
        with Path(config_path).open('r') as f:
            self.config = yaml.safe_load(f)
            
        # Load test cases
        prompts_file = self.project_root / self.config['paths']['animation_prompts_file']
        with prompts_file.open('r') as f:
            self.test_cases = json.load(f)
            
        # Initialize synthesizer with model config
        self.synthesizer = init_synthesizer(self.config['mover_model'])
        
        # Setup logging file name
        today = datetime.today()
        self.formatted_date = today.strftime("%m%d")
        test_file_name = prompts_file.name
        self.file_name = f"log_{self.formatted_date}_{test_file_name.replace('.json', '')}.txt"


    def test_program(self) -> None:        
        if isinstance(self.test_cases, dict):
            self.test_cases = [self.test_cases]
        
        output_message = []
        final_summary = []
        total_num_failed = 0
        total_num_error = 0
        total_num_test = len(self.test_cases)
        failed_prompts = []
        
        for (i, test_case) in enumerate(self.test_cases):
            try:
                header = f"\n----Prompt #{i}: {test_case['chat_id_name']}-----"
                print(header)
                output_message.append(header)
                
                # Get SVG file path
                if "svg_file_path" not in test_case:
                    svg_file_path = self.project_root / self.config['paths']['svg_dir'] / test_case["svg_name"]
                    test_case["svg_file_path"] = str(svg_file_path)
                
                # Create chat history for the synthesizer
                chat_history = [
                    self.synthesizer.compose_sys_msg(),
                    self.synthesizer.compose_initial_user_prompt(
                        test_case["animation_prompt"], 
                        test_case["svg_file_path"]
                    )
                ]
                
                # Generate program using synthesizer
                response, error = self.synthesizer.generate(chat_history)
                if error:
                    raise Exception(error)
                
                # Extract program from response
                program_str = response.split("```")[1].strip()
                program_string = program_str.rstrip('\n')
                ground_truth_program = test_case['ground_truth_program'].rstrip('\n')
                
                if program_string != ground_truth_program:
                    print("Failed")
                    # Count failures
                    total_num_failed += 1
                    
                    failed_message = f"Prompt #{i} failed: {test_case['chat_id_name']}\n\n"
                    failed_message += f"Prompt: {test_case["animation_prompt"]}\n\n"
                    failed_message += f"Generated program:\n{program_str}\n"
                    
                    diff = unified_diff(ground_truth_program.splitlines(), program_string.splitlines(), lineterm='')
                    failed_message += '\nString differences:\n'
                    failed_message += '\n'.join(list(diff))
                    failed_message += '\n\n'
                    output_message.append(failed_message)
                    
                    # Add to failed prompts list
                    failed_prompt = {
                        "svg_name": test_case["svg_name"],
                        "animation_prompt": test_case["animation_prompt"],
                        "chat_id_name": test_case["chat_id_name"],
                        "ground_truth_program": ground_truth_program,
                        "generated_program": program_string,
                        "has_run": False
                    }
                    failed_prompts.append(failed_prompt)
                    
                else:
                    print("Passed")
                    output_message.append(f"Test case {i} passed\n")
                    
            except Exception as e:
                # Count errors
                total_num_error += 1
                
                output_message.append(f"Test case {i} error\n")
                output_message.append(f"Prompt: {test_case["animation_prompt"]}")
                output_message.append(str(traceback.format_exc()))
            
        summary = f"\n## Summary: {total_num_failed} / {total_num_test} failed, {total_num_error} / {total_num_test} errored\n"
        output_message.append(summary)
        final_summary.append(summary)
        print(summary)
            
        # Write test results
        results_dir = self.project_root / "tests" / "results"
        results_dir.mkdir(exist_ok=True)
        results_file = results_dir / self.file_name
        results_file.write_text('\n'.join(output_message))

        # Write final summary
        with results_file.open('a') as f:
            f.write('\n\n###################\n')
            f.write('\n\n'.join([sum.rstrip('\n') for sum in final_summary]))
            f.write(f"\n\nTotal: {total_num_failed} / {total_num_test} failed, {total_num_error} / {total_num_test} errored\n")

        # Write failed prompts
        if failed_prompts:
            failed_prompts_dir = self.project_root / "tests" / "failed_prompts"
            failed_prompts_dir.mkdir(exist_ok=True)
            test_file_name = Path(self.config['paths']['animation_prompts_file']).name
            failed_prompts_file = failed_prompts_dir / f"failed_prompts_{self.formatted_date}_{test_file_name}"
            failed_prompts_file.write_text(json.dumps(failed_prompts, indent=4))


if __name__ == '__main__':
    args = parse_args()
    test = TestProgramSynthesis(args.config_file)
    test.test_program()