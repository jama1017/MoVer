import json
import argparse
import subprocess
import traceback
from typing import Dict, Any
from pathlib import Path
import yaml
from mover.synthesizers.animation_synthesizer import AnimationSynthesizer
from mover.synthesizers.mover_synthesizer import MoverSynthesizer
from mover.dsl.mover_verifier import MoverVerifier
from mover.converter.mover_converter import convert_animation


class AnimationPipeline:
    """Main pipeline class for handling animation generation and verification."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize pipeline with configuration.
        
        Args:
            config: Dictionary containing pipeline configuration
        """
        ## Store config
        self.config = config
        
        ## Initialize synthesizers and verifier
        self.animation_synthesizer = self._init_synthesizer(
            AnimationSynthesizer,
            self.config['animation_model']
        )
        self.mover_synthesizer = self._init_synthesizer(
            MoverSynthesizer,
            self.config['mover_model']
        )
        self.verifier = MoverVerifier()


    def _init_synthesizer(self, synthesizer_class: Any, config: Dict[str, Any]) -> Any:
        """Initialize a synthesizer with its configuration.
        
        Args:
            synthesizer_class: The synthesizer class to instantiate
            config: Configuration dictionary for the synthesizer
            
        Returns:
            Initialized synthesizer instance
        """
        params = {k: v for k, v in config.get('params', {}).items() if v is not None}
        if config.get('vllm_serve_port') is not None:
            params['vllm_serve_port'] = config['vllm_serve_port']
        
        return synthesizer_class(
            model_name=config['name'],
            provider=config['provider'],
            num_ctx=config.get('num_ctx', 128000),
            params=params
        )


    def _write_log(self, log_path: Path, status: str, num_iter: int, message: str = "") -> None:
        """Write a log file with consistent formatting.
        
        Args:
            log_path: Path to write the log file
            status: Status of the run (success/fail/error)
            num_iter: Number of iterations taken
            message: Optional additional message to include
        """
        with open(log_path, "w") as f:
            if message:
                f.write(f"{message}\n\n")
            
            f.write(f"Prompt {log_path.parent.name} {status} in {num_iter} iterations.\n")
            
            ## Log all config sections
            for section, values in self.config.items():
                f.write(f"\n[{section}]\n")
                for key, value in values.items():
                    f.write(f"{key}: {value}\n")


    def _save_chat_messages(self, chat_file_path: Path, chat_messages: list) -> None:
        with open(chat_file_path, "w") as f:
            json.dump(chat_messages, f, indent=4)


    def run_loop(self, animation_prompt_data: Dict[str, Any]) -> None:
        """Run the main chat loop for animation generation and verification.
        
        Args:
            animation_prompt_data: Data about the animation prompt including the prompt text,
                                 chat ID, and optional ground truth program
        """
        chat_id_name = animation_prompt_data["chat_id_name"]
        animation_prompt = animation_prompt_data["animation_prompt"]
        paths_config = self.config['paths']
        
        chat_dir_path = Path(paths_config['parent_dir']) / chat_id_name
        chat_file_path = chat_dir_path / f"{chat_id_name}.json"
        
        if "svg_file_path" not in animation_prompt_data:
            svg_file_path = Path(paths_config['svg_dir']) / animation_prompt_data["svg_name"]
            animation_prompt_data["svg_file_path"] = str(svg_file_path)
        
        ## Setup chat directory
        if not chat_dir_path.exists():
            chat_dir_path.mkdir(parents=True)
        else:
            if any(f.suffix == '.log' for f in chat_dir_path.iterdir()):
                print(f"Prompt {chat_id_name} has already been run. Skipping.")
                return
            ## Clear chat directory as it has not been completed yet
            for f in chat_dir_path.iterdir():
                if f.is_file():
                    f.unlink()

        ## Get or generate verification program
        verification_program_path = chat_dir_path / f"{chat_id_name}.py"
        if not verification_program_path.exists():
            if "ground_truth_program" in animation_prompt_data:
                print("\n\nRetrieving verification program")
                verification_program = animation_prompt_data["ground_truth_program"]
                with open(verification_program_path, "w") as f:
                    f.write(verification_program)
            else:
                print("\n\nGenerating verification program")
                mover_model = self.config['mover_model']
                print(f"Model:{mover_model['name']}, Provider:{mover_model['provider']}")
                chat_history = [
                    self.mover_synthesizer.compose_sys_msg(),
                    self.mover_synthesizer.compose_initial_user_prompt(animation_prompt, animation_prompt_data["svg_file_path"])
                ]
                _, error_msg = self.mover_synthesizer.generate(
                    chat_history,
                    str(verification_program_path) ## save verification program to file
                )
                if error_msg:
                    raise ValueError(f"Error generating verification program: {error_msg}")
                with open(verification_program_path, "r") as f:
                    verification_program = f.read()
                print(f"\nGenerated verification program:\n{verification_program}")

        else:
            print("\n\nReading verification program")
            with open(verification_program_path, "r") as f:
                verification_program = f.read()

        ## Initialize chat
        chat_messages = [
            self.animation_synthesizer.compose_sys_msg(),
            self.animation_synthesizer.compose_initial_user_prompt(animation_prompt, animation_prompt_data["svg_file_path"])
        ]

        is_correct = False
        is_mover_doc_sent = False
        num_iter = 0
        server_config = self.config['server']
        pipeline_config = self.config['pipeline']
        max_iter = pipeline_config['max_iter']

        ## Run loop until correct or max iterations reached
        while not is_correct and num_iter < max_iter:
            print(f"\n\n---------iteration {num_iter}---------")
            animation_model = self.config['animation_model']
            print(f"Chat id: {chat_id_name}, Model: {animation_model['name']}, Provider: {animation_model['provider']}\n")
            
            ## Generate animation code
            html_file_name = f"{chat_id_name}_{num_iter}.html"
            html_file_path = chat_dir_path / html_file_name
            response, error_msg = self.animation_synthesizer.generate(
                chat_messages,
                animation_prompt_data["svg_file_path"],
                str(html_file_path)
            )

            ## Update chat history
            chat_messages.append({"role": "assistant", "content": response})
            self._save_chat_messages(chat_file_path, chat_messages)

            ## Handle error case
            if error_msg:
                print(f"\n\nError extracting code block: {error_msg}")
                print("Continuing to next iteration with help message.")
                chat_messages.append({
                    "role": "user",
                    "content": "Animation program code is not wrapped in ```javascript and ``` tags. Please wrap it in ```javascript and ``` tags."
                })
                self._save_chat_messages(chat_file_path, chat_messages)

                num_iter += 1
                continue

            ## Convert HTML to animation data
            animation_data_file = f"{chat_id_name}_{num_iter}_data.json"
            animation_data_path = chat_dir_path / animation_data_file
            convert_animation(str(html_file_path), server_config['port'], server_config['create_video'])
            
            ## Verify animation
            try:
                verification_result = self.verifier.verify(
                    str(verification_program_path),
                    str(animation_data_path)
                )
                print(f"\nVerification result:\n{verification_result}")
            except Exception as e:
                traceback_str = str(traceback.format_exc())
                print(f"\n\n{traceback_str}")
                print("\n\nVerifier errored out. Setting verification result to False.")
                verification_result = "False: invalid animation."
                with open(chat_dir_path / f"{chat_id_name}_{num_iter}.out", "w") as f:
                    f.write(traceback_str)
            
            ## Process verification result
            if "False" in verification_result:
                is_correct = False
                
                ## Build verification message with DSL documentation on first failure
                verification_message_template = self.verifier.get_correction_msg_template()
                
                ## Add DSL documentation only on first verification failure
                template_params = {
                    "verification_program": verification_program,
                    "verification_result": verification_result
                }
                
                if not is_mover_doc_sent:
                    template_params["dsl_documentation"] = self.mover_synthesizer.get_dsl_documentation()
                    is_mover_doc_sent = True
                
                verification_message = verification_message_template.render(**template_params)
                
                chat_messages.append({
                    "role": "user",
                    "content": verification_message
                })
                self._save_chat_messages(chat_file_path, chat_messages)
            
            else:
                is_correct = True
                self._write_log(chat_dir_path / "success.log", "succeeded", num_iter)
            
            num_iter += 1
        
        ## Handle failure case
        if not is_correct and num_iter >= max_iter:
            self._write_log(chat_dir_path / "fail.log", "failed", num_iter)


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Returns:
        Dictionary containing pipeline configuration
        
    Raises:
        ValueError: If required fields are missing
    """
    parser = argparse.ArgumentParser(description="Run animation pipeline with YAML configuration.")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    ## Validate required sections
    required_sections = ['paths', 'server', 'pipeline', 'animation_model', 'mover_model']
    missing_sections = [s for s in required_sections if s not in config]
    if missing_sections:
        raise ValueError(f"Missing required config sections: {', '.join(missing_sections)}")
    
    ## Validate paths section
    paths_config = config['paths']
    required_paths = ['parent_dir', 'svg_dir', 'animation_prompts_file']
    missing_paths = [p for p in required_paths if p not in paths_config]
    if missing_paths:
        raise ValueError(f"Missing required path configurations: {', '.join(missing_paths)}")
    
    ## Validate server section
    server_config = config['server']
    required_server = ['port', 'create_video']
    missing_server = [s for s in required_server if s not in server_config]
    if missing_server:
        raise ValueError(f"Missing required server configurations: {', '.join(missing_server)}")
    
    ## Validate pipeline section
    pipeline_config = config['pipeline']
    if 'max_iter' not in pipeline_config:
        raise ValueError("Missing required pipeline configuration: max_iter")
    
    ## Validate model sections
    for model_key in ['animation_model', 'mover_model']:
        model_config = config[model_key]
        required_model = ['name', 'provider']
        missing_model = [m for m in required_model if m not in model_config]
        if missing_model:
            raise ValueError(f"Missing required {model_key} configurations: {', '.join(missing_model)}")
    
    return config


def main() -> None:
    """Main entry point for the animation pipeline."""
    config = load_config()
    pipeline = AnimationPipeline(config)
    
    paths_config = config['paths']
    with open(paths_config['animation_prompts_file'], "r") as f:
        animation_prompts = json.load(f)
    
    for animation_prompt_data in animation_prompts:
        if animation_prompt_data.get("has_run", False):
            continue
            
        print(f"\n\n ---- Running {animation_prompt_data['chat_id_name']} ----\n")
        try:    
            pipeline.run_loop(animation_prompt_data)
            
        except Exception as e:
            traceback_str = str(traceback.format_exc())
            print(f"\n\n{traceback_str}")
            print("\n\nErrored out. Skipping to next animation prompt.")
            error_file = Path(paths_config['parent_dir']) / animation_prompt_data["chat_id_name"] / "error.txt"
            pipeline._write_log(error_file, "errored", 0, traceback_str)
            continue


if __name__ == "__main__":
    main()