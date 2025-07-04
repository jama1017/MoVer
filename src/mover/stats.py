import json
import heapq
import argparse
import yaml
from pathlib import Path
from typing import Tuple, List

parser = argparse.ArgumentParser()
parser.add_argument("configs", type=str, nargs='+', help="List of YAML config files to process.")

args, unknown = parser.parse_known_args()

## Helper function to extract paths from config
def get_paths_from_config(config_file: str) -> Tuple[str, str]:
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    paths_config = config['paths']
    return (
        paths_config['parent_dir'],
        paths_config['animation_prompts_file']
    )

def format_stats(total_input: int, chat_dir: str, error_count: int, total: int, num_more_than_one: int, num_iterations: int, num_fail: int, iterations_data: Tuple[int, int, int, int] = None, is_total: bool = False) -> str:
    """Helper function to format statistics in a consistent format.
    
    Args:
        total_input: Number of total input prompts
        chat_dir: Name of chat directory (or "Total stats" for totals)
        error_count: Number of errors
        total: Total number of prompts processed
        num_more_than_one: Number of prompts that took more than one attempt
        num_iterations: Number of prompts that succeeded after multiple attempts
        num_fail: Number of failed prompts
        iterations_data: Tuple of (sum_iterations, min_iter, max_iter) for calculating average
        is_total: Whether this is printing total stats
        
    Returns:
        Formatted string containing the statistics
    """
    lines = []
    if not is_total:
        lines.extend([
            f"\n\nTotal input: {total_input}",
            f"Chat dir:    {Path(chat_dir).name}"
        ])
    else:
        lines.extend([
            "\n------------",
            "Total stats\n",
            f"Total input: {total_input}"
        ])
        
    lines.extend([
        f"error:       {error_count}",
        f"Total:       {total}",
        f"pass@0:      {(total - num_more_than_one)} ({(total - num_more_than_one) / total * 100})",
        f"pass@1+:     {num_iterations} ({num_iterations / total * 100})",
        f"fail:        {num_fail} ({num_fail / total * 100})"
    ])
    
    if iterations_data and iterations_data[1] > 0:  # If we have iteration data and at least one multi-attempt success
        avg_iters = iterations_data[0] / iterations_data[1]
        lines.append(f"avg. iters.: {avg_iters} ({iterations_data[2]}-{iterations_data[3]})")
    else:
        lines.append("avg. iters.: N/A")
        
    return "\n".join(lines)


def get_stats(parent_dirs: List[str], animation_prompts_files: List[str], chat_id_prefix: str = None) -> str:
    total_prompts = []
    total_input_prompts = []
    total_num_more_than_one = []
    total_num_fail = []
    total_num_iterations = []
    total_error_count = []

    for parent_dir in parent_dirs:
        with open(animation_prompts_files[parent_dirs.index(parent_dir)], 'r') as file:
            chat_prompts = json.load(file)
        chat_id_names = [chat_prompt["chat_id_name"] for chat_prompt in chat_prompts]
        total_input_prompts.append(len(chat_prompts))
        
        ## for each subdirectory in parent_dir, count the number of html files
        html_file_count = []
        parent_path = Path(parent_dir)
        for subdir in parent_path.iterdir():
            if not subdir.is_dir():
                continue
            
            error_file = list(subdir.glob('error.txt'))
            subdir_name = subdir.name
            error_count = len(error_file)
            
            ## read the json chat_file named the same as the subdir_name and count the number of objects with "role": "assistant"
            chat_file = subdir / f"{subdir_name}.json"
            num_assistant_messages = 0
            if chat_file.exists():
                with open(chat_file, 'r') as f:
                    chat_data = json.load(f)
                    num_assistant_messages = sum(1 for message in chat_data if message.get("role") == "assistant")
            else:
                error_count = 1
                
            if subdir_name in chat_id_names:
                html_file_count.append((subdir_name, num_assistant_messages, error_count))
        
        ## sort the subdirectories by its name
        html_file_count = sorted(html_file_count, key=lambda x: x[0].split("_")[2:], reverse=False)

        ## print the html file count in a table format
        num_more_than_one = 0
        num_fail = 0
        count_num_iterations = []
        num_iterations = 0
        filtered_html_file_count = []
        num_error_count = 0
        for subdir, count, error_count in html_file_count:
            if chat_id_prefix is not None:
                if not subdir.startswith(chat_id_prefix):
                    continue
            
            if error_count > 0:
                num_error_count += 1
                continue
            
            filtered_html_file_count.append((subdir, count, error_count))
            success_log_file = Path(parent_dir) / subdir / "success.log"
            fail_log_file = Path(parent_dir) / subdir / "fail.log"
            
            if count > 1:
                num_more_than_one += 1
            
            if success_log_file.exists():
                ## prompt took more than one attempt
                if count > 1:
                    num_iterations += 1
                    count_num_iterations.append(count)
                    
            elif fail_log_file.exists():
                num_fail += 1
            
        html_file_count = filtered_html_file_count
        total_prompts.append(len(html_file_count))
        total_num_more_than_one.append(num_more_than_one)
        total_num_fail.append(num_fail)
        total_num_iterations.append((sum(count_num_iterations), num_iterations, min(count_num_iterations, default=0), heapq.nlargest(1, count_num_iterations)[0] if count_num_iterations else 0))
        total_error_count.append(num_error_count)
        
    output = []
    for i in range(len(parent_dirs)):
        stats = format_stats(
            total_input=total_input_prompts[i],
            chat_dir=parent_dirs[i],
            error_count=total_error_count[i],
            total=total_prompts[i],
            num_more_than_one=total_num_more_than_one[i],
            num_iterations=total_num_iterations[i][1],
            num_fail=total_num_fail[i],
            iterations_data=total_num_iterations[i]
        )
        output.append(stats)
    
    # Only show total stats if there is more than one config
    if len(parent_dirs) > 1:
        num_total_prompts = sum(total_prompts)
        num_total_more_than_one = sum(total_num_more_than_one)
        num_total_50 = sum(total_num_fail)
        count_num_total_iterations = sum([(x[0]) for x in total_num_iterations])
        num_total_iterations = sum([x[1] for x in total_num_iterations])
        min_iterations = min([x[2] for x in total_num_iterations if x[2] > 0], default=0)
        max_iterations = max([x[3] for x in total_num_iterations], default=0)
        
        total_stats = format_stats(
            total_input=sum(total_input_prompts),
            chat_dir="Total stats",
            error_count=sum(total_error_count),
            total=num_total_prompts,
            num_more_than_one=num_total_more_than_one,
            num_iterations=num_total_iterations,
            num_fail=num_total_50,
            iterations_data=(count_num_total_iterations, num_total_iterations, min_iterations, max_iterations),
            is_total=True
        )
        output.append(total_stats)
    
    return "\n".join(output)

if __name__ == "__main__":
    ## Extract paths from config files
    parent_dirs = []
    animation_prompts_files = []
    for config_file in args.configs:
        parent_dir, prompts_file = get_paths_from_config(config_file)
        parent_dirs.append(parent_dir)
        animation_prompts_files.append(prompts_file)
        
    stats_output = get_stats(parent_dirs, animation_prompts_files)
    print(stats_output)