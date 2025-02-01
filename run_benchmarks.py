import os
import json
from modules.openrouter import prompt as openrouter_prompt
# from modules.deepseek import prompt as deepseek_prompt
# from modules.gemini import prompt as gemini_prompt
import glob

def run_benchmarks(llm: str):
    """
    Run benchmarks for the specified LLM.

    Args:
        llm: The LLM to use ('deepseek' or 'gemini').
    """
    benchmark_dir = "bench"
    prompt_files = glob.glob(os.path.join(benchmark_dir, "*/prompt.txt"))

    for prompt_file in prompt_files:
        test_case_dir = os.path.dirname(prompt_file)
        expected_output_file = os.path.join(test_case_dir, "expected_output.txt")

        with open(prompt_file, "r") as f:
            prompt_text = f.read().strip()

        actual_output = openrouter_prompt(prompt_text).strip()
        # if llm == "deepseek":
        #     actual_output = deepseek_prompt(prompt_text).strip()
        # elif llm == "gemini":
        #     actual_output = gemini_prompt(prompt_text).strip()
        # else:
        #     raise ValueError(f"Invalid LLM: {llm}")

        with open(expected_output_file, "r") as f:
            expected_output = f.read().strip()

        if actual_output == expected_output:
            print(f"✅ Test case {test_case_dir} passed.")
        else:
            print(f"❌ Test case {test_case_dir} failed.")
            print(f"  Expected: {expected_output}")
            print(f"  Actual: {actual_output}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, required=True, help="LLM to use ('deepseek' or 'gemini')")
    args = parser.parse_args()
    run_benchmarks(args.llm)
