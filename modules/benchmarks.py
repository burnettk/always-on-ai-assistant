import os
import json
from modules.deepseek import prompt as deepseek_prompt
from modules.gemini import prompt as gemini_prompt
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
        prompt_template_path_file = os.path.join(test_case_dir, "prompt_template_path.txt")

        if os.path.exists(prompt_template_path_file):
            with open(prompt_template_path_file, "r") as f:
                prompt_template_path = f.read().strip()

            with open(prompt_template_path, "r") as f:
                prompt_template = f.read()

            # Load placeholders
            context_files_path = os.path.join(test_case_dir, "context_files.txt")
            scratch_pad_path = os.path.join(test_case_dir, "scratch_pad.txt")
            natural_language_request_path = os.path.join(test_case_dir, "natural_language_request.txt")

            context_files_content = ""
            if os.path.exists(context_files_path):
                with open(context_files_path, "r") as f:
                    context_files_content = f.read()

            scratch_pad_content = ""
            if os.path.exists(scratch_pad_path):
                with open(scratch_pad_path, "r") as f:
                    scratch_pad_content = f.read()

            natural_language_request_content = ""
            if os.path.exists(natural_language_request_path):
                with open(natural_language_request_path, "r") as f:
                    natural_language_request_content = f.read()

            # Fill the template
            prompt_text = (
                prompt_template.replace("{{context_files}}", context_files_content)
                .replace("{{scratch_pad}}", scratch_pad_content)
                .replace("{{natural_language_request}}", natural_language_request_content)
            )
        else:
            with open(prompt_file, "r") as f:
                prompt_text = f.read().strip()
        test_case_dir = os.path.dirname(prompt_file)
        expected_output_file = os.path.join(test_case_dir, "expected_output.txt")

        with open(prompt_file, "r") as f:
            prompt_text = f.read().strip()

        if llm == "deepseek":
            actual_output = deepseek_prompt(prompt_text).strip()
        elif llm == "gemini":
            actual_output = gemini_prompt(prompt_text).strip()
        else:
            raise ValueError(f"Invalid LLM: {llm}")

        with open(expected_output_file, "r") as f:
            expected_output = f.read().strip()

        if actual_output == expected_output:
            print(f"✅ Test case {test_case_dir} passed.")
        else:
            print(f"❌ Test case {test_case_dir} failed.")
            print(f"  Expected: {expected_output}")
            print(f"  Actual: {actual_output}")
