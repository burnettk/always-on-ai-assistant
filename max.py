import os
import json
import litellm
import logging
from pathlib import Path
import inspect # Used by litellm.utils.function_to_dict implicitly
from typing import Dict, List, Any, Callable

from RealtimeSTT import AudioToTextRecorder

# --- Configuration ---
WAKE_WORD = "max"
LLM_MODEL = "gpt-3.5-turbo-1106"  # Model that supports tool calling. Ensure OPENAI_API_KEY is set.
# Or use other models like "openrouter/deepseek/deepseek-coder" if OPENROUTER_API_KEY is set.

CONFIG_DIR = Path.home() / ".config" / "max"
CWD_FILE = CONFIG_DIR / "cwd"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
litellm.set_verbose = False # Can be True for debugging litellm calls

# --- CWD Management ---
def _ensure_config_dir_exists():
    """Ensures the configuration directory ~/.config/max exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

def get_current_max_directory() -> Path:
    """Gets the current working directory for Max from the config file."""
    _ensure_config_dir_exists()
    if not CWD_FILE.exists():
        # Initialize with current OS working directory if file doesn't exist
        current_os_cwd = Path(os.getcwd())
        CWD_FILE.write_text(str(current_os_cwd))
        logger.info(f"Initialized Max CWD to: {current_os_cwd}")
        return current_os_cwd
    return Path(CWD_FILE.read_text().strip())

def set_current_max_directory(new_path_str: str) -> str:
    """Sets Max's current working directory in the config file."""
    _ensure_config_dir_exists()
    new_path = Path(new_path_str).expanduser() # Expand ~
    
    # Try to resolve relative paths against Max's current directory
    if not new_path.is_absolute():
        current_max_dir = get_current_max_directory()
        new_path = (current_max_dir / new_path).resolve()
    else:
        new_path = new_path.resolve()

    if not new_path.is_dir():
        return f"Error: Path '{new_path}' is not a valid directory."
    
    CWD_FILE.write_text(str(new_path))
    return f"Max's current directory changed to: {new_path}"

# --- Tool Functions ---
# Functions need type hints and docstrings for litellm.utils.function_to_dict

def change_max_directory(directory_path: str) -> str:
    """
    Changes Max's current working directory.

    Parameters
    ----------
    directory_path : str
        The absolute or relative path to the new directory.
        Relative paths are resolved against Max's current directory.
    """
    logger.info(f"Attempting to change Max directory to: {directory_path}")
    return set_current_max_directory(directory_path)

def what_is_your_name() -> str:
    """
    Responds with the assistant's name.
    """
    logger.info("Function 'what_is_your_name' called.")
    return "My name is Max."

def add_file_to_context(user_query_about_file: str) -> str:
    """
    Identifies a file based on user query and current directory listing,
    then confirms which file would be 'added' to the context.

    Parameters
    ----------
    user_query_about_file : str
        The user's description, name, or intent regarding the file to add.
    """
    logger.info(f"Function 'add_file_to_context' called with query: {user_query_about_file}")
    current_dir = get_current_max_directory()
    
    try:
        files_in_dir = [f.name for f in current_dir.iterdir() if f.is_file()]
    except OSError as e:
        logger.error(f"Error listing files in {current_dir}: {e}")
        return f"Error: Could not list files in {current_dir}."

    if not files_in_dir:
        return f"No files found in the current directory: {current_dir}."

    file_list_str = ", ".join(files_in_dir)
    prompt_for_file_selection = (
        f"The user wants to add a file. Their query was: '{user_query_about_file}'.\n"
        f"From the following list of files in the directory '{current_dir}', "
        f"which single file is the user most likely referring to?\n"
        f"Files: {file_list_str}\n"
        f"Respond with only the filename from the list. "
        f"If no suitable file is found or the choice is ambiguous, respond with the exact string 'ERROR: No suitable file found'."
    )

    logger.debug(f"Second LLM call prompt for file selection: {prompt_for_file_selection}")
    
    try:
        messages = [{"role": "user", "content": prompt_for_file_selection}]
        response = litellm.completion(model=LLM_MODEL, messages=messages) # Simple completion, no tools
        selected_file_name = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error during second LLM call for file selection: {e}")
        return "Error: Could not determine the file due to an LLM issue."

    logger.info(f"LLM response for file selection: '{selected_file_name}'")

    if selected_file_name == "ERROR: No suitable file found" or selected_file_name not in files_in_dir:
        return f"Could not confidently determine which file to add from '{current_dir}' based on your query. Files available: {file_list_str}"
    else:
        # Actual "adding" logic would go here (e.g., add to a list, copy file, etc.)
        return f"File '{selected_file_name}' from '{current_dir}' would be added to context."


# --- Assistant Core ---
AVAILABLE_FUNCTIONS: Dict[str, Callable[..., str]] = {
    "change_max_directory": change_max_directory,
    "what_is_your_name": what_is_your_name,
    "add_file_to_context": add_file_to_context,
}

TOOLS_SCHEMA = []
for func_name, func_obj in AVAILABLE_FUNCTIONS.items():
    try:
        # Ensure the function object is callable and has a docstring
        if callable(func_obj) and func_obj.__doc__:
            tool_dict = litellm.utils.function_to_dict(func_obj)
            TOOLS_SCHEMA.append({"type": "function", "function": tool_dict})
        else:
            logger.warning(f"Function {func_name} is not callable or has no docstring, skipping for tool generation.")
    except Exception as e:
        logger.error(f"Failed to convert function {func_name} to dict: {e}")

if not TOOLS_SCHEMA:
    logger.error("No tools were successfully generated. Assistant may not function correctly.")
    # Potentially exit or raise an error if no tools can be used.

RECORDER: AudioToTextRecorder = None # Global recorder instance

def process_transcribed_text(text: str):
    global RECORDER
    logger.info(f"\n🎤 Heard: {text}")

    if WAKE_WORD.lower() not in text.lower():
        logger.info(f"🤖 Not '{WAKE_WORD}' - ignoring")
        return

    if RECORDER:
        RECORDER.stop() # Pause STT while processing

    try:
        messages = [{"role": "user", "content": text}]
        
        logger.debug(f"Sending to LLM with tools. Messages: {messages}, Tools: {TOOLS_SCHEMA}")
        
        response = litellm.completion(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            logger.info(f"LLM wants to call tools: {tool_calls}")
            # For simplicity, this example handles one tool call.
            # OpenAI and some models support parallel tool calls.
            # If multiple tool_calls, you might need to loop and aggregate results
            # or make subsequent calls to the LLM with tool responses.
            
            for tool_call in tool_calls: # Iterate in case of parallel calls
                function_name = tool_call.function.name
                function_args_str = tool_call.function.arguments
                
                if function_name in AVAILABLE_FUNCTIONS:
                    try:
                        function_args = json.loads(function_args_str)
                        function_to_call = AVAILABLE_FUNCTIONS[function_name]
                        
                        logger.info(f"Calling function: {function_name} with args: {function_args}")
                        result = function_to_call(**function_args)
                        logger.info(f"🤖 Function {function_name} result: {result}")
                        print(f"🤖 Max: {result}") # Print/speak result
                    except json.JSONDecodeError:
                        logger.error(f"Error decoding JSON arguments for {function_name}: {function_args_str}")
                        print(f"🤖 Max: Error processing arguments for {function_name}.")
                    except TypeError as e:
                        logger.error(f"TypeError calling {function_name} with {function_args}: {e}")
                        print(f"🤖 Max: Error with arguments for function {function_name}.")
                    except Exception as e:
                        logger.error(f"Error executing function {function_name}: {e}")
                        print(f"🤖 Max: An error occurred while executing {function_name}.")
                else:
                    logger.warning(f"LLM requested unknown function: {function_name}")
                    print(f"🤖 Max: I tried to use a tool I don't have: {function_name}")
        
        elif response_message.content:
            # Handle cases where the LLM responds directly without a tool_call
            direct_response = response_message.content.strip()
            logger.info(f"LLM direct response: {direct_response}")
            print(f"🤖 Max: {direct_response}")
        else:
            logger.info("LLM response had no tool_calls and no content.")
            print(f"🤖 Max: I'm not sure how to respond to that.")

    except litellm.exceptions.APIConnectionError as e:
        logger.error(f"LLM API Connection Error: {e}")
        print("🤖 Max: Sorry, I'm having trouble connecting to my brain (API connection error).")
    except litellm.exceptions.AuthenticationError as e:
        logger.error(f"LLM Authentication Error: {e}. Ensure API key is correct and valid.")
        print("🤖 Max: Sorry, there's an issue with my authentication (API key error).")
    except Exception as e:
        logger.error(f"Error processing text with LLM: {e}", exc_info=True)
        print(f"🤖 Max: Oops, something went wrong: {str(e)}")
    finally:
        if RECORDER:
            RECORDER.start() # Resume STT

def main():
    global RECORDER
    # Ensure API keys are set in environment variables, e.g., OPENAI_API_KEY
    # For OpenRouter, set OPENROUTER_API_KEY and use model="openrouter/..."
    if not os.getenv("OPENAI_API_KEY") and "gpt-" in LLM_MODEL: # Basic check
        logger.warning("OPENAI_API_KEY environment variable not set. LLM calls may fail.")
        print("Warning: OPENAI_API_KEY not set. Max might not be able to think.")
    
    # Initialize CWD
    _ensure_config_dir_exists()
    initial_cwd = get_current_max_directory()
    logger.info(f"Max assistant started. Current Max directory: {initial_cwd}")
    print(f"Max assistant started. Current Max directory: {initial_cwd}")
    print(f"Supported tools: {list(AVAILABLE_FUNCTIONS.keys())}")


    RECORDER = AudioToTextRecorder(
        spinner=False,
        post_speech_silence_duration=1.0, # shorter silence for quicker interaction
        compute_type="float32", # as per main_typer_assistant.py
        model="small.en", # Balance speed and accuracy
        # beam_size=5, # Can improve accuracy at cost of speed
        language="en",
        print_transcription_time=False, # Less verbose
    )

    print(f"\n🎤 Speak now (say '{WAKE_WORD}' followed by your command)... (press Ctrl+C to exit)")
    try:
        while True:
            RECORDER.text(process_transcribed_text)
    except KeyboardInterrupt:
        logger.info("Max assistant stopped by user.")
        print("\n🤖 Max: Goodbye!")
    finally:
        if RECORDER:
            RECORDER.stop() # Ensure recorder is stopped cleanly


if __name__ == "__main__":
    main()
