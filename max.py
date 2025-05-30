
import logging
# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

import litellm
# Reduce litellm verbosity
litellm.suppress_debug_info = True

import json
import os
import inspect
import sys
import argparse
from pydantic import BaseModel, create_model, Field
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
from RealtimeSTT import AudioToTextRecorder
from elevenlabs import play, Voice, VoiceSettings
from elevenlabs.client import ElevenLabs
import subprocess
import shutil

# --- Configuration ---
MAGIC_QUERY_PARAM_NAME = "full_user_query"
CWD_FILE = Path.home() / ".config" / "max" / "cwd"
ASSISTANT_NAME = "Max"
WAKE_WORD = "max"  # Case-insensitive check

# LITELLM_MODEL = os.getenv("LITELLM_MODEL", "gemini/gemini-2.0-flash-lite")
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "gemini/gemini-2.5-flash-preview-05-20")

ELEVENLABS_API_KEY = os.getenv("ELEVEN_API_KEY")
# LiteLLM uses various keys like OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
# Ensure the relevant key for LITELLM_MODEL is set in the environment.

USE_ELEVENLABS_TTS = False  # Global flag, set by CLI
SAY_COMMAND_AVAILABLE = None # To cache whether 'say' command exists




# --- ElevenLabs Client ---
elevenlabs_client = None
# Initialization will be handled in cli_main based on --elevenlabs flag

# --- Tool Definition and Schema Generation ---
_tool_functions: Dict[str, Callable] = {}

def tool_function(func: Callable) -> Callable:
    """Decorator to mark functions as tools and register them."""
    func._is_tool = True
    _tool_functions[func.__name__] = func
    logger.debug(f"Registered tool: {func.__name__}")
    return func

def get_function_schema(func: Callable) -> Dict[str, Any]:
    """Generates a JSON schema for a function's parameters using Pydantic."""
    
    func_name = func.__name__
    description = inspect.getdoc(func) or f"Executes the {func_name} action."
    
    sig = inspect.signature(func)
    param_fields = {}
    for name, param in sig.parameters.items():
        if name == MAGIC_QUERY_PARAM_NAME:
            continue # Skip magic parameter for schema generation

        annotation = param.annotation
        if annotation == inspect.Parameter.empty:
            raise ValueError(f"Missing type annotation for parameter '{name}' in function '{func_name}'")
        
        default_value = param.default if param.default != inspect.Parameter.empty else ...
        param_fields[name] = (annotation, default_value)

    # create_model handles empty param_fields correctly, 
    # resulting in a schema with 'type': 'object' and 'properties': {}
    parameters_model = create_model(
        f"{func_name}Params", 
        **param_fields,
        __base__=BaseModel
    )
    parameters_json_schema = parameters_model.model_json_schema()
    
    schema = {
        "type": "function",
        "function": {
            "name": func_name,
            "description": description,
            "parameters": parameters_json_schema
        }
    }
    return schema

def get_all_tool_schemas() -> List[Dict[str, Any]]:
    """Gets schemas for all registered tool functions by checking the _tool_functions registry."""
    schemas = []
    for name, func in _tool_functions.items():
        try:
            schemas.append(get_function_schema(func))
        except ValueError as e:
            logger.error(f"Skipping tool {name} due to schema generation error: {e}")
    return schemas

# --- Helper Functions ---
def get_current_directory() -> str:
    """Reads the current working directory from the CWD_FILE."""
    try:
        if not CWD_FILE.exists():
            default_cwd = str(Path.home())
            CWD_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CWD_FILE, "w") as f:
                f.write(default_cwd)
            logger.info(f"CWD file not found. Created and set to default: {default_cwd}")
            return default_cwd
        
        with open(CWD_FILE, "r") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error accessing CWD file {CWD_FILE}: {e}. Falling back to home directory.")
        return str(Path.home())

def check_say_command():
    """Checks if the 'say' command is available and caches the result."""
    global SAY_COMMAND_AVAILABLE
    if SAY_COMMAND_AVAILABLE is None: # Check only once
        SAY_COMMAND_AVAILABLE = shutil.which("say") is not None
        if SAY_COMMAND_AVAILABLE:
            logger.info("TTS: 'say' command is available.")
        else:
            logger.warning("TTS: 'say' command not found. Console print will be used if ElevenLabs is not active or fails.")

def speak(text: str):
    """Speaks the given text using ElevenLabs (if enabled and available) or 'say' command, otherwise prints to console."""
    global USE_ELEVENLABS_TTS, SAY_COMMAND_AVAILABLE

    if USE_ELEVENLABS_TTS:
        if elevenlabs_client:
            try:
                logger.info(f"Attempting to speak with ElevenLabs: \"{text}\"")
                voice_id = os.getenv("ELEVENLABS_VOICE_ID", 'pNInz6obpgDQGcFmaJgB') # Default example voice
                audio = elevenlabs_client.generate(
                    text=text,
                    voice=Voice(
                        voice_id=voice_id,
                        settings=VoiceSettings(stability=0.7, similarity_boost=0.6, style=0.0, use_speaker_boost=True)
                    ),
                    model=os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2") # Fast model
                )
                play(audio)
                logger.info(f"Spoke with ElevenLabs: \"{text}\"")
                return
            except Exception as e:
                logger.error(f"ElevenLabs TTS failed: {e}. Falling back.")
        else:
            logger.warning("ElevenLabs TTS requested via --elevenlabs, but client not available (e.g., API key missing). Falling back.")

    # Fallback to 'say' command or print
    # SAY_COMMAND_AVAILABLE should have been initialized by check_say_command() in cli_main.
    # If it's somehow None here, check_say_command() inside would log a warning if 'say' is not found.
    if SAY_COMMAND_AVAILABLE is None:
        check_say_command() # Defensive check, normally already called.

    if SAY_COMMAND_AVAILABLE:
        try:
            # Using capture_output to prevent 'say' from writing to stdout/stderr unless it's an error we want to log.
            # However, 'say' typically doesn't output on success.
            subprocess.run(["say", text], check=True, capture_output=True, text=True)
            logger.info(f"TTS (say): \"{text}\"")
            return
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"TTS with 'say' command failed: {e}. Falling back to print.")
    
    # Final fallback: print to console
    print(f"{ASSISTANT_NAME}: {text}")
    logger.info(f"TTS (console print): \"{text}\"")

# --- Tool Functions ---
@tool_function
def change_directory(new_path: str):
    """
    Changes Max's current working directory.
    Use this to navigate the file system.
    The path should be an absolute path or relative to the user's home directory (e.g., ~/projects).
    """
    try:
        expanded_path = Path(new_path).expanduser().resolve()
        if not expanded_path.is_dir():
            return f"Error: '{expanded_path}' is not a valid directory or does not exist."
        
        CWD_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CWD_FILE, "w") as f:
            f.write(str(expanded_path))
        logger.info(f"Changed CWD to: {expanded_path}")
        return f"Okay, I've changed the current directory to {expanded_path}."
    except Exception as e:
        logger.error(f"Error changing directory to {new_path}: {e}")
        return f"Sorry, I couldn't change the directory. Error: {str(e)}"

@tool_function
def what_is_your_name():
    """
    Responds with the assistant's name.
    Call this function if the user asks for your name.
    """
    return f"My name is {ASSISTANT_NAME}."

@tool_function
def add_file(full_user_query: str): # Renamed parameter
    """
    Identifies a file to add based on user query and files in the current directory, then simulates adding it.
    This function lists files in the current directory.
    Then, it uses an LLM to determine which specific file the user wants to add based on their query and the list of available files.
    Finally, it confirms the action of adding the selected file. This can be used to create new files or acknowledge existing ones in context of "adding".
    """
    current_dir_path_str = get_current_directory()
    current_dir = Path(current_dir_path_str)
    
    if not current_dir.is_dir():
        return f"Error: The current directory '{current_dir_path_str}' is not valid."

    try:
        # Use git ls-files to get a list of tracked files in the current directory.
        # This respects .gitignore and only lists files (not directories).
        # The command is run in current_dir_path_str, so paths are relative to it.
        process = subprocess.run(
            ["git", "ls-files"], 
            cwd=current_dir_path_str,
            capture_output=True,
            text=True,
            check=True  # Will raise CalledProcessError if git command fails
        )
        
        relative_file_paths = process.stdout.splitlines()
        
        items_in_dir = []
        for rel_path_str in relative_file_paths:
            # A path like "file.txt" has parent ".", "subdir/file.txt" has parent "subdir"
            # We only want files directly in the current directory.
            path_obj = Path(rel_path_str)
            if path_obj.parent == Path('.'): 
                # Ensure it's a file, not a submodule entry or other non-file tracked item
                if (current_dir / path_obj).is_file():
                    items_in_dir.append(path_obj.name)
        
    except subprocess.CalledProcessError as e:
        # This typically means current_dir is not a git repo or git command failed.
        if "not a git repository" in e.stderr.lower():
            logger.warning(f"Directory {current_dir_path_str} is not a git repository. Cannot use git ls-files.")
            return f"The current directory '{current_dir_path_str}' is not a Git repository. I can only list files from Git-tracked directories for this command."
        logger.error(f"Error running 'git ls-files' in {current_dir_path_str}: {e.stderr}")
        return f"Sorry, I couldn't list files using Git. Git command failed: {e.stderr}"
    except FileNotFoundError: # git command itself not found
        logger.error("'git' command not found. Cannot list files using git ls-files.")
        return "Sorry, the 'git' command is not installed or not in PATH. I need it to list files."
    except Exception as e: # Other unexpected errors
        logger.error(f"Unexpected error listing files with git ls-files in {current_dir_path_str}: {e}")
        return f"An unexpected error occurred while trying to list files using Git: {str(e)}"

    items_list_str = ", ".join(items_in_dir) if items_in_dir else "none"
    prompt_for_file_selection = (
        f"The user wants to add a file to the context. Their original request was: '{full_user_query}'.\n"
        f"Respond with only the full file name, which may not exactly match the user request. If completely unsure, respond with 'UNCLEAR'."
        f"Based on the user's request and the available files (following), what is the exact name of the single file they most likely mean? "
        # f"If the user seems to be referring to a new file that doesn't exist, provide the name for the new file. "
        f"The files currently available in '{current_dir_path_str}' are: [{items_list_str}].\n"
    )
    
    logger.info(f"Asking LLM to select file with prompt: {prompt_for_file_selection}")

    try:
        messages_for_file_selection = [{"role": "user", "content": prompt_for_file_selection}]
        response = litellm.completion(
            model=LITELLM_MODEL,
            messages=messages_for_file_selection
        )
        
        selected_item_name = response.choices[0].message.content.strip()
        logger.info(f"LLM suggested item name: '{selected_item_name}'")

        if selected_item_name.upper() == "UNCLEAR" or not selected_item_name:
            return f"I'm not sure which file you mean from your request: '{full_user_query}'. Could you be more specific?"

        # Validate that the selected item is one of the files found by git ls-files
        if selected_item_name not in items_in_dir:
            logger.warning(f"LLM selected '{selected_item_name}', which is not in the git ls-files list for {current_dir_path_str}. Files found: {items_in_dir}")
            if not items_in_dir:
                return f"I couldn't find any files in the current directory '{current_dir_path_str}' tracked by Git. So, I cannot add '{selected_item_name}'."
            return (f"I'm sorry, but '{selected_item_name}' is not among the files I found in the current directory "
                    f"('{current_dir_path_str}'). Please choose from the available files: {', '.join(items_in_dir)}.")
        
        # If selected_item_name is in items_in_dir, it's an existing file in the current directory.
        action_taken_msg = f"Okay, I've noted your intent to 'add' the file '{selected_item_name}' from {current_dir_path_str}."
        
        logger.info(f"Simulated action for add_file: {action_taken_msg}")
        return action_taken_msg

    except Exception as e:
        logger.error(f"Error during LLM call for file selection or processing: {e}")
        return f"Sorry, I had trouble determining which file to add. Error: {str(e)}"

# --- STT and Main Assistant Logic ---

def handle_llm_interaction(user_query: str) -> Optional[str]:
    """
    Handles the core interaction with the LLM, including tool calls.
    Returns the final textual response from the LLM, or None if no specific response.
    """
    logger.info(f"Core LLM interaction for query: \"{user_query}\"")
    ensure_tools_loaded()

    if not _cached_tool_schemas or not _cached_function_dispatch_table:
        logger.warning("No tools available. Attempting basic chat.")
        try:
            messages = [{"role": "user", "content": user_query}]
            response = litellm.completion(model=LITELLM_MODEL, messages=messages)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in basic LLM completion: {e}")
            return "Sorry, I had trouble processing that."

    messages = [{"role": "user", "content": user_query}]
    try:
        response_obj = litellm.completion(
            model=LITELLM_MODEL,
            messages=messages,
            tools=_cached_tool_schemas,
            tool_choice="auto"
        )
        
        response_message = response_obj.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            logger.info(f"LLM returned tool calls: {tool_calls}")
            messages.append(response_message) # Add assistant's decision to call tool(s)

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                if function_name not in _cached_function_dispatch_table:
                    logger.error(f"LLM requested unknown function: {function_name}")
                    messages.append({
                        "tool_call_id": tool_call.id, "role": "tool", "name": function_name,
                        "content": f"Error: Function {function_name} is not recognized."
                    })
                    continue

                function_to_call = _cached_function_dispatch_table[function_name]
                try:
                    args_str = tool_call.function.arguments
                    args_dict = json.loads(args_str)
                    
                    func_sig = inspect.signature(function_to_call)
                    if MAGIC_QUERY_PARAM_NAME in func_sig.parameters:
                        args_dict[MAGIC_QUERY_PARAM_NAME] = user_query
                        logger.info(f"Injected magic param '{MAGIC_QUERY_PARAM_NAME}' for {function_name}")
                    
                    logger.info(f"Calling tool: {function_name} with args: {args_dict}")
                    tool_response_content = function_to_call(**args_dict)
                except Exception as e:
                    logger.error(f"Error executing tool {function_name} with args '{args_str}': {e}")
                    tool_response_content = f"Error executing {function_name}: {str(e)}"
                
                logger.info(f"Tool {function_name} response: {tool_response_content}")
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(tool_response_content) # Ensure string
                })
            
            logger.info("Sending tool responses to LLM for final summarization.")
            final_response_obj = litellm.completion(model=LITELLM_MODEL, messages=messages)
            return final_response_obj.choices[0].message.content
        else: # No tool_calls
            logger.info("LLM did not return any tool calls. Using direct response.")
            return response_message.content # This could be None if LLM sends empty content

    except Exception as e:
        logger.error(f"Error in LLM completion or tool processing: {e}", exc_info=True)
        return "Sorry, I encountered an error while processing your request."
    # Fallback, should ideally be unreachable if all paths above return something.
    return None


_recorder_instance: Optional[AudioToTextRecorder] = None
_cached_tool_schemas: Optional[List[Dict[str, Any]]] = None
_cached_function_dispatch_table: Optional[Dict[str, Callable]] = None

def ensure_tools_loaded():
    """Loads tool schemas and dispatch table if not already loaded."""
    global _cached_tool_schemas, _cached_function_dispatch_table
    if _cached_tool_schemas is None or _cached_function_dispatch_table is None:
        _cached_tool_schemas = get_all_tool_schemas()
        _cached_function_dispatch_table = {
            tool_schema["function"]["name"]: _tool_functions[tool_schema["function"]["name"]]
            for tool_schema in _cached_tool_schemas
        }
        logger.info(f"Tools loaded: {[name for name in _cached_function_dispatch_table.keys()]}")

def process_transcription(transcribed_text: str):
    """Processes transcribed text, checks for wake word, and interacts with LLM and tools."""
    logger.info(f"Processing transcription: \"{transcribed_text}\"")
    
    if WAKE_WORD.lower() not in transcribed_text.lower():
        logger.debug(f"Wake word '{WAKE_WORD}' not detected.")
        return
    
    try:
        query_parts = transcribed_text.lower().split(WAKE_WORD.lower(), 1)
        user_query = query_parts[1].strip() if len(query_parts) > 1 and query_parts[1].strip() else transcribed_text
    except Exception:
        user_query = transcribed_text # Fallback
    
    logger.info(f"User query for LLM (from transcription): \"{user_query}\"")

    final_content = handle_llm_interaction(user_query)

    if final_content: # This includes error messages from handle_llm_interaction
        speak(final_content)
    elif final_content == "": # Explicitly empty string from LLM (e.g. after tools, no summary)
        speak("Done.")
    # If final_content is None (e.g. LLM gave no tools, no direct content, and no error), 
    # nothing is spoken, aligning with original behavior of ignoring if no tool selected and no direct answer.

def process_text_query(user_query: str):
    """Processes a text query, interacts with LLM/tools, and prints the response."""
    logger.info(f"Processing text query: \"{user_query}\"")
    
    final_content = handle_llm_interaction(user_query)

    if final_content: # This includes error messages from handle_llm_interaction
        print(f"{ASSISTANT_NAME}: {final_content}")
    elif final_content == "": # Explicitly empty string from LLM
        print(f"{ASSISTANT_NAME}: Done.")
    else: # final_content is None (no tools, no direct content, no error)
        print(f"{ASSISTANT_NAME}: I processed your request, but there was no specific information to return.")

def main_assistant_loop():
    """Initializes STT and runs the main loop for voice interaction."""
    global _recorder_instance
    # Warnings about ElevenLabs API key or 'say' command availability are handled
    # in cli_main or by the speak() function's fallbacks.

    ensure_tools_loaded() # Load tools at startup

    recorder_config = {
        "spinner": False, "model": os.getenv("STT_MODEL", "tiny.en"), "language": "en",
        "post_speech_silence_duration": float(os.getenv("STT_SILENCE_DURATION", 1.2)),
        "beam_size": int(os.getenv("STT_BEAM_SIZE", 5)),
        "print_transcription_time": False,
        "realtime_processing_pause": 0.2,
    }
    logger.info(f"Initializing RealtimeSTT with config: {recorder_config}")
    
    try:
        _recorder_instance = AudioToTextRecorder(**recorder_config)
    except Exception as e:
        logger.error(f"Failed to initialize AudioToTextRecorder: {e}. Voice assistant cannot start.")
        print(f"Error: Could not start audio recorder. Ensure microphone is available and permissions are set. Details: {e}")
        return

    # Re-assert logging configuration for the application's logger
    # This helps ensure our logs appear as intended, even if libraries modify root logging settings.
    logger.setLevel(logging.INFO) # Set desired level for our logger

    # Remove any handlers that might have been attached to this specific logger by other libraries,
    # or to reset its handlers to a known state.
    if logger.hasHandlers():
        for handler in logger.handlers[:]: # Iterate over a copy of the list
            logger.removeHandler(handler)

    # Add our desired handler
    app_handler = logging.StreamHandler() # Defaults to sys.stderr
    # Use the same format as basicConfig
    app_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    app_handler.setFormatter(app_formatter)
    # The handler will respect the logger's level (INFO), 
    # so no need to setLevel on handler unless a more restrictive level is needed for this specific handler.
    logger.addHandler(app_handler)

    # Prevent messages from this logger from propagating to the root logger.
    # This is important if the root logger's handlers have been altered by libraries.
    logger.propagate = False
        
    logger.info(f"'{ASSISTANT_NAME}' is listening... Say '{WAKE_WORD}' followed by your command.")
    speak(f"Hello! I'm {ASSISTANT_NAME}. How can I assist you today?")

    try:
        while True:
            transcribed_text = _recorder_instance.text()
            if transcribed_text: # Process if not empty
                process_transcription(transcribed_text)
    except KeyboardInterrupt:
        logger.info(f"'{ASSISTANT_NAME}' assistant stopped by user.")
        speak("Goodbye!")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
        speak("An unexpected error occurred. Shutting down.")
    finally:
        logger.info("Cleaning up assistant resources.")
        # RealtimeSTT recorder usually cleans itself up or doesn't require explicit stop for this usage.

# --- CLI ---
def cli_main():
    parser = argparse.ArgumentParser(description=f"{ASSISTANT_NAME} Voice Assistant")
    parser.add_argument(
        "--tools",
        action="store_true",
        help="Dump the tool call metadata (JSON schemas for functions) and exit.",
    )
    parser.add_argument(
        "--elevenlabs",
        action="store_true",
        help="Use ElevenLabs for Text-to-Speech instead of the default 'say' command (macOS).",
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Process a text query directly, print the response, and exit. Bypasses voice input/output.",
    )
    args = parser.parse_args()

    global USE_ELEVENLABS_TTS
    global elevenlabs_client # Allow modification of the global client

    if args.elevenlabs:
        USE_ELEVENLABS_TTS = True
        logger.info("ElevenLabs TTS explicitly enabled via --elevenlabs flag.")
        if ELEVENLABS_API_KEY:
            try:
                elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
                logger.info("ElevenLabs client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize ElevenLabs client: {e}. ElevenLabs TTS will not be available.")
                USE_ELEVENLABS_TTS = False # Disable if client fails to init
        else:
            logger.warning("ElevenLabs TTS enabled via flag, but ELEVEN_API_KEY is not set. ElevenLabs TTS will not be available.")
            USE_ELEVENLABS_TTS = False # Disable if no API key
    
    # Initialize say command check early, so its availability is known before the first speak() call.
    check_say_command()

    if args.tools:
        ensure_tools_loaded() # Ensure tools are discovered
        print(json.dumps(_cached_tool_schemas, indent=2))
        sys.exit(0)
    
    if args.query:
        # Ensure CWD file and its directory exist before processing query
        if not CWD_FILE.parent.exists():
            CWD_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not CWD_FILE.exists():
            get_current_directory() # This will create it with default if not present
        process_text_query(args.query)
        sys.exit(0)
    else:
        # Ensure CWD file and its directory exist before starting assistant
        if not CWD_FILE.parent.exists():
            CWD_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not CWD_FILE.exists():
            get_current_directory() # This will create it with default if not present
            
        main_assistant_loop()

if __name__ == "__main__":
    cli_main()
