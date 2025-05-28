
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
CWD_FILE = Path.home() / ".config" / "max" / "cwd"
ASSISTANT_NAME = "Max"
WAKE_WORD = "max"  # Case-insensitive check
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "gemini/gemini-2.0-flash-lite")
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
    sig = inspect.signature(func)
    param_fields = {}
    for name, param in sig.parameters.items():
        annotation = param.annotation
        if annotation == inspect.Parameter.empty:
            raise ValueError(f"Missing type annotation for parameter '{name}' in function '{func.__name__}'")
        
        default_value = param.default if param.default != inspect.Parameter.empty else ...
        param_fields[name] = (annotation, default_value)

    parameters_model = create_model(
        f"{func.__name__}Params", 
        **param_fields,
        __base__=BaseModel
    )
    
    schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": inspect.getdoc(func) or f"Executes the {func.__name__} action.", # Default description
            "parameters": parameters_model.model_json_schema()
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
def add_file(user_query_for_filename: str):
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
        # List both files and directories, as "add file" might refer to a directory in some contexts
        items_in_dir = [item.name for item in current_dir.iterdir()]
    except Exception as e:
        logger.error(f"Error listing items in {current_dir_path_str}: {e}")
        return f"Sorry, I couldn't list items in the current directory. Error: {str(e)}"

    items_list_str = ", ".join(items_in_dir) if items_in_dir else "none"
    prompt_for_file_selection = (
        f"The user wants to add a file or item. Their original request was: '{user_query_for_filename}'.\n"
        f"The items currently available in '{current_dir_path_str}' are: [{items_list_str}].\n"
        f"Based on the user's request and the available items, what is the exact name of the single file or directory they most likely mean? "
        f"If the user seems to be referring to a new file that doesn't exist, provide the name for the new file. "
        f"Respond with only the name. If completely unsure, respond with 'UNCLEAR'."
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
            return f"I'm not sure which file or item you mean from your request: '{user_query_for_filename}'. Could you be more specific?"
        
        target_path = current_dir / selected_item_name
        
        # Simulate "adding" the file. For this exercise, we'll just confirm.
        # In a real scenario, this could mean `target_path.touch()`, `git add`, etc.
        if target_path.exists():
            action_taken_msg = f"The item '{selected_item_name}' already exists in {current_dir_path_str}. I've noted your intent to 'add' it."
        else:
            # Simulate creating it for the purpose of "adding"
            # target_path.touch() # Uncomment to actually create an empty file
            action_taken_msg = f"Okay, I'll consider '{selected_item_name}' added in {current_dir_path_str}. If it's a new file, it would be created here."
        
        logger.info(f"Simulated action for add_file: {action_taken_msg}")
        return action_taken_msg

    except Exception as e:
        logger.error(f"Error during LLM call for file selection or processing: {e}")
        return f"Sorry, I had trouble determining which file to add. Error: {str(e)}"

# --- STT and Main Assistant Logic ---
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

    # speak("Yes?") # Acknowledge wake word
    
    try:
        # Attempt to extract query part after the wake word
        query_parts = transcribed_text.lower().split(WAKE_WORD.lower(), 1)
        user_query = query_parts[1].strip() if len(query_parts) > 1 and query_parts[1].strip() else transcribed_text
    except Exception:
        user_query = transcribed_text # Fallback
    
    logger.info(f"User query for LLM: \"{user_query}\"")

    messages = [{"role": "user", "content": user_query}]
    ensure_tools_loaded() # Make sure _cached_tool_schemas and _cached_function_dispatch_table are populated

    if not _cached_tool_schemas or not _cached_function_dispatch_table:
        logger.warning("No tools available. Proceeding with basic chat.")
        # Basic chat response if no tools
        try:
            response = litellm.completion(model=LITELLM_MODEL, messages=messages)
            content = response.choices[0].message.content
            if content: speak(content)
            else: speak("I'm not sure how to respond to that.")
        except Exception as e:
            logger.error(f"Error in basic LLM completion: {e}")
            speak("Sorry, I had trouble processing that.")
        return

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
                    
                    # Ensure 'add_file' gets the original user query for its internal LLM call
                    if function_name == "add_file":
                        args_dict["user_query_for_filename"] = user_query
                    
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
            
            # Get final response from LLM after tool execution(s)
            logger.info("Sending tool responses to LLM for final summarization.")
            final_response_obj = litellm.completion(model=LITELLM_MODEL, messages=messages)
            final_content = final_response_obj.choices[0].message.content
            if final_content: speak(final_content)
            else: speak("Done.") # Fallback if LLM gives no summary

        elif response_message.content:
            logger.info(f"LLM returned direct response: {response_message.content}")
            speak(response_message.content)
        else:
            logger.warning("LLM returned no tool calls and no content.")
            speak("I'm not quite sure how to help with that.")

    except Exception as e:
        logger.error(f"Error in LLM completion or tool processing: {e}", exc_info=True)
        speak("Sorry, I encountered an error while processing your request.")

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
    else:
        # Ensure CWD file and its directory exist before starting assistant
        if not CWD_FILE.parent.exists():
            CWD_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not CWD_FILE.exists():
            get_current_directory() # This will create it with default if not present
            
        main_assistant_loop()

if __name__ == "__main__":
    cli_main()
