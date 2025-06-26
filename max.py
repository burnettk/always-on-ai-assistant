"""
Max: A voice-activated or text-based AI assistant for developers.

USAGE: uv run max.py -q "change directory to always"

Max can perform actions on your local machine through a tool-based architecture.
It is designed to be invoked with a wake word ("Max") or via a direct text query.

Features:
- Voice activation using RealtimeSTT.
- Text-to-speech output using ElevenLabs or macOS 'say' command.
- Extensible tool system for adding new capabilities.
- LLM-powered tool selection and natural language understanding.
- Manages its own state (e.g., current working directory) for context-aware actions.

Usage:
- Voice mode: `uv run max.py`
- Text query: `uv run max.py -q "your command"`
- See tools:  `uv run max.py --tools`
"""
import logging
import json
import os
import inspect
import sys
import argparse
import subprocess
import shutil
import re
from pydantic import BaseModel, create_model
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path

# Third-party imports
import litellm
from RealtimeSTT import AudioToTextRecorder
from elevenlabs import play, Voice, VoiceSettings
from elevenlabs.client import ElevenLabs

# --- Configuration ---
MAGIC_QUERY_PARAM_NAME = "full_user_query"
CONFIG_DIR = Path.home() / ".config" / "max"
CWD_FILE = CONFIG_DIR / "cwd"
ALL_DIRS_FILE = CONFIG_DIR / "all_dirs"
FILES_IN_CONTEXT_FILE = CONFIG_DIR / "files_in_context"
CODING_MODEL_ALIAS_FILE = CONFIG_DIR / "coding_model_alias"
ASSISTANT_NAME = "Max"
WAKE_WORD = "max"

LITELLM_MODEL = os.getenv("LITELLM_MODEL", "gemini/gemini-2.5-flash-preview-05-20")
ELEVENLABS_API_KEY = os.getenv("ELEVEN_API_KEY")

# --- Global State ---
logger = logging.getLogger(__name__)
_tool_functions: Dict[str, Callable] = {}
_recorder_instance: Optional[AudioToTextRecorder] = None
_cached_tool_schemas: Optional[List[Dict[str, Any]]] = None
_cached_function_dispatch_table: Optional[Dict[str, Callable]] = None
elevenlabs_client: Optional[ElevenLabs] = None
USE_ELEVENLABS_TTS = False
SAY_COMMAND_AVAILABLE: Optional[bool] = None

# --- Setup and Initialization ---

def _setup_logging():
    """Sets up the initial logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    litellm.suppress_debug_info = True

def _create_config_dir_if_not_exists():
    """Ensures the configuration directory and the CWD file exist."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if not CWD_FILE.exists():
            # Call get_current_directory to create it with the default value
            get_current_directory()
    except Exception as e:
        logger.error(f"Fatal: Could not create config directory {CONFIG_DIR}: {e}", exc_info=True)
        sys.exit(1)

def _initialize_tts(use_elevenlabs: bool):
    """Initializes the TTS system based on user flags for voice mode."""
    global USE_ELEVENLABS_TTS, elevenlabs_client
    if use_elevenlabs:
        logger.info("ElevenLabs TTS explicitly enabled.")
        if ELEVENLABS_API_KEY:
            try:
                elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
                logger.info("ElevenLabs client initialized successfully.")
                USE_ELEVENLABS_TTS = True
            except Exception as e:
                logger.error(f"Failed to initialize ElevenLabs client: {e}. Disabling ElevenLabs TTS.")
                USE_ELEVENLABS_TTS = False
        else:
            logger.warning("ElevenLabs TTS flag is set, but ELEVEN_API_KEY is not. Disabling ElevenLabs TTS.")
            USE_ELEVENLABS_TTS = False
    else:
        USE_ELEVENLABS_TTS = False
    
    # Always check for the fallback 'say' command.
    check_say_command()

# --- Tool Definition and Schema Generation ---

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
            continue

        annotation = param.annotation
        if annotation == inspect.Parameter.empty:
            raise ValueError(f"Missing type annotation for parameter '{name}' in function '{func_name}'")
        
        default_value = param.default if param.default != inspect.Parameter.empty else ...
        param_fields[name] = (annotation, default_value)

    parameters_model = create_model(f"{func_name}Params", **param_fields, __base__=BaseModel)
    parameters_json_schema = parameters_model.model_json_schema()
    
    return {
        "type": "function",
        "function": {
            "name": func_name,
            "description": description,
            "parameters": parameters_json_schema
        }
    }

def get_all_tool_schemas() -> List[Dict[str, Any]]:
    """Gets schemas for all registered tool functions."""
    schemas = []
    for name, func in _tool_functions.items():
        try:
            schemas.append(get_function_schema(func))
        except ValueError as e:
            logger.error(f"Skipping tool {name} due to schema generation error: {e}")
    return schemas

# --- Helper Functions ---

def get_current_directory() -> str:
    """Reads the current working directory from the CWD_FILE, creating it if necessary."""
    try:
        if not CWD_FILE.exists():
            default_cwd = str(Path.home())
            CWD_FILE.parent.mkdir(parents=True, exist_ok=True)
            CWD_FILE.write_text(default_cwd)
            logger.info(f"CWD file not found. Created and set to default: {default_cwd}")
            return default_cwd
        return CWD_FILE.read_text().strip()
    except Exception as e:
        logger.error(f"Error accessing CWD file {CWD_FILE}: {e}. Falling back to home directory.")
        return str(Path.home())

def check_say_command():
    """Checks if the 'say' command is available and caches the result."""
    global SAY_COMMAND_AVAILABLE
    if SAY_COMMAND_AVAILABLE is None:
        SAY_COMMAND_AVAILABLE = shutil.which("say") is not None
        if SAY_COMMAND_AVAILABLE:
            logger.info("TTS: 'say' command is available.")
        else:
            logger.warning("TTS: 'say' command not found. Console print will be used if ElevenLabs is not active or fails.")

def speak(text: str):
    """Speaks text using ElevenLabs or a fallback 'say' command, otherwise prints to console."""
    if USE_ELEVENLABS_TTS and elevenlabs_client:
        try:
            logger.info(f'Attempting to speak with ElevenLabs: "{text}"')
            voice_id = os.getenv("ELEVENLABS_VOICE_ID", 'pNInz6obpgDQGcFmaJgB')
            audio = elevenlabs_client.generate(
                text=text,
                voice=Voice(
                    voice_id=voice_id,
                    settings=VoiceSettings(stability=0.7, similarity_boost=0.6, style=0.0, use_speaker_boost=True)
                ),
                model=os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2")
            )
            play(audio)
            logger.info(f'Spoke with ElevenLabs: "{text}"')
            return
        except Exception as e:
            logger.error(f"ElevenLabs TTS failed: {e}. Falling back.")
    
    if SAY_COMMAND_AVAILABLE:
        try:
            subprocess.run(["say", text], check=True, capture_output=True, text=True)
            logger.info(f'TTS (say): "{text}"')
            return
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"TTS with 'say' command failed: {e}. Falling back to print.")
    
    print(f"{ASSISTANT_NAME}: {text}")
    logger.info(f'TTS (console print): "{text}"')

def _select_item_with_llm(user_query: str, item_type_name: str, items: List[str], action_description: str) -> str:
    """Uses an LLM to select a single item from a list based on a user query."""
    items_list_str = "\n".join(f"- {item}" for item in items)
    prompt = (
        f"The user wants to {action_description}. Their original request was: '{user_query}'.\n"
        f"Based on their request and the following list of available {item_type_name}s, which single option do they most likely mean? "
        f"Respond with only the full item from the list. If completely unsure, respond with 'UNCLEAR'.\n"
        f"Available {item_type_name}s:\n{items_list_str}\n"
    )

    logger.info(f"Asking LLM to select {item_type_name}...")
    try:
        messages = [{"role": "user", "content": prompt}]
        response = litellm.completion(model=LITELLM_MODEL, messages=messages)
        selected_item = response.choices[0].message.content.strip()
        logger.info(f"LLM suggested {item_type_name}: '{selected_item}'")
        
        if not selected_item or selected_item.upper() == "UNCLEAR":
            return "UNCLEAR"
        return selected_item
    except Exception as e:
        logger.error(f"Error during LLM call for {item_type_name} selection: {e}", exc_info=True)
        return f"LLM_ERROR: {str(e)}"

# --- Tool Functions ---

@tool_function
def change_directory(full_user_query: str):
    """
    Identifies a target directory based on user query and a predefined list, then changes Max's current working directory.
    """
    if not ALL_DIRS_FILE.exists():
        return f"Sorry, the configuration file ({ALL_DIRS_FILE}) listing known directories is missing."

    try:
        possible_dirs = [d.strip() for d in ALL_DIRS_FILE.read_text().splitlines() if d.strip()]
        if not possible_dirs:
            return "Sorry, the list of known directories is empty. I don't have anywhere specific to go."
    except Exception as e:
        return f"Sorry, I had trouble reading the list of known directories. Error: {str(e)}"

    selected_dir_str = _select_item_with_llm(full_user_query, "directory", possible_dirs, "change the current directory")
    
    if selected_dir_str == "UNCLEAR":
        return f"I'm not sure which directory you mean from your request: '{full_user_query}'. Could you be more specific?"
    if selected_dir_str.startswith("LLM_ERROR"):
        return f"Sorry, I had trouble determining which directory to use. Error: {selected_dir_str}"

    normalized_selected_dir = str(Path(selected_dir_str).expanduser().resolve())
    normalized_possible_dirs_map = {str(Path(d).expanduser().resolve()): d for d in possible_dirs}

    if normalized_selected_dir not in normalized_possible_dirs_map:
        logger.warning(f"LLM selected '{selected_dir_str}', which is not in the predefined list.")
        return f"I'm sorry, but '{selected_dir_str}' is not one of the predefined directories I know."

    final_path_to_change = normalized_possible_dirs_map[normalized_selected_dir]
    expanded_path = Path(final_path_to_change).expanduser().resolve()

    if not expanded_path.is_dir():
        return f"Error: The selected directory '{expanded_path}' is not a valid directory."
    
    try:
        CWD_FILE.write_text(str(expanded_path))
        logger.info(f"Changed CWD to: {expanded_path}")
        return f"Okay, I've changed the current directory to {expanded_path.name}."
    except Exception as e:
        logger.error(f"Error writing to CWD file {CWD_FILE}: {e}", exc_info=True)
        return f"Sorry, I encountered an error while changing the directory. Error: {str(e)}"

@tool_function
def add_file(full_user_query: str):
    """
    Identifies a Python file to add to a persistent context for the current directory.
    """
    current_dir_str = get_current_directory()
    current_dir = Path(current_dir_str)
    
    if not current_dir.is_dir():
        return f"Error: The current directory '{current_dir_str}' is not valid."

    try:
        process = subprocess.run(
            ["git", "ls-files"], cwd=current_dir_str, capture_output=True, text=True, check=True
        )
        python_files = [
            path for path in process.stdout.splitlines() 
            if path.endswith('.py') and (current_dir / path).is_file()
        ]
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Error getting git-tracked files in {current_dir_str}: {e}")
        return "Sorry, I can only add files from Git-tracked directories, and I had trouble listing them."
    
    if not python_files:
        return f"I couldn't find any Python (.py) files tracked by Git in '{current_dir_str}'."

    selected_file = _select_item_with_llm(full_user_query, "Python file", python_files, "add a file to the context")

    if selected_file == "UNCLEAR":
        return f"I'm not sure which file you mean from your request: '{full_user_query}'. Could you be more specific?"
    if selected_file.startswith("LLM_ERROR"):
        return f"Sorry, I had trouble determining which file to add. Error: {selected_file}"

    if selected_file not in python_files:
        logger.warning(f"LLM selected '{selected_file}', which is not in the list of available files.")
        return f"I'm sorry, but '{selected_file}' is not among the Python files I found."

    try:
        context_data = {}
        if FILES_IN_CONTEXT_FILE.exists() and FILES_IN_CONTEXT_FILE.stat().st_size > 0:
            with open(FILES_IN_CONTEXT_FILE, "r") as f:
                context_data = json.load(f)
        
        files_for_dir = context_data.setdefault(current_dir_str, [])
        if selected_file not in files_for_dir:
            files_for_dir.append(selected_file)
            with open(FILES_IN_CONTEXT_FILE, "w") as f:
                json.dump(context_data, f, indent=2)
            logger.info(f"Added '{selected_file}' to context for '{current_dir_str}'")
            return f"Okay, I've added the file '{selected_file}' to the context."
        else:
            return f"The file '{selected_file}' is already in the context for this directory."
    except Exception as e:
        logger.error(f"Error updating context file {FILES_IN_CONTEXT_FILE}: {e}", exc_info=True)
        return f"Sorry, I encountered an error updating the file context. Error: {str(e)}"

@tool_function
def set_coding_model(full_user_query: str):
    """
    Sets the coding model alias by selecting from a list of available models from 'ca -l'.
    """
    try:
        process = subprocess.run(["ca", "-l"], capture_output=True, text=True, check=True)
        ca_output = process.stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        logger.error(f"Error running 'ca -l': {e}")
        return "Sorry, I need the 'ca' command to set the coding model, and it's not working."

    name_to_alias_map = {}
    alias_regex = re.compile(r'\(aliases: ([^\)]+)\)')
    for line in ca_output.strip().splitlines():
        if ':' not in line: continue
        model_name = line.split(':')[0].strip()
        match = alias_regex.search(line)
        if match:
            aliases = [a.strip() for a in match.group(1).split(',')]
            if aliases:
                primary_alias = aliases[0]
                name_to_alias_map[model_name] = primary_alias
                for alias in aliases:
                    name_to_alias_map[alias] = alias

    if not name_to_alias_map:
        return "I ran 'ca -l' but couldn't find any models with defined aliases."

    available_choices = sorted(list(name_to_alias_map.keys()))
    selected_choice = _select_item_with_llm(full_user_query, "model or alias", available_choices, "set the coding model")

    if selected_choice == "UNCLEAR":
        return f"I'm not sure which coding model you mean from your request: '{full_user_query}'."
    if selected_choice.startswith("LLM_ERROR"):
        return f"Sorry, I had trouble determining which model to set. Error: {selected_choice}"

    if selected_choice not in name_to_alias_map:
        logger.warning(f"LLM selected '{selected_choice}', which is not a valid option.")
        return f"I'm sorry, but '{selected_choice}' is not a valid model or alias I can set."

    alias_to_save = name_to_alias_map[selected_choice]
    try:
        CODING_MODEL_ALIAS_FILE.write_text(alias_to_save)
        logger.info(f"Set coding model alias to '{alias_to_save}'")
        return f"Okay, I've set the coding model alias to {alias_to_save}."
    except Exception as e:
        logger.error(f"Error writing to coding model alias file {CODING_MODEL_ALIAS_FILE}: {e}", exc_info=True)
        return f"Sorry, I couldn't save the coding model alias. Error: {str(e)}"

@tool_function
def update_code(full_user_query: str):
    """
    Updates code in the files currently in context using the configured coding model.
    """
    try:
        if not CODING_MODEL_ALIAS_FILE.exists() or CODING_MODEL_ALIAS_FILE.stat().st_size == 0:
            return "The coding model alias is not set. Please set it first."
        coding_model_alias = CODING_MODEL_ALIAS_FILE.read_text().strip()
    except Exception as e:
        return f"Sorry, I had trouble reading the coding model alias. Error: {str(e)}"

    current_dir_str = get_current_directory()
    files_in_context = []
    try:
        if FILES_IN_CONTEXT_FILE.exists() and FILES_IN_CONTEXT_FILE.stat().st_size > 0:
            context_data = json.loads(FILES_IN_CONTEXT_FILE.read_text())
            files_in_context = context_data.get(current_dir_str, [])
        if not files_in_context:
            return "There are no files in the context for the current directory. Please add files first."
    except Exception as e:
        return f"Sorry, I had trouble reading the file context. Error: {str(e)}"

    command = ["ca", coding_model_alias] + files_in_context + ["-m", full_user_query]
    logger.info(f"Executing code update command: {' '.join(command)}")

    try:
        process = subprocess.run(command, cwd=current_dir_str, capture_output=True, text=True)
        stdout, stderr = process.stdout.strip(), process.stderr.strip()
        
        logger.info(f"'ca' command stdout: {stdout}")
        if stderr: logger.warning(f"'ca' command stderr: {stderr}")

        if stderr and not stdout:
            return f"The code update command failed with an error: {stderr}"
        
        response = stdout
        if stderr:
            response += f"\n\nNotes from the process:\n{stderr}"
        
        return response or "The code update command ran but produced no output."
    except FileNotFoundError:
        return "Sorry, the 'ca' command is not installed or not in your PATH. I need it to update code."
    except Exception as e:
        logger.error(f"An unexpected error occurred while running 'ca': {e}", exc_info=True)
        return f"An unexpected error occurred while trying to update code: {str(e)}"

# --- STT and Main Assistant Logic ---

def handle_llm_interaction(user_query: str) -> Optional[str]:
    """Handles the core interaction with the LLM, including tool calls."""
    logger.info(f'Core LLM interaction for query: "{user_query}"')
    ensure_tools_loaded()

    system_prompt = (
        "You are a voice assistant named Max. Your primary purpose is to take action by calling tools. "
        "Analyze the user's request and select the most appropriate tool. "
        "Your responses will be spoken out loud, so be concise and use natural, conversational language. "
        "Avoid complex punctuation. When a tool is used, provide a brief confirmation. "
        "Do not engage in chit-chat; your value is in successfully executing tools."
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}]

    if not _cached_tool_schemas:
        logger.warning("No tools available. Attempting basic chat.")
        try:
            response = litellm.completion(model=LITELLM_MODEL, messages=messages)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in basic LLM completion: {e}")
            return "Sorry, I had trouble processing that."

    try:
        response_obj = litellm.completion(model=LITELLM_MODEL, messages=messages, tools=_cached_tool_schemas, tool_choice="auto")
        response_message = response_obj.choices[0].message
        tool_calls = response_message.tool_calls

        if not tool_calls:
            logger.info("LLM did not return any tool calls. Using direct response.")
            return response_message.content

        logger.info(f"LLM returned tool calls: {tool_calls}")
        messages.append(response_message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = _cached_function_dispatch_table.get(function_name)

            if not function_to_call:
                logger.error(f"LLM requested unknown function: {function_name}")
                content = f"Error: Function {function_name} is not recognized."
            else:
                try:
                    args_dict = json.loads(tool_call.function.arguments)
                    if MAGIC_QUERY_PARAM_NAME in inspect.signature(function_to_call).parameters:
                        args_dict[MAGIC_QUERY_PARAM_NAME] = user_query
                    
                    logger.info(f"Calling tool: {function_name} with args: {args_dict}")
                    content = function_to_call(**args_dict)
                except Exception as e:
                    logger.error(f"Error executing tool {function_name}: {e}", exc_info=True)
                    content = f"Error executing {function_name}: {str(e)}"
            
            messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": str(content)})
        
        final_response_obj = litellm.completion(model=LITELLM_MODEL, messages=messages)
        return final_response_obj.choices[0].message.content

    except Exception as e:
        logger.error(f"Error in LLM completion or tool processing: {e}", exc_info=True)
        return "Sorry, I encountered an error while processing your request."

def ensure_tools_loaded():
    """Loads tool schemas and dispatch table if not already loaded."""
    global _cached_tool_schemas, _cached_function_dispatch_table
    if _cached_tool_schemas is None:
        _cached_tool_schemas = get_all_tool_schemas()
        _cached_function_dispatch_table = {
            tool["function"]["name"]: _tool_functions[tool["function"]["name"]]
            for tool in _cached_tool_schemas
        }
        logger.info(f"Tools loaded: {list(_cached_function_dispatch_table.keys())}")

def process_transcription(transcribed_text: str):
    """Processes transcribed text, checks for wake word, and interacts with LLM."""
    global _recorder_instance
    logger.info(f'Processing transcription: "{transcribed_text}"')
    
    if WAKE_WORD.lower() not in transcribed_text.lower():
        logger.info(f"Wake word '{WAKE_WORD}' not detected. Ignoring.")
        return
    
    query_parts = transcribed_text.lower().split(WAKE_WORD.lower(), 1)
    user_query = query_parts[1].strip() if len(query_parts) > 1 and query_parts[1].strip() else transcribed_text
    
    logger.info(f'User query for LLM: "{user_query}"')
    final_content = handle_llm_interaction(user_query)

    if final_content is not None and _recorder_instance:
        _recorder_instance.pause_recording = True
        try:
            speak(final_content or "Done.")
        finally:
            _recorder_instance.pause_recording = False

def process_text_query(user_query: str):
    """Processes a text query, interacts with LLM/tools, and prints the response."""
    logger.info(f'Processing text query: "{user_query}"')
    final_content = handle_llm_interaction(user_query)
    
    if final_content:
        print(f"{ASSISTANT_NAME}: {final_content}")
    elif final_content == "":
        print(f"{ASSISTANT_NAME}: Done.")
    else:
        print(f"{ASSISTANT_NAME}: I processed your request, but there was no specific information to return.")

def main_assistant_loop():
    """Initializes STT and runs the main loop for voice interaction."""
    global _recorder_instance
    ensure_tools_loaded()

    recorder_config = {
        "spinner": False, "model": os.getenv("STT_MODEL", "tiny.en"), "language": "en",
        "post_speech_silence_duration": float(os.getenv("STT_SILENCE_DURATION", 1.2)),
        "beam_size": int(os.getenv("STT_BEAM_SIZE", 5)),
        "print_transcription_time": False, "realtime_processing_pause": 0.2,
    }
    logger.info(f"Initializing RealtimeSTT with config: {recorder_config}")
    
    try:
        _recorder_instance = AudioToTextRecorder(**recorder_config)
    except Exception as e:
        logger.error(f"Failed to initialize AudioToTextRecorder: {e}. Voice assistant cannot start.", exc_info=True)
        print(f"Error: Could not start audio recorder. Details: {e}")
        return

    logger.info(f"'{ASSISTANT_NAME}' is listening... Say '{WAKE_WORD}' followed by your command.")
    speak(f"Hello! I'm {ASSISTANT_NAME}. How can I assist you today?")

    try:
        while True:
            transcribed_text = _recorder_instance.text()
            if transcribed_text:
                process_transcription(transcribed_text)
    except KeyboardInterrupt:
        speak("Goodbye!")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
        speak("An unexpected error occurred. Shutting down.")
    finally:
        logger.info(f"'{ASSISTANT_NAME}' assistant stopped.")

# --- CLI ---
def cli_main():
    """Command Line Interface for the assistant."""
    parser = argparse.ArgumentParser(description=f"{ASSISTANT_NAME} Voice Assistant")
    parser.add_argument("--tools", action="store_true", help="Dump tool schemas and exit.")
    parser.add_argument("--elevenlabs", action="store_true", help="Use ElevenLabs for TTS.")
    parser.add_argument("-q", "--query", type=str, help="Process a text query and exit.")
    args = parser.parse_args()

    _setup_logging()
    _create_config_dir_if_not_exists()

    if args.tools:
        ensure_tools_loaded()
        print(json.dumps(_cached_tool_schemas, indent=2))
        sys.exit(0)
    
    if args.query:
        process_text_query(args.query)
    else:
        _initialize_tts(args.elevenlabs)
        main_assistant_loop()

if __name__ == "__main__":
    cli_main()
