from typing import List
import os
import logging
import json
from datetime import datetime
from modules.assistant_config import get_config
from modules.utils import (
    build_file_name_session,
    create_session_logger_id,
    setup_logging,
)
from modules.deepseek import prefix_prompt
from modules.execute_python import execute_uv_python, execute
from elevenlabs import play
from elevenlabs.client import ElevenLabs
import time


from modules.gemini import conversational_prompt as gemini_conversational_prompt

class TyperAgent:
    def __init__(self, logger: logging.Logger, session_id: str, llm: str):
        self.logger = logger
        self.session_id = session_id
        self.log_file = build_file_name_session("session.log", session_id)
        self.elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))
        self.previous_successful_requests = []
        self.previous_responses = []
        self.llm = llm

    def _validate_markdown(self, file_path: str) -> bool:
        """Validate that file is markdown and has expected structure"""
        if not file_path.endswith((".md", ".markdown")):
            self.logger.error(f"📄 Scratchpad file {file_path} must be a markdown file")
            return False

        try:
            with open(file_path, "r") as f:
                content = f.read()
                # Basic validation - could be expanded based on needs
                if not content.strip():
                    self.logger.warning("📄 Markdown file is empty")
                return True
        except Exception as e:
            self.logger.error(f"📄 Error reading markdown file: {str(e)}")
            return False

    @classmethod
    def build_agent(cls, typer_file: str, scratchpad: List[str], llm: str ):
        """Create and configure a new TyperAssistant instance"""
        session_id = create_session_logger_id()
        logger = setup_logging(session_id)
        logger.info(f"🚀 Starting STT session {session_id}")

        if not os.path.exists(typer_file):
            logger.error(f"📂 Typer file {typer_file} does not exist")
            raise FileNotFoundError(f"Typer file {typer_file} does not exist")

        # Validate markdown scratchpad
        agent = cls(logger, session_id, llm)
        if scratchpad and not agent._validate_markdown(scratchpad[0]):
            raise ValueError(f"Invalid markdown scratchpad file: {scratchpad[0]}")

        return agent, typer_file, scratchpad[0]
    
    def _get_llm_response(self, formatted_prompt: str, prefix: str = "") -> str:
        """Get response from the selected LLM"""
        if self.llm == "deepseek":
            if not os.getenv("DEEPSEEK_API_KEY"):
                raise ValueError("DEEPSEEK_API_KEY environment variable not set.")
            return prefix_prompt(prompt=formatted_prompt, prefix=prefix)
        elif self.llm == "gemini":
            if not os.getenv("GEMINI_API_KEY"):
                raise ValueError("GEMINI_API_KEY environment variable not set.")
            return gemini_conversational_prompt(messages=[{"role": "user", "content": formatted_prompt}])
        else:
            raise ValueError(f"Invalid LLM: {self.llm}")

    def build_prompt(
        self,
        typer_file: str,
        scratchpad: str,
        context_files: List[str],
        prompt_text: str,
    ) -> str:
        """Build and format the prompt template with current state"""
        try:
            # Load typer file
            self.logger.info("📂 Loading typer file...")
            with open(typer_file, "r") as f:
                typer_content = f.read()

            # Load scratchpad file
            self.logger.info("📝 Loading scratchpad file...")
            if not os.path.exists(scratchpad):
                self.logger.error(f"📄 Scratchpad file {scratchpad} does not exist")
                raise FileNotFoundError(f"Scratchpad file {scratchpad} does not exist")

            with open(scratchpad, "r") as f:
                scratchpad_content = f.read()

            # Load context files
            context_content = ""
            for file_path in context_files:
                if not os.path.exists(file_path):
                    self.logger.error(f"📄 Context file {file_path} does not exist")
                    raise FileNotFoundError(f"Context file {file_path} does not exist")

                with open(file_path, "r") as f:
                    file_content = f.read()
                    file_name = os.path.basename(file_path)
                    context_content += f'\t<context name="{file_name}">\n{file_content}\n</context>\n\n'

            # Load and format prompt template
            self.logger.info("📝 Loading prompt template...")
            with open("prompts/typer-commands.xml", "r") as f:
                prompt_template = f.read()

            # Replace template placeholders
            formatted_prompt = (
                prompt_template.replace("{{typer-commands}}", typer_content)
                .replace("{{scratch_pad}}", scratchpad_content)
                .replace("{{context_files}}", context_content)
                .replace("{{natural_language_request}}", prompt_text)
            )

            # Log the filled prompt template to file only (not stdout)
            with open(self.log_file, "a") as log:
                log.write("\n📝 Filled prompt template:\n")
                log.write(formatted_prompt)
                log.write("\n\n")

            return formatted_prompt

        except Exception as e:
            self.logger.error(f"❌ Error building prompt: {str(e)}")
            raise

    def _get_dir_list(self) -> List[str]:
        """Get list of directories from config file"""
        dir_file = os.path.expanduser("~/.config/max/all_dirs")
        if not os.path.exists(dir_file):
            self.logger.error(f"Directory list file {dir_file} not found")
            return []
        
        with open(dir_file, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def process_text(
        self,
        text: str,
        typer_file: str,
        scratchpad: str,
        context_files: List[str],
        mode: str,
    ) -> str:
        """Process text input and handle based on execution mode"""
        try:
            # Handle exact match case for directory change
            if "change dir" in text.lower() or "change directory" in text.lower():
                dirs = self._get_dir_list()
                if not dirs:
                    self.think_speak("No directories found in config")
                    return "No directories found in config"
                
                # Add dirs to context for LLM to choose from
                formatted_prompt = self.build_prompt(
                    typer_file, 
                    scratchpad, 
                    context_files, 
                    f"The argument to change_dir, the dir_full_path, should be a single directory intelligently picked from the directory list based on the user query. directory list: {', '.join(dirs)}. user query: {text}"
                )
            else:
                # Normal processing
                formatted_prompt = self.build_prompt(
                    typer_file, scratchpad, context_files, text
                )

            # Generate command using the selected LLM
            self.logger.info(f"🤖 Processing text with {self.llm}...")
            prefix = f"uv run python {typer_file}"
            command = self._get_llm_response(formatted_prompt, prefix)

            if command == prefix.strip():
                self.logger.info(f"🤖 Command not found for '{text}'")
                self.speak("I couldn't find that command")
                return "Command not found"

            # Handle different modes with markdown formatting
            assistant_name = get_config("typer_assistant.assistant_name")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # command_with_prefix = f"uv run python {typer_file} {command}"
            command_with_prefix = command

            if mode == "default":
                result = (
                    f"\n## {assistant_name} Generated Command ({timestamp})\n\n"
                    f"> Request: {text}\n\n"
                    f"```bash\n{command_with_prefix}\n```"
                )
                with open(scratchpad, "a") as f:
                    f.write(result)
                self.think_speak(f"Command generated")
                return result

            elif mode == "execute":
                self.logger.info(f"⚡ Executing command: `{command_with_prefix}`")
                output = execute(command)

                # Check if output is JSON
                try:
                    output_json = json.loads(output)
                    if "verbatim_vocal_response" in output_json:
                        self.speak(output_json["verbatim_vocal_response"])
                        return output_json["verbatim_vocal_response"]
                    elif "vocal_response_input" in output_json:
                        self.think_speak(output_json["vocal_response_input"])
                        return output_json["vocal_response_input"]
                    else:
                        self.speak("Error: Required keys not found in JSON response")
                        return "Error: Required keys not found in JSON response"
                except json.JSONDecodeError:
                    # Not JSON, proceed as usual
                    result = (
                        f"\n\n## {assistant_name} Executed Command ({timestamp})\n\n"
                        f"> Request: {text}\n\n"
                        f"**{assistant_name}'s Command:** \n```bash\n{command_with_prefix}\n```\n\n"
                        f"**Output:** \n```\n{output}```"
                    )
                    with open(scratchpad, "a") as f:
                        f.write(result)
                    self.think_speak(f"Command generated and executed")
                    return output

            elif mode == "execute-no-scratch":
                self.logger.info(f"⚡ Executing command: `{command_with_prefix}`")
                output = execute(command)

                # Check if output is JSON
                try:
                    output_json = json.loads(output)
                    if "verbatim_vocal_response" in output_json:
                        self.speak(output_json["verbatim_vocal_response"])
                        return output_json["verbatim_vocal_response"]
                    elif "vocal_response_input" in output_json:
                        self.think_speak(output_json["vocal_response_input"])
                        return output_json["vocal_response_input"]
                    else:
                        self.speak("Error: Required keys not found in JSON response")
                        return "Error: Required keys not found in JSON response"
                except json.JSONDecodeError:
                    # Not JSON, proceed as usual
                    self.think_speak(f"Command generated and executed")
                    return output

            else:
                self.think_speak(f"I had trouble running that command")
                raise ValueError(f"Invalid mode: {mode}")

        except Exception as e:
            self.logger.error(f"❌ Error occurred: {str(e)}")
            raise

    def think_speak(self, text: str):
        response_prompt_base = ""
        with open("prompts/concise-assistant-response.xml", "r") as f:
            response_prompt_base = f.read()

        assistant_name = get_config("typer_assistant.assistant_name")
        human_companion_name = get_config("typer_assistant.human_companion_name")

        response_prompt = response_prompt_base.replace("{{latest_action}}", text)
        response_prompt = response_prompt.replace(
            "{{human_companion_name}}", human_companion_name
        )
        response_prompt = response_prompt.replace(
            "{{personal_ai_assistant_name}}", assistant_name
        )
        prompt_prefix = f"Your Conversational Response: "
        if self.llm == "deepseek":
            response = prefix_prompt(
                prompt=response_prompt, prefix=prompt_prefix, no_prefix=True
            )
        elif self.llm == "gemini":
            response = gemini_conversational_prompt(
                messages=[{"role": "user", "content": response_prompt}]
            )
        else:
            raise ValueError(f"Invalid LLM: {self.llm}")

        self.logger.info(f"🤖 Spoken response: '{response}'")
        self.speak(response)

    def speak(self, text: str):

        start_time = time.time()
        model = "eleven_flash_v2_5"
        # model="eleven_flash_v2"
        # model = "eleven_turbo_v2"
        # model = "eleven_turbo_v2_5"
        # model="eleven_multilingual_v2"
        voice = get_config("typer_assistant.elevenlabs_voice")

        audio_generator = self.elevenlabs_client.generate(
            text=text,
            voice=voice,
            model=model,
            stream=False,
        )
        audio_bytes = b"".join(list(audio_generator))
        duration = time.time() - start_time
        self.logger.info(f"Model {model} completed tts in {duration:.2f} seconds")
        play(audio_bytes)
