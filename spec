i need a new voice assistant in max.py using tts like the other one, but this one should just use functions defined inline like in types.py in order to auto-generate available tool/function calls.
we should use litellm with tool calling for deciding which function to use, assuming that the call word (max) has been uttered in the last tts transcription.
the functions that we should support are change directory, what is your name (max), and add file.
add file needs to call the llm once to determine which tool to use (add file), and then needs to call the LLM again to determine what file to add based on the listing of files that are available in the current directory
note that the the current directory is stored in here: > cat ~/.config/max/cwd
/Users/kevin/projects/github/discoveryedu/infr-foundation-api

I also need a cli switch (--tools) that just dumps the tool call metadata out as pure json rather than starting the assistant.
