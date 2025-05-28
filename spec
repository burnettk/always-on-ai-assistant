i need a new voice assistant in max.py using tts like the other one, but this one should just use functions defined inline. those functions should have annotations that allow them to be converted into tool call
s. we should use litellm with tool calling for deciding which function to use, assuming that the call word (max) has been uttered in the last tts transcription. the two functions that we should support are chan
ge directory, what is your name (max), and add file. add file needs to call the llm once to determine which tool to use (add file), and then needs to call the LLM again to determine what file to add based on th
e listing of files that are available in the current directory (where the current directory is stored in here: > cat ~/.config/max/cwd
/Users/kevin/projects/github/discoveryedu/infr-foundation-api
