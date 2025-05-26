#!/usr/bin/env bash

function error_handler() {
  >&2 echo "Exited with BAD EXIT CODE '${2}' in ${0} script at line: ${1}."
  exit "$2"
}
trap 'error_handler ${LINENO} $?' ERR
set -o errtrace -o errexit -o nounset -o pipefail

echo >scratchpad.md
uv run python main_typer_assistant.py awaken --typer-file commands/template.py --scratchpad scratchpad.md --mode execute-no-scratch --llm deepseek
