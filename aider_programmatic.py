# spec sample:
# # # Task Specification                                                           
#  > Description of what needs to be done
# ## High-Level Objective
# - Main goal
#
# ## Mid-Level Objective
#  - Specific steps
#
#  ## Implementation Notes
#  - Requirements
#  - Constraints
#
#  ## Context
#  ### Beginning Context
#  - List of files to start with                                                                                                          
# ### Ending Context
#  - Expected modified files
#
# ## Low-Level Tasks                                                            1. First task
# 2. Second task
# 
#
#
#
#
#
# from website:
#
# dependencies = [
#     "aider-chat>=0.65.0",
# from aider.coders import Coder
# from aider.models import Model
# from aider.io import InputOutput
#
# model = Model(self.config.coder_model)
# coder = Coder.create(
#     main_model=model,
#     io=InputOutput(yes=True),
#     fnames=self.config.context_editable,
#     read_only_fnames=self.config.context_read_only,
#     auto_commits=False,
#     suggest_shell_commands=False,
#     detect_urls=False,
# )
# coder.run(prompt)
#
#
#
#
#
#
# Agentic developer workflow script:
from pathlib import Path
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput

def your_task(parameters):
    # Verify project structure
    if not Path("pyproject.toml").exists():
        raise FileNotFoundError("Must run from project root")

    # Optional: Load specification
    spec_path = Path("specs/your-task-spec.md")
    with open(spec_path) as f:
        spec_content = f.read()

    # Define context
    context_editable = [
        "src/your_project/file1.py",
        "src/your_project/file2.py"
    ]
    context_read_only = [
        "pyproject.toml",
        "README.md"
    ]

    # Define prompt
    prompt = f"""
    Your specific instructions here
    {spec_content}  # If using a spec
    """

    # Initialize model
    model = Model("gpt-4o")  # or claude-3

    # Create coder instance
    coder = Coder.create(
        main_model=model,
        edit_format="architect",  # or other formats
        io=InputOutput(yes=True),  # for automated workflows
        fnames=context_editable,
        read_only_fnames=context_read_only,
        auto_commits=False,
        suggest_shell_commands=False
    )

    # Run the task
    coder.run(prompt)

if __name__ == "__main__":
    your_task()
