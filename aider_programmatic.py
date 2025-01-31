dependencies = [
    "aider-chat>=0.65.0",
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput

model = Model(self.config.coder_model)
coder = Coder.create(
    main_model=model,
    io=InputOutput(yes=True),
    fnames=self.config.context_editable,
    read_only_fnames=self.config.context_read_only,
    auto_commits=False,
    suggest_shell_commands=False,
    detect_urls=False,
)
coder.run(prompt)
