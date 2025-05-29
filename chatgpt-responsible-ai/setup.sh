# Change working directory into git_repo_root/chatgpt-responsible-ai
cd $(git rev-parse --show-toplevel)/chatgpt-responsible-ai

# Create & Activate Virtual Environment named "venv_cg_rai"
python -m venv venv_cg_rai
source venv_cg_rai/bin/activate

# Install stuff
pip install --upgrade pip
pip install ipykernel~=6.13.0
pip install -r requirements.txt

# On Brent's mac, getting a weird error, so leaving this commented out and just using .py files (instead of .ipynb)
# Error: (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64'))
  # Tried these and they didn't work:
  #   https://stackoverflow.com/questions/72619143/unable-to-import-psutil-on-m1-mac-with-miniforge-mach-o-file-but-is-an-incomp
  #   https://stackoverflow.com/questions/71882029/mach-o-file-but-is-an-incompatible-architecture-have-arm64-need-x86-64-i
# ipython kernel install --name "venv_cg_rai" --user
