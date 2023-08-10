# Make sure .venv is in .gitignore
if grep -q '^.venv$' .gitignore; then
  echo "Virtual environment already in .gitignore"
else
  echo '.venv' >> .gitignore
fi

# Set platform-specific subdir (handles if running this on Windows)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    VENV_SUBDIR="Scripts"
else
    VENV_SUBDIR="bin"
fi

# Create virtual environment & activate it
python3 -m venv .venv
source .venv/$VENV_SUBDIR/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# If you need to run this virtual environment in a Jupyter Notebook, uncomment these lines
# pip install ipykernel
# python3 -m ipykernel install --user --name=NAME_IT_HERE --display-name=DISPLAY_NAME_HERE
