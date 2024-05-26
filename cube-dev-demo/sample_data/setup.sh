# User-defined variables
VENV_NAME=.venv_cube_dev;
WORKING_DIRECTORY=cube-dev-demo/sample_data;
REQUIREMENTS_TXT_DIR=cube-dev-demo/sample_data;
REQUIREMENTS_TXT_FILE=requirements.txt;

# Set working directory
cd $(git rev-parse --show-toplevel)/$WORKING_DIRECTORY;

# Create .gitignore if doesn't exist
if [[ ! -e .gitignore ]]; then
    touch .gitignore;
fi

# Deactivate virtual environment if currently active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo 'deactivating Virtual Env';
    deactivate;
fi

# Set platform-specific subdir (handles if running this on Windows)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    VENV_SUBDIR="Scripts";
    PYTHON_EXEC=python;
else
    VENV_SUBDIR="bin";
    PYTHON_EXEC=python3;
fi

# Make sure VENV_NAME is in .gitignore
if grep -q "^$VENV_NAME" .gitignore; then
    echo "Virtual environment already in .gitignore";
else
    echo $VENV_NAME >> .gitignore;
fi

# Create virtual environment & activate it
$PYTHON_EXEC -m venv $VENV_NAME;
source "$VENV_NAME/$VENV_SUBDIR/activate";
$PYTHON_EXEC -m pip install --upgrade pip;
pip install -r $(git rev-parse --show-toplevel)/$REQUIREMENTS_TXT_DIR/$REQUIREMENTS_TXT_FILE;
