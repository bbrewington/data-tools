# Set working directory to /dbt
cd $(git rev-parse --show-toplevel)/dbt;

# Create .gitignore if doesn't exist
if [[ ! -e .gitignore ]]; then
    touch .gitignore;
fi

# Deactivate virtual environment if currently active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo 'deactivating Virtual Env';
    deactivate;
fi

# Set variables

#   General variables
VENV_NAME=.venv_dbt;

#   Set platform-specific subdir (handles if running this on Windows)
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
pip install -r requirements_dbt.txt;

# 6. If you need to run this virtual environment in a Jupyter Notebook, uncomment these lines
# pip install ipykernel
# $PYTHON_EXEC -m ipykernel install --user --name ADD_NAME_HERE
