cd $(git rev-parse --show-toplevel)/datacamp-data-eng-with-python

python -m venv ~/.virtualenvs/venv_datacamp_de &&
source ~/.virtualenvs/venv_datacamp_de/bin/activate &&
pip install --upgrade pip &&
pip install -r requirements.txt
