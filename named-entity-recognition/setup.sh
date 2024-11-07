python -m venv venv_ner &&
source venv_ner/bin/activate &&
pip install --upgrade pip &&
pip install -r requirements.txt &&
python -m ipykernel install --user --name=venv_ner
