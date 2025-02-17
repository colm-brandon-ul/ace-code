:: filepath: /Users/colmbrandon/Desktop/segarray-code/setup.bat
:: create virtual env
python -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt
pip install -r requirements-torch.txt