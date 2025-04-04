#!/bin/bash
pip install --upgrade "pip<23.0"
pip install rich==13.7.0
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"
