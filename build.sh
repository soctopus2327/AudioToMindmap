#!/bin/bash
sudo apt-get update
sudo apt-get install -y graphviz
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
