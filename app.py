import os
import re
import streamlit as st
import whisper
import torch
import numpy as np
import nltk
import tempfile
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import networkx as nx
import graphviz
from transformers import BertTokenizer, BertModel
import spacy
from io import BytesIO
import subprocess
import importlib.util

nltk.download('punkt', quiet=True)

# Load or install spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --------------------------
# Caching BERT model
# --------------------------
@st.cache_resource
def load_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

tokenizer, bert_model = load_bert()

# --------------------------
# 1. Text Processing
# --------------------------
def process_text(text):
    """Extract sentences and BERT embeddings"""
    try:
        sentences = nltk.sent_tokenize(text)
    except:
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    sentences = [s.strip() for s in sentences if 5 <= len(s.split()) <= 30]
    
    if len(sentences) < 2:
        return None, None
    
    def get_bert_embeddings(sentence):
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    embeddings = [get_bert_embeddings(sent) for sent in sentences]
    return sentences, embeddings

# --------------------------
# 2. Relationship Extraction
# --------------------------
def extract_relationships(sentences, embeddings):
    relationships = []
    threshold = 0.65

    for _ in range(3):
        relationships = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                if sim > threshold:
                    relationships.append((sentences[i], sentences[j], sim))
        if len(relationships) >= max(3, len(sentences) // 2):
            break
        threshold -= 0.05

    return relationships if relationships else None

def extract_key_elements(text):
    doc = nlp(text)
    key_phrases = list(set([chunk.text for chunk in doc.noun_chunks if 2 <= len(chunk.text.split()) <= 4]))
    entities = {
        'people': list(set([ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG"]])),
        'numbers': list(set([ent.text for ent in doc.ents if ent.label_ in ["MONEY", "PERCENT", "DATE", "QUANTITY"]]))
    }
    return key_phrases, entities

# --------------------------
# 3. Mindmap Generation
# --------------------------
def create_mindmap(text):
    sentences, embeddings = process_text(text)
    if not sentences:
        return None, None

    relationships = extract_relationships(sentences, embeddings)
    if not relationships:
        return None, None

    key_phrases, entities = extract_key_elements(text)

    nx_graph = nx.Graph()
    connection_counts = defaultdict(int)
    for rel in relationships:
        connection_counts[rel[0]] += 1
        connection_counts[rel[1]] += 1
    central_node = max(connection_counts.items(), key=lambda x: x[1])[0]
    nx_graph.add_node(central_node, level=0, type='central', size=3000)

    for rel in relationships:
        if rel[0] == central_node or rel[1] == central_node:
            other_node = rel[1] if rel[0] == central_node else rel[0]
            nx_graph.add_node(other_node, level=1, type='sentence', size=2000)
            nx_graph.add_edge(central_node, other_node, weight=rel[2])

    for phrase in key_phrases[:10]:
        nx_graph.add_node(phrase, level=2, type='phrase', size=1500)
        for node in nx_graph.nodes():
            if nx_graph.nodes[node]['type'] == 'sentence' and phrase.lower() in node.lower():
                nx_graph.add_edge(node, phrase, weight=0.8)
                break

    gv_graph = graphviz.Digraph(engine='dot')
    gv_graph.attr(rankdir='TB', size='12,12', ratio='auto')

    for node in nx_graph.nodes():
        node_type = nx_graph.nodes[node]['type']
        wrapped_text = '\n'.join([node[i:i+30] for i in range(0, len(node), 30)])
        node_id = str(hash(node))
        if node_type == 'central':
            gv_graph.node(node_id, wrapped_text, shape='doublecircle', style='filled', fillcolor='#4E79A7', fontsize='14')
        elif node_type == 'sentence':
            gv_graph.node(node_id, wrapped_text, shape='ellipse', style='filled', fillcolor='#F28E2B', fontsize='12')
        else:
            gv_graph.node(node_id, wrapped_text, shape='box', style='filled', fillcolor='#59A14F', fontsize='10')

    for edge in nx_graph.edges():
        gv_graph.edge(str(hash(edge[0])), str(hash(edge[1])))

    return nx_graph, gv_graph

# --------------------------
# 4. Visualization
# --------------------------
def draw_networkx_graph(G):
    plt.figure(figsize=(16, 12))
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        node_type = G.nodes[node]['type']
        if node_type == 'central':
            node_colors.append('#4E79A7')
            node_sizes.append(3000)
        elif node_type == 'sentence':
            node_colors.append('#F28E2B')
            node_sizes.append(2000)
        else:
            node_colors.append('#59A14F')
            node_sizes.append(1500)

    pos = nx.spring_layout(G, k=0.6, seed=42)
    nx.draw(G, pos, 
            node_color=node_colors,
            node_size=node_sizes,
            edge_color='#888888',
            width=[G.edges[e]['weight'] for e in G.edges()],
            with_labels=True,
            font_size=9)

    plt.title("Semantic Mindmap", pad=20)
    plt.axis('off')
    return plt

def save_graph_as_image(plt_obj, format='jpeg'):
    buf = BytesIO()
    plt_obj.savefig(buf, format=format, dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf

def save_graphviz_as_image(gv_graph, format='jpeg'):
    try:
        img_data = gv_graph.pipe(format='jpeg')
        return BytesIO(img_data)
    except Exception as e:
        st.error(f"Failed to export Graphviz image: {str(e)}")
        return None

# --------------------------
# 5. Streamlit App
# --------------------------
st.title("🧠 Audio to Mindmap Generator")

uploaded_file = st.file_uploader("Upload audio file (WAV/MP3)", type=["wav", "mp3"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        audio_path = tmp_file.name

    # Transcription
    with st.spinner("Transcribing audio..."):
        try:
            model = whisper.load_model("tiny")
            result = model.transcribe(audio_path)
            text = result["text"]
            st.subheader("Transcript")
            st.text_area("", text, height=200)
            del model  # Free memory
            torch.cuda.empty_cache()
        except Exception as e:
            st.error(f"Transcription failed: {str(e)}")
            st.stop()

    # Mindmap Generation
    with st.spinner("Building mindmap..."):
        nx_graph, gv_graph = create_mindmap(text)
        if nx_graph and gv_graph:
            viz_option = st.radio("Choose Visualization Type", ["NetworkX", "Graphviz"])
            
            if viz_option == "NetworkX":
                st.subheader("NetworkX Mindmap")
                plt_obj = draw_networkx_graph(nx_graph)
                st.pyplot(plt_obj)
                nx_img = save_graph_as_image(plt_obj)
                st.download_button("Download NetworkX MindMap as JPEG", data=nx_img, file_name="networkx_mindmap.jpeg", mime="image/jpeg")
            else:
                st.subheader("Graphviz Mindmap")
                st.graphviz_chart(gv_graph)
                gv_img = save_graphviz_as_image(gv_graph)
                if gv_img:
                    st.download_button("Download Graphviz MindMap as JPEG", data=gv_img, file_name="graphviz_mindmap.jpeg", mime="image/jpeg")
        else:
            st.warning("Could not extract meaningful relationships from the audio content.")

# --------------------------
# 6. Installation
# --------------------------
st.subheader("⚙️ Setup Instructions")
st.code("""
pip install streamlit openai-whisper torch numpy nltk matplotlib networkx graphviz transformers spacy
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"

# For Whisper:
# Download ffmpeg build and add to PATH

# For Graphviz:
# Windows: Download from graphviz.org and add to PATH
# Mac: brew install graphviz
# Linux: sudo apt install graphviz
""")
