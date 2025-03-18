import streamlit as st
import whisper
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import nltk
import re
from transformers import BertTokenizer, BertModel
from nltk.corpus import stopwords
import soundfile as sf
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import graphviz

st.title("Audio Transcription and Relationship Inference")
st.write("Upload an audio file or record your voice to transcribe and analyze relationships between sentences.")

option = st.radio("Select input method:", ("Record Audio", "Upload Audio"))
audio_path = None

def save_audio(audio_frames):
    audio_path = "recorded_audio.wav"
    with open(audio_path, "wb") as f:
        for frame in audio_frames:
            f.write(frame.to_ndarray().tobytes())
    return audio_path

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame):
        self.frames.append(frame)
        return frame

if option == "Record Audio":
    st.write("Click 'Start' to record and 'Stop' to save the audio.")
    webrtc_ctx = webrtc_streamer(key="audio", audio_receiver_size=256, media_stream_constraints={"audio": True})
    if webrtc_ctx and webrtc_ctx.state.playing and webrtc_ctx.audio_receiver:
        audio_frames = webrtc_ctx.audio_receiver.get_frames()
        if audio_frames:
            audio_path = save_audio(audio_frames)
            st.audio(audio_path)
        else:
            st.warning("No audio frames captured. Please check your microphone settings.")
elif option == "Upload Audio":
    uploaded_file = st.file_uploader("Upload an audio file (WAV format):", type=["wav"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            audio_path = tmp_file.name
            st.audio(audio_path)

if audio_path:
    model = whisper.load_model("base")
    transcription = model.transcribe(audio_path)
    text = transcription["text"]
    st.subheader("Transcription:")
    st.markdown("<div style='max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px;'>" + text + "</div>", unsafe_allow_html=True)

    try:
        nltk.download("punkt")
        sentences = nltk.sent_tokenize(text)
    except:
        sentences = re.split(r'(?<=[.!?])\s+', text)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")

    def get_bert_embeddings(sentence):
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    sentence_embeddings = [get_bert_embeddings(sent) for sent in sentences]

    def infer_relationships(sentences, embeddings):
        relationships = []
        threshold = 0.7
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                if sim > threshold:
                    relationships.append((sentences[i], sentences[j]))
        return relationships

    relations = infer_relationships(sentences, sentence_embeddings)

    def create_mindmap(relationships, central_theme):
        dot = graphviz.Digraph(comment='Mind Map')
        dot.attr(size="8,8!")
        dot.attr(rankdir="TB", ranksep="1.5", nodesep="0.5")
        dot.node("Central Theme", central_theme)
        added_nodes = {}
        for sent1, sent2 in relationships:
            if sent1 not in added_nodes:
                dot.node(sent1, sent1)
                added_nodes[sent1] = sent1
            if sent2 not in added_nodes:
                dot.node(sent2, sent2)
                added_nodes[sent2] = sent2
            dot.edge(added_nodes[sent1], added_nodes[sent2])
        return dot

    # def create_mindmap(relationships):
    #     dot = graphviz.Digraph(comment='Mind Map')
    #     dot.attr(size="8,8!")
    #     dot.attr(rankdir="TB", ranksep="1.5", nodesep="0.5")  # Top-to-Bottom layout
    #     dot.node("Central Theme", "Main Topic")
    #     for idx, (sent1, sent2) in enumerate(relationships[:10]):
    #         dot.node(f"node{idx}1", sent1)
    #         dot.node(f"node{idx}2", sent2)
    #         dot.edge("Central Theme", f"node{idx}1")
    #         dot.edge(f"node{idx}1", f"node{idx}2")
    #     return dot


    st.subheader("Inferred Relationships:")
    st.markdown("<div style='max-height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px;'>", unsafe_allow_html=True)
    for relation in relations:
        st.write(f"- {relation[0]} ↔ {relation[1]}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Mind Map:")
    mindmap = create_mindmap(relations, "THEME")
    st.graphviz_chart(mindmap.source)
    st.download_button("Download Mind Map as .dot", mindmap.source, file_name="mindmap.dot")

    mindmap.render("mindmap", format="png")
    with open("mindmap.png", "rb") as img_file:
        st.download_button("Download Mind Map as Image", img_file, file_name="mindmap.png", mime="image/png")

st.subheader("Setup Instructions:")
st.code("""
pip install streamlit openai-whisper torch transformers nltk networkx matplotlib numpy soundfile streamlit-webrtc av graphviz
streamlit run app.py
""")
