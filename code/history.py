# # # import streamlit as st
# # # import whisper
# # # import torch
# # # import numpy as np
# # # import networkx as nx
# # # import matplotlib.pyplot as plt
# # # import nltk
# # # import re
# # # from transformers import BertTokenizer, BertModel
# # # from nltk.corpus import stopwords
# # # import soundfile as sf
# # # import tempfile
# # # import os
# # # from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
# # # import av
# # # import graphviz
# # # import nlp

# # # st.title("Audio Transcription and Relationship Inference")
# # # st.write("Upload an audio file or record your voice to transcribe and analyze relationships between sentences.")

# # # option = st.radio("Select input method:", ("Record Audio", "Upload Audio"))
# # # audio_path = None

# # # def save_audio(audio_frames):
# # #     audio_path = "recorded_audio.wav"
# # #     with open(audio_path, "wb") as f:
# # #         for frame in audio_frames:
# # #             f.write(frame.to_ndarray().tobytes())
# # #     return audio_path

# # # class AudioProcessor(AudioProcessorBase):
# # #     def __init__(self):
# # #         self.frames = []

# # #     def recv(self, frame):
# # #         self.frames.append(frame)
# # #         return frame

# # # if option == "Record Audio":
# # #     st.write("Click 'Start' to record and 'Stop' to save the audio.")
# # #     webrtc_ctx = webrtc_streamer(key="audio", audio_receiver_size=256, media_stream_constraints={"audio": True})
# # #     if webrtc_ctx and webrtc_ctx.state.playing and webrtc_ctx.audio_receiver:
# # #         audio_frames = webrtc_ctx.audio_receiver.get_frames()
# # #         if audio_frames:
# # #             audio_path = save_audio(audio_frames)
# # #             st.audio(audio_path)
# # #         else:
# # #             st.warning("No audio frames captured. Please check your microphone settings.")
# # # elif option == "Upload Audio":
# # #     uploaded_file = st.file_uploader("Upload an audio file (WAV format):", type=["wav"])
# # #     if uploaded_file is not None:
# # #         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
# # #             tmp_file.write(uploaded_file.read())
# # #             audio_path = tmp_file.name
# # #             st.audio(audio_path)

# # # if audio_path:
# # #     model = whisper.load_model("base")
# # #     transcription = model.transcribe(audio_path)
# # #     text = transcription["text"]
# # #     st.subheader("Transcription:")
# # #     st.markdown("<div style='max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px;'>" + text + "</div>", unsafe_allow_html=True)

# # #     try:
# # #         nltk.download("punkt")
# # #         sentences = nltk.sent_tokenize(text)
# # #     except:
# # #         sentences = re.split(r'(?<=[.!?])\s+', text)

# # #     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# # #     bert_model = BertModel.from_pretrained("bert-base-uncased")

# # #     def get_bert_embeddings(sentence):
# # #         inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
# # #         with torch.no_grad():
# # #             outputs = bert_model(**inputs)
# # #         return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# # #     sentence_embeddings = [get_bert_embeddings(sent) for sent in sentences]

# # #     def infer_relationships(sentences, embeddings):
# # #         relationships = []
# # #         threshold = 0.7
# # #         for i in range(len(embeddings)):
# # #             for j in range(i + 1, len(embeddings)):
# # #                 sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
# # #                 if sim > threshold:
# # #                     relationships.append((sentences[i], sentences[j]))
# # #         return relationships

# # #     relations = infer_relationships(sentences, sentence_embeddings)

# # #     def extract_keywords(text):
# # #         """Extracts key topics, subtopics, and notable figures from text."""

# # #     # Ensure text is a string, not a list of tuples
# # #         if isinstance(text, list):
# # #             text = " ".join([t[0] if isinstance(t, tuple) else str(t) for t in text])

# # #         doc = nlp(text)

# # #         # Extract word frequency (excluding stopwords & punctuation)
# # #         word_freq = Counter(token.text.lower() for token in doc if token.is_alpha and not token.is_stop)

# # #     # Extract Named Entities (People & Organizations)
# # #         figures = {ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG"]}
# # #         numbers = {ent.text for ent in doc.ents if ent.label_ in ["MONEY", "PERCENT", "DATE", "QUANTITY"]}
# # #     # Extract more words for a detailed map
# # #         sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

# # #     # Select more main topics (top 10)
# # #         main_topics = [word for word, freq in sorted_words[:10]]
# # #         topic_tree = {topic: [] for topic in main_topics}

# # #     # Assign subtopics dynamically
# # #         for word, freq in sorted_words[10:30]:
# # #             for topic in main_topics:
# # #                 if word.startswith(topic[:3]):  # Simple heuristic
# # #                     topic_tree[topic].append(word)
# # #                     break

# # #         return topic_tree, figures, numbers

# # #     def generate_mind_map_from_text(text):
# # #         """Creates a Graphviz mind map with extracted topics and notable figures."""
# # #         keywords, figures, numbers = extract_keywords(text)

# # #         dot = graphviz.Digraph(format="png")
# # #         dot.attr(rankdir="TB", size="10")  # Vertical Layout

# # #     # Root Node
# # #         dot.node("Main Idea", shape="box", style="filled", fillcolor="lightblue")
# # # # First Layer (Main Topics)
# # #         topic_layers = {}
# # #         for topic in keywords.keys():
# # #             dot.node(topic, shape="ellipse", style="filled", fillcolor="lightyellow")
# # #             dot.edge("Main Idea", topic)
# # #             topic_layers[topic] = []  # Storing topics to further expand

# # #     # Second & Third Layer (Subtopics & Sub-subtopics)
# # #         for topic, subtopics in keywords.items():
# # #             for subtopic in subtopics:
# # #                 dot.node(subtopic, shape="circle", style="filled", fillcolor="lightgray")
# # #                 dot.edge(topic, subtopic)
# # #                 topic_layers[topic].append(subtopic)  # Storing for deeper levels

# # #     # Fourth - Tenth Layer (Random Expansion)
# # #         for topic, subtopics in topic_layers.items():
# # #             for subtopic in subtopics:
# # #                 for i in range(random.randint(2, 4)):  # Randomly adding deeper layers
# # #                     detail = f"{subtopic}_D{i}"
# # #                     dot.node(detail, shape="circle", style="filled", fillcolor="white")
# # #                     dot.edge(subtopic, detail)

# # #                 # Even deeper levels
# # #                     for j in range(random.randint(1, 3)):
# # #                         fact = f"{detail}_F{j}"
# # #                         dot.node(fact, shape="box", style="filled", fillcolor="lightblue")
# # #                         dot.edge(detail, fact)

# # #     # Notable Figures & Organizations
# # #         for figure in figures:
# # #             dot.node(figure, shape="diamond", style="filled", fillcolor="lightgreen")
# # #             dot.edge("Main Idea", figure)

# # #     # Adding their contributions
# # #         for i in range(random.randint(2, 4)):
# # #             contribution = f"{figure}_C{i}"
# # #             dot.node(contribution, shape="parallelogram", style="filled", fillcolor="lightpink")
# # #             dot.edge(figure, contribution)

# # #         # Important Numbers (Dates, Percentages, Money)
# # #         for number in numbers:
# # #             dot.node(number, shape="square", style="filled", fillcolor="red")
# # #             dot.edge("Main Idea", number)
# # #         return dot

# # #     st.subheader("Inferred Relationships:")
# # #     st.markdown("<div style='max-height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px;'>", unsafe_allow_html=True)
# # #     for relation in relations:
# # #         st.write(f"- {relation[0]} ↔ {relation[1]}")
# # #     st.markdown("</div>", unsafe_allow_html=True)

# # # # === Streamlit App Integration ===
# # #     st.subheader("Mind Map:")
# # #     mindmap = generate_mind_map_from_text(relations)
# # #     st.graphviz_chart(mindmap.source)  # Display Graphviz Mind Map

# # #     # Provide download option
# # #     st.download_button("Download Mind Map", mindmap.source, file_name="mindmap.dot")
# # # st.subheader("Setup Instructions:")
# # # st.code("""
# # #     pip install streamlit openai-whisper torch transformers nltk networkx matplotlib numpy soundfile streamlit-webrtc av graphviz
# # #     streamlit run app.py
# # #     """)

# # #     # def create_mindmap(relationships, central_theme):
# # #     #     dot = graphviz.Digraph(comment='Mind Map')
# # #     #     dot.attr(size="8,8!")
# # #     #     dot.attr(rankdir="TB", ranksep="1.5", nodesep="0.5")
# # #     #     dot.node("Central Theme", central_theme)
# # #     #     added_nodes = {}
# # #     #     for sent1, sent2 in relationships:
# # #     #         if sent1 not in added_nodes:
# # #     #             dot.node(sent1, sent1)
# # #     #             added_nodes[sent1] = sent1
# # #     #         if sent2 not in added_nodes:
# # #     #             dot.node(sent2, sent2)
# # #     #             added_nodes[sent2] = sent2
# # #     #         dot.edge(added_nodes[sent1], added_nodes[sent2])
# # #     #     return dot

# # #     # def create_mindmap(relationships):
# # #     #     dot = graphviz.Digraph(comment='Mind Map')
# # #     #     dot.attr(size="8,8!")
# # #     #     dot.attr(rankdir="TB", ranksep="1.5", nodesep="0.5")  # Top-to-Bottom layout
# # #     #     dot.node("Central Theme", "Main Topic")
# # #     #     for idx, (sent1, sent2) in enumerate(relationships[:10]):
# # #     #         dot.node(f"node{idx}1", sent1)
# # #     #         dot.node(f"node{idx}2", sent2)
# # #     #         dot.edge("Central Theme", f"node{idx}1")
# # #     #         dot.edge(f"node{idx}1", f"node{idx}2")
# # #     #     return dot


# # # #     st.subheader("Inferred Relationships:")
# # # #     st.markdown("<div style='max-height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px;'>", unsafe_allow_html=True)
# # # #     for relation in relations:
# # # #         st.write(f"- {relation[0]} ↔ {relation[1]}")
# # # #     st.markdown("</div>", unsafe_allow_html=True)

# # # #     st.subheader("Mind Map:")
# # # #     mindmap = create_mindmap(relations, "THEME")
# # # #     st.graphviz_chart(mindmap.source)
# # # #     st.download_button("Download Mind Map as .dot", mindmap.source, file_name="mindmap.dot")

# # # #     mindmap.render("mindmap", format="png")
# # # #     with open("mindmap.png", "rb") as img_file:
# # # #         st.download_button("Download Mind Map as Image", img_file, file_name="mindmap.png", mime="image/png")

# # # # st.subheader("Setup Instructions:")
# # # # st.code("""
# # # # pip install streamlit openai-whisper torch transformers nltk networkx matplotlib numpy soundfile streamlit-webrtc av graphviz
# # # # streamlit run app.py
# # # # """)
# # import streamlit as st
# # import whisper
# # import torch
# # import numpy as np
# # import re
# # from transformers import BertTokenizer, BertModel
# # import nltk
# # import graphviz
# # from collections import Counter
# # import random
# # import spacy 
# # import soundfile as sf
# # import tempfile
# # import os
# # from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
# # import av

# # # Load NLP models
# # try:
# #     nlp = spacy.load("en_core_web_sm")
# # except:
# #     st.error("Please install the English language model: python -m spacy download en_core_web_sm")
# #     st.stop()

# # st.title("Audio Transcription and Mindmap Generator")
# # st.write("Upload an audio file or record your voice to create a mindmap from the content.")

# # # Audio input section
# # option = st.radio("Select input method:", ("Record Audio", "Upload Audio"))
# # audio_path = None

# # def save_audio(audio_frames):
# #     audio_path = "recorded_audio.wav"
# #     with open(audio_path, "wb") as f:
# #         for frame in audio_frames:
# #             f.write(frame.to_ndarray().tobytes())
# #     return audio_path

# # class AudioProcessor(AudioProcessorBase):
# #     def __init__(self):
# #         self.frames = []

# #     def recv(self, frame):
# #         self.frames.append(frame)
# #         return frame

# # if option == "Record Audio":
# #     st.write("Click 'Start' to record and 'Stop' to save the audio.")
# #     webrtc_ctx = webrtc_streamer(key="audio", audio_receiver_size=256, media_stream_constraints={"audio": True})
# #     if webrtc_ctx and webrtc_ctx.state.playing and webrtc_ctx.audio_receiver:
# #         audio_frames = webrtc_ctx.audio_receiver.get_frames()
# #         if audio_frames:
# #             audio_path = save_audio(audio_frames)
# #             st.audio(audio_path)
# #         else:
# #             st.warning("No audio frames captured. Please check your microphone settings.")
# # elif option == "Upload Audio":
# #     uploaded_file = st.file_uploader("Upload an audio file (WAV format):", type=["wav"])
# #     if uploaded_file is not None:
# #         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
# #             tmp_file.write(uploaded_file.read())
# #             audio_path = tmp_file.name
# #             st.audio(audio_path)

# # if audio_path and os.path.exists(audio_path):
# #     # Transcription
# #     with st.spinner("Transcribing audio..."):
# #         model = whisper.load_model("base")
# #         transcription = model.transcribe(audio_path)
# #         text = transcription["text"]
        
# #     st.subheader("Transcription:")
# #     st.markdown(f"<div style='max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px;'>{text}</div>", 
# #                 unsafe_allow_html=True)

# #     # Sentence processing
# #     with st.spinner("Processing text..."):
# #         try:
# #             nltk.download("punkt")
# #             sentences = nltk.sent_tokenize(text)
# #         except:
# #             sentences = re.split(r'(?<=[.!?])\s+', text)

# #         if len(sentences) < 2:
# #             st.warning("Not enough sentences found to analyze relationships.")
# #             st.stop()

# #         # Get BERT embeddings
# #         tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# #         bert_model = BertModel.from_pretrained("bert-base-uncased")

# #         @st.cache_resource
# #         def get_bert_embeddings(sentence):
# #             inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
# #             with torch.no_grad():
# #                 outputs = bert_model(**inputs)
# #             return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# #         sentence_embeddings = [get_bert_embeddings(sent) for sent in sentences]

# #         # Find the most central sentence (closest to average embedding)
# #         avg_embedding = np.mean(sentence_embeddings, axis=0)
# #         central_idx = np.argmin([np.linalg.norm(emb - avg_embedding) for emb in sentence_embeddings])
# #         central_topic = sentences[central_idx]

# #     # Keyword extraction
# #     def extract_keywords(text):
# #         """Extracts key topics, subtopics, and notable figures from text."""
# #         doc = nlp(text)
        
# #         # Extract word frequency (excluding stopwords & punctuation)
# #         word_freq = Counter(token.text.lower() for token in doc 
# #                           if token.is_alpha and not token.is_stop and len(token.text) > 2)
        
# #         # Extract Named Entities
# #         figures = {ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG"]}
# #         numbers = {ent.text for ent in doc.ents if ent.label_ in ["MONEY", "PERCENT", "DATE", "QUANTITY"]}
        
# #         # Get main topics (top 10 most frequent words)
# #         sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
# #         main_topics = [word for word, freq in sorted_words[:10] if freq > 1]
        
# #         # If we don't have enough topics, add some from entities
# #         if len(main_topics) < 5:
# #             main_topics.extend(list(figures)[:5-len(main_topics)])
        
# #         topic_tree = {topic: [] for topic in main_topics}
        
# #         # Assign subtopics based on co-occurrence
# #         for word, freq in sorted_words[10:30]:
# #             for topic in main_topics:
# #                 if word in topic or topic in word:  # Simple heuristic
# #                     topic_tree[topic].append(word)
# #                     break
        
# #         return topic_tree, figures, numbers, central_topic

# #     # Generate mindmap
# #     def generate_mind_map(text):
# #         """Creates a Graphviz mind map with extracted topics and relationships."""
# #         keywords, figures, numbers, central_topic = extract_keywords(text)
        
# #         dot = graphviz.Digraph()
# #         dot.attr(rankdir="TB", size="10", ratio="auto")
        
# #         # Central node with the most important topic
# #         dot.node("CENTRAL", central_topic[:50], shape="doublecircle", style="filled", fillcolor="lightblue", fontsize="16")
        
# #         # Main topics (first level)
# #         for i, topic in enumerate(keywords.keys()):
# #             dot.node(f"TOPIC_{i}", topic[:30], shape="ellipse", style="filled", fillcolor="lightyellow")
# #             dot.edge("CENTRAL", f"TOPIC_{i}")
            
# #             # Subtopics (second level)
# #             for j, subtopic in enumerate(keywords[topic][:5]):  # Limit to 5 subtopics per main topic
# #                 dot.node(f"SUBTOPIC_{i}_{j}", subtopic[:25], shape="box", style="filled", fillcolor="lightgray")
# #                 dot.edge(f"TOPIC_{i}", f"SUBTOPIC_{i}_{j}")
                
# #                 # Details (third level) - limited to 3 per subtopic
# #                 for k in range(min(3, len(subtopic.split())//2)):
# #                     detail = ' '.join(subtopic.split()[k*2:(k+1)*2])
# #                     if detail.strip():
# #                         dot.node(f"DETAIL_{i}_{j}_{k}", detail[:20], shape="note", style="filled", fillcolor="white")
# #                         dot.edge(f"SUBTOPIC_{i}_{j}", f"DETAIL_{i}_{j}_{k}")
        
# #         # Notable figures
# #         for i, figure in enumerate(figures):
# #             if i < 5:  # Limit to 5 figures
# #                 dot.node(f"FIGURE_{i}", figure[:25], shape="diamond", style="filled", fillcolor="lightgreen")
# #                 dot.edge("CENTRAL", f"FIGURE_{i}")
        
# #         # Important numbers
# #         for i, number in enumerate(numbers):
# #             if i < 5:  # Limit to 5 numbers
# #                 dot.node(f"NUMBER_{i}", number[:15], shape="square", style="filled", fillcolor="lightpink")
# #                 dot.edge("CENTRAL", f"NUMBER_{i}")
        
# #         return dot

# #     # Display mindmap
# #     st.subheader("Generated Mindmap:")
# #     with st.spinner("Generating mindmap..."):
# #         mindmap = generate_mind_map(text)
# #         st.graphviz_chart(mindmap)
    
# #     # Download options
# #     col1, col2 = st.columns(2)
# #     with col1:
# #         st.download_button("Download Mindmap (DOT)", mindmap.source, file_name="mindmap.dot")
# #     with col2:
# #         mindmap.format = 'png'
# #         mindmap.render('temp_mindmap', cleanup=True)
# #         with open('temp_mindmap.png', 'rb') as f:
# #             st.download_button("Download Mindmap (PNG)", f, file_name="mindmap.png")

# # # Setup instructions
# # st.subheader("Setup Instructions:")
# # st.code("""
# # pip install streamlit openai-whisper torch transformers nltk spacy soundfile streamlit-webrtc av graphviz
# # python -m spacy download en_core_web_sm
# # streamlit run app.py
# # """)


# # import streamlit as st
# # import whisper
# # import torch
# # import numpy as np
# # import nltk
# # import graphviz
# # import tempfile
# # import os
# # import sys

# # # --------------------------
# # # 1. NLTK Data Fix (Fixes punkt_tab error)
# # # --------------------------
# # def fix_nltk_data():
# #     try:
# #         # First try the standard download
# #         try:
# #             nltk.data.find('tokenizers/punkt')
# #         except LookupError:
# #             nltk.download('punkt')
        
# #         # Direct download of punkt_tab if still missing
# #         punkt_tab_path = os.path.join(nltk.data.path[0], 'tokenizers/punkt_tab/english')
# #         if not os.path.exists(punkt_tab_path):
# #             nltk.download('punkt', download_dir=nltk.data.path[0])
# #             nltk.download('punkt_tab', download_dir=nltk.data.path[0])
        
# #         # Verify all required data
# #         nltk.download('stopwords')
# #         nltk.download('averaged_perceptron_tagger')
# #     except Exception as e:
# #         st.error(f"NLTK setup failed: {str(e)}")
# #         st.error("""
# #         Please manually download NLTK data:
# #         1. Open Python interpreter
# #         2. Run these commands:
# #         import nltk
# #         nltk.download('punkt')
# #         nltk.download('punkt_tab')
# #         nltk.download('stopwords')
# #         nltk.download('averaged_perceptron_tagger')
# #         """)
# #         st.stop()

# # with st.spinner("Loading NLP resources..."):
# #     fix_nltk_data()

# # # --------------------------
# # # 2. Graphviz Verification (Fixes dot.exe error)
# # # --------------------------
# # # def verify_graphviz():
# # #     try:
# # #         # Check if graphviz binaries are in PATH
# # #         if sys.platform.startswith('win'):
# # #             graphviz_bin = os.path.join(os.environ.get('PROGRAMFILES', 'C:\\Program Files'), 'Graphviz', 'bin')
# # #             if graphviz_bin not in os.environ['PATH']:
# # #                 os.environ['PATH'] += os.pathsep + graphviz_bin
        
# # #         # Verify installation
# # #         graphviz.backend.Executable('dot').version()
# # #         return True
# # #     except:
# # #         return False

# # # if not verify_graphviz():
# # #     st.error("""
# # #     Graphviz not properly installed. Please:
# # #     1. Download from https://graphviz.org/download/
# # #     2. Run the installer
# # #     3. Add to PATH: C:\\Program Files\\Graphviz\\bin\\
# # #     4. RESTART your computer
# # #     """)
# # #     st.stop()

# # # 1. Configure Graphviz path - Universal solution
# # graphviz_bin_path = r'C:\Program Files\Graphviz\bin'

# # # Add to system PATH
# # os.environ["PATH"] = graphviz_bin_path + os.pathsep + os.environ["PATH"]

# # # 2. Direct executable specification (works with all graphviz versions)
# # if sys.platform.startswith('win'):
# #     config = {
# #         'engine': os.path.join(graphviz_bin_path, 'dot.exe'),
# #         'format': 'png'
# #     }
# # else:
# #     config = {'format': 'png'}

# # # 3. Verification function
# # def verify_graphviz():
# #     try:
# #         # Create simple graph to test
# #         test = graphviz.Digraph(**config)
# #         test.node('test')
# #         test.render('test_graph', cleanup=True, format='png')
# #         os.remove('test_graph.png')
# #         return True
# #     except Exception as e:
# #         st.error(f"Graphviz test failed: {str(e)}")
# #         return False

# # # Check installation
# # if not verify_graphviz():
# #     st.error("""
# #     Graphviz not properly configured. Please:
# #     1. Download from https://graphviz.org/download/
# #     2. Install to C:\\Program Files\\Graphviz
# #     3. Restart your computer
# #     """)
# #     st.stop()

# # # --------------------------
# # # Main Application
# # # --------------------------
# # st.title("🎙️ Audio to Mindmap Converter")
# # st.write("Upload an audio file to generate a mindmap")

# # # Audio Upload
# # uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])
# # if uploaded_file is not None:
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
# #         tmp_file.write(uploaded_file.read())
# #         audio_path = tmp_file.name
# #         st.audio(audio_path)

# #     # Transcription
# #     with st.spinner("Transcribing audio..."):
# #         try:
# #             model = whisper.load_model("base")
# #             result = model.transcribe(audio_path)
# #             text = result["text"]
# #         except Exception as e:
# #             st.error(f"Transcription failed: {str(e)}")
# #             st.stop()

# #     st.subheader("Transcript")
# #     st.write(text)

# #     # Mindmap Generation
# #     with st.spinner("Creating mindmap..."):
# #         try:
# #             # Create a simple mindmap
# #             dot = graphviz.Digraph()
# #             dot.attr(rankdir='TB', size='12,12')
            
# #             # Central node
# #             sentences = nltk.sent_tokenize(text)
# #             central = sentences[0][:30] + "..." if len(sentences[0]) > 30 else sentences[0]
# #             dot.node('central', central, shape='circle', style='filled', fillcolor='lightblue')
            
# #             # Add main ideas
# #             for i, sent in enumerate(sentences[1:6]):
# #                 dot.node(f'node_{i}', sent[:30] + "...", shape='box')
# #                 dot.edge('central', f'node_{i}')
            
# #             # Display
# #             st.graphviz_chart(dot)
            
# #             # Download options
# #             dot.render('mindmap', format='png', cleanup=True)
# #             with open('mindmap.png', 'rb') as f:
# #                 st.download_button("Download Mindmap", f, file_name="mindmap.png")
# #         except Exception as e:
# #             st.error(f"Mindmap creation failed: {str(e)}")

# # # --------------------------
# # # Complete Setup Guide
# # # --------------------------
# # st.subheader("Complete Setup Instructions")
# # st.markdown("""
# # 1. **Install Python Packages**:
# # ```bash
# # pip install streamlit openai-whisper torch nltk graphviz""")



# # import os
# # import sys
# # import streamlit as st
# # import whisper
# # import torch
# # import nltk
# # import tempfile
# # from collections import Counter
# # import matplotlib.pyplot as plt
# # import networkx as nx

# # # --------------------------
# # # 1. NLTK Setup (Fixes punkt_tab error)
# # # --------------------------
# # try:
# #     nltk.data.find('tokenizers/punkt')
# # except LookupError:
# #     nltk.download('punkt')

# # try:
# #     nltk.data.find('corpora/stopwords')
# # except LookupError:
# #     nltk.download('stopwords')

# # # --------------------------
# # # 2. Pure Python Mindmap Generator (No Graphviz needed)
# # # --------------------------
# # def create_mindmap(text):
# #     """Create mindmap using networkx and matplotlib"""
# #     plt.figure(figsize=(12, 8))
# #     G = nx.Graph()
    
# #     # Process text
# #     sentences = nltk.sent_tokenize(text)
# #     words = [word.lower() for word in nltk.word_tokenize(text) 
# #             if word.isalpha() and word.lower() not in nltk.corpus.stopwords.words('english')]
# #     word_freq = Counter(words)
    
# #     # Central node (most frequent word or first sentence)
# #     central = max(word_freq.items(), key=lambda x: x[1])[0] if word_freq else sentences[0][:20]
# #     G.add_node(central, size=3000, color='lightblue')
    
# #     # Add main nodes (top 5 frequent words)
# #     for i, (word, freq) in enumerate(word_freq.most_common(5)):
# #         G.add_node(word, size=1500 + freq*100, color='lightgreen')
# #         G.add_edge(central, word)
        
# #         # Add related sentences
# #         for j, sent in enumerate([s for s in sentences if word in s.lower()][:2]):
# #             sent_short = ' '.join(sent.split()[:5]) + ('...' if len(sent.split()) > 5 else '')
# #             G.add_node(f"{word}_{j}", label=sent_short, size=800, color='lightyellow')
# #             G.add_edge(word, f"{word}_{j}")
    
# #     # Draw the graph
# #     pos = nx.spring_layout(G, k=0.5)
# #     colors = [G.nodes[n]['color'] for n in G.nodes()]
# #     sizes = [G.nodes[n]['size'] for n in G.nodes()]
# #     labels = {n: n if 'label' not in G.nodes[n] else G.nodes[n]['label'] for n in G.nodes()}
    
# #     nx.draw(G, pos, with_labels=True, labels=labels, 
# #             node_color=colors, node_size=sizes,
# #             font_size=10, edge_color='gray')
# #     plt.axis('off')
# #     return plt

# # # --------------------------
# # # 3. Streamlit App
# # # --------------------------
# # st.title("🎙️ Audio to Mindmap Converter")
# # st.write("Upload an audio file to generate a mindmap (No Graphviz required)")

# # # Audio processing
# # uploaded_file = st.file_uploader("Choose audio file", type=["wav", "mp3"])
# # if uploaded_file:
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
# #         tmp_file.write(uploaded_file.read())
# #         audio_path = tmp_file.name
    
# #     # Transcription
# #     with st.spinner("Transcribing audio..."):
# #         try:
# #             model = whisper.load_model("base")
# #             result = model.transcribe(audio_path)
# #             text = result["text"]
# #             st.text_area("Transcript", text, height=200)
# #         except Exception as e:
# #             st.error(f"Transcription failed: {str(e)}")
# #             st.stop()
    
# #     # Mindmap generation
# #     with st.spinner("Creating mindmap..."):
# #         try:
# #             fig = create_mindmap(text)
# #             st.pyplot(fig)
# #         except Exception as e:
# #             st.error(f"Mindmap creation failed: {str(e)}")

# # # --------------------------
# # # 4. Installation Instructions
# # # --------------------------
# # st.subheader("⚙️ Setup Instructions")
# # st.code("""
# # # Core requirements
# # pip install streamlit openai-whisper torch nltk matplotlib networkx

# # # NLTK data (run in Python)
# # import nltk
# # nltk.download('punkt')
# # nltk.download('stopwords')
# # """)


# # import os
# # import re
# # import streamlit as st
# # import whisper
# # import torch
# # import nltk
# # import tempfile
# # from collections import Counter
# # import matplotlib.pyplot as plt
# # import networkx as nx
# # from textblob import TextBlob  # For phrase extraction

# # # --------------------------
# # # 1. NLP Setup
# # # --------------------------
# # try:
# #     nltk.data.find('tokenizers/punkt')
# #     nltk.data.find('corpora/stopwords')
# # except LookupError:
# #     nltk.download('punkt')
# #     nltk.download('stopwords')

# # # --------------------------
# # # 2. Enhanced Text Processing
# # # --------------------------
# # def extract_key_phrases(text, max_phrase_len=4):
# #     """Extract meaningful phrases using noun phrase chunking"""
# #     blob = TextBlob(text)
# #     phrases = []
    
# #     # Get noun phrases and filter by length
# #     for np in blob.noun_phrases:
# #         words = np.split()
# #         if 2 <= len(words) <= max_phrase_len:
# #             phrases.append(np)
    
# #     # Also include important verbs
# #     pos_tags = nltk.pos_tag(nltk.word_tokenize(text))
# #     important_verbs = [word.lower() for word, pos in pos_tags 
# #                       if pos.startswith('VB') and word.lower() not in nltk.corpus.stopwords.words('english')]
    
# #     return list(set(phrases + important_verbs))  # Remove duplicates

# # def select_central_theme(text, phrases):
# #     """Select the most representative central theme"""
# #     sentences = nltk.sent_tokenize(text)
# #     if not sentences:
# #         return "Main Topic"
    
# #     # Find sentence containing most frequent phrases
# #     phrase_counts = Counter(phrases)
# #     if phrase_counts:
# #         top_phrase = phrase_counts.most_common(1)[0][0]
# #         for sent in sentences:
# #             if top_phrase in sent.lower():
# #                 return ' '.join(sent.split()[:8]) + ('...' if len(sent.split()) > 8 else '')
    
# #     return ' '.join(sentences[0].split()[:8]) + ('...' if len(sentences[0].split()) > 8 else '')

# # # --------------------------
# # # 3. Refined Mindmap Generator
# # # --------------------------
# # def create_mindmap(text):
# #     plt.figure(figsize=(14, 10))
# #     G = nx.Graph()
    
# #     # Extract key content
# #     phrases = extract_key_phrases(text)
# #     central_theme = select_central_theme(text, phrases)
    
# #     # Add central node
# #     G.add_node(central_theme, size=3500, color='#4E79A7', 
# #                fontsize=14, fontweight='bold')
    
# #     # Group phrases by similarity to avoid duplicates
# #     unique_concepts = []
# #     for phrase in phrases:
# #         if not any(phrase in concept or concept in phrase for concept in unique_concepts):
# #             unique_concepts.append(phrase)
    
# #     # Add main nodes (top 5 unique concepts)
# #     for i, concept in enumerate(unique_concepts[:5]):
# #         G.add_node(concept, size=2000, color='#F28E2B')
# #         G.add_edge(central_theme, concept)
        
# #         # Add supporting facts (sentences containing the concept)
# #         relevant_sents = [sent for sent in nltk.sent_tokenize(text) 
# #                          if concept in sent.lower()][:2]
# #         for j, sent in enumerate(relevant_sents):
# #             # Clean and shorten sentence
# #             clean_sent = re.sub(r'\s+', ' ', sent).strip()[:50] + ('...' if len(sent) > 50 else '')
# #             node_id = f"{concept[:3]}_{j}"
# #             G.add_node(node_id, label=clean_sent, size=1200, color='#59A14F')
# #             G.add_edge(concept, node_id)
    
# #     # Draw the graph
# #     pos = nx.spring_layout(G, k=0.6, seed=42)  # Consistent layout
# #     node_colors = [G.nodes[n]['color'] for n in G.nodes()]
# #     node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
# #     labels = {n: n if 'label' not in G.nodes[n] else G.nodes[n]['label'] for n in G.nodes()}
    
# #     nx.draw(G, pos, with_labels=True, labels=labels,
# #             node_color=node_colors, node_size=node_sizes,
# #             font_size=10, edge_color='#79706E', width=1.5,
# #             alpha=0.9)
# #     plt.axis('off')
# #     plt.tight_layout()
# #     return plt

# # # --------------------------
# # # 4. Streamlit App
# # # --------------------------
# # st.title("🎙️ Audio to Mindmap Converter")
# # st.write("Upload audio to generate a structured mindmap")

# # uploaded_file = st.file_uploader("Choose file", type=["wav", "mp3"])
# # if uploaded_file:
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
# #         tmp_file.write(uploaded_file.read())
# #         audio_path = tmp_file.name
    
# #     # Transcription
# #     with st.spinner("Transcribing audio..."):
# #         try:
# #             model = whisper.load_model("base")
# #             result = model.transcribe(audio_path)
# #             text = result["text"]
# #             st.subheader("Transcript")
# #             st.text_area("", text, height=200)
# #         except Exception as e:
# #             st.error(f"Transcription failed: {str(e)}")
# #             st.stop()
    
# #     # Mindmap generation
# #     with st.spinner("Creating optimized mindmap..."):
# #         try:
# #             fig = create_mindmap(text)
# #             st.subheader("Mindmap")
# #             st.pyplot(fig)
# #         except Exception as e:
# #             st.error(f"Mindmap creation failed: {str(e)}")

# # # --------------------------
# # # 5. Installation
# # # --------------------------
# # st.subheader("⚙️ Setup Instructions")
# # st.code("""
# # pip install streamlit openai-whisper torch nltk matplotlib networkx textblob
# # python -m textblob.download_corpora
# # python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
# # """)




# # import os
# # import re
# # import streamlit as st
# # import whisper
# # import torch
# # import nltk
# # import tempfile
# # from collections import Counter
# # import matplotlib.pyplot as plt
# # import networkx as nx
# # from textblob import TextBlob

# # # --------------------------
# # # 1. NLP Setup
# # # --------------------------
# # try:
# #     nltk.data.find('tokenizers/punkt')
# #     nltk.data.find('corpora/stopwords')
# # except LookupError:
# #     nltk.download('punkt')
# #     nltk.download('stopwords')

# # # --------------------------
# # # 2. Enhanced Text Processing
# # # --------------------------
# # def extract_key_phrases(text):
# #     """Extract meaningful phrases using noun phrase chunking"""
# #     blob = TextBlob(text)
# #     phrases = []
    
# #     # Get noun phrases (2-4 words long)
# #     for np in blob.noun_phrases:
# #         words = np.split()
# #         if 2 <= len(words) <= 4:
# #             phrases.append(np)
    
# #     return list(set(phrases))  # Remove duplicates

# # def get_complete_sentences(text, max_sentences=10):
# #     """Get complete sentences with proper cleaning"""
# #     sentences = nltk.sent_tokenize(text)
# #     cleaned_sentences = []
    
# #     for sent in sentences[:max_sentences]:
# #         # Clean sentence while keeping it complete
# #         clean_sent = re.sub(r'\s+', ' ', sent).strip()
# #         clean_sent = re.sub(r'[^\w\s.,;?!-]', '', clean_sent)  # Remove special chars
# #         cleaned_sentences.append(clean_sent)
    
# #     return cleaned_sentences

# # # --------------------------
# # # 3. Mindmap Generator with Complete Sentences
# # # --------------------------
# # def create_mindmap(text):
# #     plt.figure(figsize=(16, 12))
# #     G = nx.Graph()
    
# #     # Extract key content
# #     phrases = extract_key_phrases(text)
# #     sentences = get_complete_sentences(text)
    
# #     # Select central theme (first sentence or most frequent phrase)
# #     central_theme = sentences[0] if sentences else (phrases[0] if phrases else "Main Topic")
    
# #     # Add central node with adjusted font size
# #     G.add_node(central_theme, size=4000, color='#4E79A7', 
# #                fontsize=12 if len(central_theme) > 50 else 14, 
# #                fontweight='bold')
    
# #     # Add main nodes (top phrases)
# #     for i, phrase in enumerate(phrases[:5]):  # Limit to 5 main branches
# #         G.add_node(phrase, size=2500, color='#F28E2B', fontsize=11)
# #         G.add_edge(central_theme, phrase)
        
# #         # Add supporting sentences containing this phrase
# #         relevant_sents = [sent for sent in sentences if phrase.lower() in sent.lower()][:2]
# #         for j, sent in enumerate(relevant_sents):
# #             node_id = f"{phrase[:3]}_{j}"
# #             # Use complete sentence with dynamic font sizing
# #             font_size = 10 if len(sent) > 80 else (11 if len(sent) > 50 else 12)
# #             G.add_node(node_id, label=sent, size=1500, color='#59A14F', fontsize=font_size)
# #             G.add_edge(phrase, node_id)
    
# #     # Layout and drawing
# #     pos = nx.spring_layout(G, k=0.7, seed=42)  # Consistent layout
# #     node_colors = [G.nodes[n]['color'] for n in G.nodes()]
# #     node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
    
# #     # Custom label drawing to handle long text
# #     nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
# #     nx.draw_networkx_edges(G, pos, edge_color='#79706E', width=1.5, alpha=0.7)
    
# #     # Draw labels with proper text wrapping
# #     for node, (x, y) in pos.items():
# #         label = G.nodes[node].get('label', node)
# #         font_size = G.nodes[node].get('fontsize', 10)
# #         font_weight = G.nodes[node].get('fontweight', 'normal')
        
# #         # Smart text wrapping
# #         if len(label) > 50:
# #             wrapped_text = '\n'.join([label[i:i+30] for i in range(0, len(label), 30)])
# #             plt.text(x, y, wrapped_text, 
# #                     ha='center', va='center',
# #                     fontsize=font_size, 
# #                     fontweight=font_weight,
# #                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
# #         else:
# #             plt.text(x, y, label, 
# #                     ha='center', va='center',
# #                     fontsize=font_size,
# #                     fontweight=font_weight)
    
# #     plt.axis('off')
# #     plt.tight_layout()
# #     return plt

# # # --------------------------
# # # 4. Streamlit App
# # # --------------------------
# # st.title("🎙️ Audio to Mindmap Converter")
# # st.write("Upload audio to generate a mindmap with complete sentences")

# # uploaded_file = st.file_uploader("Choose audio file", type=["wav", "mp3"])
# # if uploaded_file:
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
# #         tmp_file.write(uploaded_file.read())
# #         audio_path = tmp_file.name
    
# #     # Transcription
# #     with st.spinner("Transcribing audio..."):
# #         try:
# #             model = whisper.load_model("base")
# #             result = model.transcribe(audio_path)
# #             text = result["text"]
# #             st.subheader("Transcript")
# #             st.text_area("", text, height=200)
# #         except Exception as e:
# #             st.error(f"Transcription failed: {str(e)}")
# #             st.stop()
    
# #     # Mindmap generation
# #     with st.spinner("Creating mindmap with complete sentences..."):
# #         try:
# #             fig = create_mindmap(text)
# #             st.subheader("Mindmap")
# #             st.pyplot(fig)
# #         except Exception as e:
# #             st.error(f"Mindmap creation failed: {str(e)}")

# # # --------------------------
# # # 5. Installation
# # # --------------------------
# # st.subheader("⚙️ Setup Instructions")
# # st.code("""
# # pip install streamlit openai-whisper torch nltk matplotlib networkx textblob
# # python -m textblob.download_corpora
# # python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
# # """)


# import os
# import re
# import streamlit as st
# import whisper
# import torch
# import nltk
# import tempfile
# from collections import Counter
# import matplotlib.pyplot as plt
# import networkx as nx
# from textblob import TextBlob

# # --------------------------
# # 1. NLP Setup
# # --------------------------
# try:
#     nltk.data.find('tokenizers/punkt')
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('punkt')
#     nltk.download('stopwords')

# # --------------------------
# # 2. Enhanced Text Processing
# # --------------------------
# def extract_key_elements(text):
#     """Extract hierarchical elements from text"""
#     blob = TextBlob(text)
    
#     # Level 1: Main concepts (noun phrases)
#     main_concepts = []
#     for np in blob.noun_phrases:
#         words = np.split()
#         if 2 <= len(words) <= 3:  # Keep phrases short
#             main_concepts.append(np)
    
#     # Level 2: Key actions (verbs)
#     pos_tags = nltk.pos_tag(nltk.word_tokenize(text))
#     key_actions = [word.lower() for word, pos in pos_tags 
#                   if pos.startswith('VB') and word.lower() not in nltk.corpus.stopwords.words('english')]
    
#     # Level 3: Important specifics (named entities + numbers)
#     entities = []
#     for chunk in nltk.ne_chunk(pos_tags):
#         if hasattr(chunk, 'label'):
#             entities.append(' '.join(c[0] for c in chunk))
    
#     return {
#         'concepts': list(set(main_concepts))[:5],  # Limit to top 5
#         'actions': list(set(key_actions))[:5],
#         'entities': list(set(entities))[:5]
#     }

# # --------------------------
# # 3. Compact Hierarchical Mindmap
# # --------------------------
# def create_mindmap(text):
#     plt.figure(figsize=(14, 10))
#     G = nx.DiGraph()  # Use directed graph for hierarchy
    
#     elements = extract_key_elements(text)
#     sentences = nltk.sent_tokenize(text)
#     central_theme = sentences[0][:25] if sentences else "Main Topic"
    
#     # Add central node
#     G.add_node(central_theme, size=3000, color='#4E79A7', 
#                fontsize=12, fontweight='bold', style='filled')
    
#     # Level 1: Main Concepts
#     for i, concept in enumerate(elements['concepts']):
#         G.add_node(f"C_{i}", label=concept, size=2000, 
#                   color='#F28E2B', fontsize=11)
#         G.add_edge(central_theme, f"C_{i}")
        
#         # Level 2: Related Actions
#         for j, action in enumerate(elements['actions'][:2]):  # 2 actions per concept
#             if j >= len(elements['actions']): break
#             G.add_node(f"A_{i}_{j}", label=action, size=1500,
#                       color='#59A14F', fontsize=10)
#             G.add_edge(f"C_{i}", f"A_{i}_{j}")
            
#             # Level 3: Specific Entities
#             for k, entity in enumerate(elements['entities'][:2]):  # 2 entities per action
#                 if k >= len(elements['entities']): break
#                 G.add_node(f"E_{i}_{j}_{k}", label=entity[:15], size=1200,
#                           color='#B07AA1', fontsize=9)
#                 G.add_edge(f"A_{i}_{j}", f"E_{i}_{j}_{k}")
    
#     # Layout and drawing
#     pos = nx.nx_agraph.graphviz_layout(G, prog='dot')  # Hierarchical layout
#     node_colors = [G.nodes[n]['color'] for n in G.nodes()]
#     node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
    
#     nx.draw(G, pos, 
#             node_color=node_colors, 
#             node_size=node_sizes,
#             with_labels=True,
#             font_size=10,
#             edge_color='gray',
#             arrowsize=15,
#             width=1.5)
    
#     plt.axis('off')
#     plt.tight_layout()
#     return plt

# # --------------------------
# # 4. Streamlit App
# # --------------------------
# st.title("🎙️ Audio to Mindmap Converter")
# st.write("Upload audio to generate a compact hierarchical mindmap")

# uploaded_file = st.file_uploader("Choose audio file", type=["wav", "mp3"])
# if uploaded_file:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
#         tmp_file.write(uploaded_file.read())
#         audio_path = tmp_file.name
    
#     # Transcription
#     with st.spinner("Transcribing audio..."):
#         try:
#             model = whisper.load_model("base")
#             result = model.transcribe(audio_path)
#             text = result["text"]
#             st.subheader("Transcript")
#             st.text_area("", text, height=200)
#         except Exception as e:
#             st.error(f"Transcription failed: {str(e)}")
#             st.stop()
    
#     # Mindmap generation
#     with st.spinner("Creating compact mindmap..."):
#         try:
#             fig = create_mindmap(text)
#             st.subheader("Mindmap")
#             st.pyplot(fig)
#         except Exception as e:
#             st.error(f"Mindmap creation failed: {str(e)}")

# # --------------------------
# # 5. Installation
# # --------------------------
# st.subheader("⚙️ Setup Instructions")
# st.code("""
# pip install streamlit openai-whisper torch nltk matplotlib networkx textblob pygraphviz
# python -m textblob.download_corpora
# python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker')"
# """)





### NEW DOC C DRIVE

# # # import os
# # # import re
# # # import streamlit as st
# # # import whisper
# # # import torch
# # # import nltk
# # # import tempfile
# # # from collections import Counter
# # # import matplotlib.pyplot as plt
# # # import networkx as nx
# # # import graphviz
# # # from textblob import TextBlob

# # # # --------------------------
# # # # 1. NLP Setup
# # # # --------------------------
# # # try:
# # #     nltk.data.find('tokenizers/punkt')
# # #     nltk.data.find('corpora/stopwords')
# # #     nltk.data.find('taggers/averaged_perceptron_tagger')
# # # except LookupError:
# # #     nltk.download('punkt')
# # #     nltk.download('stopwords')
# # #     nltk.download('averaged_perceptron_tagger')

# # # # --------------------------
# # # # 2. Text Processing
# # # # --------------------------
# # # def extract_hierarchical_elements(text):
# # #     """Extract elements for 3-level hierarchy"""
# # #     blob = TextBlob(text)
# # #     pos_tags = nltk.pos_tag(nltk.word_tokenize(text))
    
# # #     # Level 1: Central Theme (first sentence)
# # #     sentences = nltk.sent_tokenize(text)
# # #     central_theme = ' '.join(sentences[0].split()[:5]) if sentences else "Main Topic"
    
# # #     # Level 2: Key Concepts (noun phrases)
# # #     concepts = []
# # #     for np in blob.noun_phrases:
# # #         words = np.split()
# # #         if 2 <= len(words) <= 3:  # Keep concise
# # #             concepts.append(np)
    
# # #     # Level 3: Specific Details (verbs + entities)
# # #     details = []
# # #     for word, pos in pos_tags:
# # #         if pos.startswith('VB') or pos in ['NNP', 'CD']:  # Verbs and proper nouns
# # #             if word.lower() not in nltk.corpus.stopwords.words('english'):
# # #                 details.append(word)
    
# # #     return {
# # #         'central': central_theme,
# # #         'concepts': list(set(concepts))[:5],  # Top 5 unique
# # #         'details': list(set(details))[:10]    # Top 10 unique
# # #     }

# # # # --------------------------
# # # # 3. NetworkX Graph (Matplotlib)
# # # # --------------------------
# # # def create_networkx_mindmap(text):
# # #     elements = extract_hierarchical_elements(text)
# # #     G = nx.DiGraph()
    
# # #     # Add nodes with style attributes
# # #     G.add_node(elements['central'], level=0, color='#4E79A7', size=3000)
    
# # #     # Add concepts (level 1)
# # #     for i, concept in enumerate(elements['concepts']):
# # #         G.add_node(concept, level=1, color='#F28E2B', size=2000)
# # #         G.add_edge(elements['central'], concept)
        
# # #         # Add details (level 2) - connect to relevant concepts
# # #         for detail in elements['details'][i*2:(i+1)*2]:  # 2 details per concept
# # #             if detail.lower() not in concept.lower():  # Avoid redundancy
# # #                 G.add_node(detail, level=2, color='#59A14F', size=1500)
# # #                 G.add_edge(concept, detail)
    
# # #     # Draw graph
# # #     plt.figure(figsize=(12, 8))
# # #     pos = nx.multipartite_layout(G, subset_key="level")
# # #     colors = [G.nodes[n]['color'] for n in G.nodes()]
# # #     sizes = [G.nodes[n]['size'] for n in G.nodes()]
    
# # #     nx.draw(G, pos, with_labels=True, 
# # #             node_color=colors, node_size=sizes,
# # #             font_size=10, edge_color='gray', width=1.5)
# # #     plt.title("NetworkX Mindmap")
# # #     plt.axis('off')
# # #     return plt

# # # # --------------------------
# # # # 4. Graphviz Graph
# # # # --------------------------
# # # def create_graphviz_mindmap(text):
# # #     elements = extract_hierarchical_elements(text)
# # #     dot = graphviz.Digraph(engine='dot')
    
# # #     # Graph style
# # #     dot.attr(rankdir='TB', size='12,12', ratio='auto')
    
# # #     # Central node
# # #     dot.node('central', elements['central'], 
# # #              shape='doublecircle', style='filled',
# # #              fillcolor='#4E79A7', fontsize='14')
    
# # #     # Concepts (level 1)
# # #     for i, concept in enumerate(elements['concepts']):
# # #         dot.node(f'concept_{i}', concept, 
# # #                 shape='ellipse', style='filled',
# # #                 fillcolor='#F28E2B')
# # #         dot.edge('central', f'concept_{i}')
        
# # #         # Details (level 2)
# # #         for j, detail in enumerate(elements['details'][i*2:(i+1)*2]):
# # #             dot.node(f'detail_{i}_{j}', detail,
# # #                     shape='box', style='filled',
# # #                     fillcolor='#59A14F')
# # #             dot.edge(f'concept_{i}', f'detail_{i}_{j}')
    
# # #     return dot

# # # # --------------------------
# # # # 5. Streamlit App
# # # # --------------------------
# # # st.title("🎙️ Audio to Mindmap Converter")
# # # st.write("Generates both NetworkX and Graphviz mindmaps")

# # # uploaded_file = st.file_uploader("Choose audio file", type=["wav", "mp3"])
# # # if uploaded_file:
# # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
# # #         tmp_file.write(uploaded_file.read())
# # #         audio_path = tmp_file.name
    
# # #     # Transcription
# # #     with st.spinner("Transcribing audio..."):
# # #         try:
# # #             model = whisper.load_model("base")
# # #             result = model.transcribe(audio_path)
# # #             text = result["text"]
# # #             st.subheader("Transcript")
# # #             st.text_area("", text, height=200)
# # #         except Exception as e:
# # #             st.error(f"Transcription failed: {str(e)}")
# # #             st.stop()
    
# # #     # Generate both mindmaps
# # #     col1, col2 = st.columns(2)
    
# # #     with col1:
# # #         st.subheader("NetworkX Version")
# # #         with st.spinner("Creating NetworkX mindmap..."):
# # #             try:
# # #                 nx_plot = create_networkx_mindmap(text)
# # #                 st.pyplot(nx_plot)
# # #             except Exception as e:
# # #                 st.error(f"NetworkX error: {str(e)}")
    
# # #     with col2:
# # #         st.subheader("Graphviz Version")
# # #         with st.spinner("Creating Graphviz mindmap..."):
# # #             try:
# # #                 gv_graph = create_graphviz_mindmap(text)
# # #                 st.graphviz_chart(gv_graph)
# # #             except Exception as e:
# # #                 st.error(f"Graphviz error: {str(e)}")
# # #                 st.info("Ensure Graphviz is installed and in PATH")

# # # # --------------------------
# # # # 6. Installation
# # # # --------------------------
# # # st.subheader("⚙️ Setup Instructions")
# # # st.code("""
# # # # Core requirements
# # # pip install streamlit openai-whisper torch nltk matplotlib networkx graphviz textblob

# # # # NLTK data
# # # python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"

# # # # Graphviz (Windows):
# # # 1. Download from https://graphviz.org/download/
# # # 2. Install to C:\Program Files\Graphviz
# # # 3. Add to PATH: C:\Program Files\Graphviz\bin\
# # # 4. Restart computer

# # # # TextBlob data
# # # python -m textblob.download_corpora
# # # """)




# # # import os
# # # import re
# # # import streamlit as st
# # # import whisper
# # # import torch
# # # import nltk
# # # import tempfile
# # # from collections import Counter
# # # import matplotlib.pyplot as plt
# # # import networkx as nx
# # # import graphviz
# # # from textblob import TextBlob
# # # from sklearn.feature_extraction.text import TfidfVectorizer

# # # # --------------------------
# # # # 1. Comprehensive NLTK Setup
# # # # --------------------------
# # # def setup_nltk():
# # #     try:
# # #         # Download all required NLTK data
# # #         nltk.download('punkt', quiet=True)
# # #         nltk.download('stopwords', quiet=True)
# # #         nltk.download('averaged_perceptron_tagger', quiet=True)
# # #         nltk.download('maxent_ne_chunker', quiet=True)
# # #         nltk.download('words', quiet=True)
        
# # #         # Verify all resources are available
# # #         nltk.data.find('tokenizers/punkt')
# # #         nltk.data.find('corpora/stopwords')
# # #         nltk.data.find('taggers/averaged_perceptron_tagger')
# # #         nltk.data.find('chunkers/maxent_ne_chunker')
# # #         return True
# # #     except Exception as e:
# # #         st.error(f"NLTK setup failed: {str(e)}")
# # #         st.error("""
# # #         Please manually download NLTK data:
# # #         1. Open Python interpreter
# # #         2. Run these commands:
# # #         import nltk
# # #         nltk.download('punkt')
# # #         nltk.download('stopwords')
# # #         nltk.download('averaged_perceptron_tagger')
# # #         nltk.download('maxent_ne_chunker')
# # #         nltk.download('words')
# # #         """)
# # #         return False

# # # if not setup_nltk():
# # #     st.stop()

# # # # --------------------------
# # # # 2. Enhanced Text Processing
# # # # --------------------------
# # # def extract_semantic_elements(text):
# # #     """Extract coherent hierarchical elements with fallbacks"""
# # #     sentences = nltk.sent_tokenize(text)
    
# # #     # Central Theme Selection with TF-IDF fallback
# # #     try:
# # #         vectorizer = TfidfVectorizer(max_features=5, stop_words='english')
# # #         tfidf = vectorizer.fit_transform(sentences)
# # #         central_idx = tfidf.sum(axis=1).argmax()
# # #         central_theme = ' '.join(sentences[central_idx].split()[:6])
# # #     except:
# # #         central_theme = sentences[0][:50] if sentences else "Main Topic"
    
# # #     # Extract elements with multiple approaches
# # #     blob = TextBlob(text)
# # #     pos_tags = nltk.pos_tag(nltk.word_tokenize(text))
    
# # #     # Level 1: Key Topics (noun phrases + important nouns)
# # #     topics = []
# # #     for np in blob.noun_phrases:
# # #         if 2 <= len(np.split()) <= 4:
# # #             topics.append(np)
    
# # #     # Fallback to important nouns if no phrases found
# # #     if not topics:
# # #         topics = [word for word, pos in pos_tags 
# # #                  if pos.startswith('NN') and word.lower() not in nltk.corpus.stopwords.words('english')][:5]
    
# # #     # Level 2: Key Actions (verbs + short phrases)
# # #     actions = []
# # #     for word, pos in pos_tags:
# # #         if pos.startswith('VB') and word.lower() not in nltk.corpus.stopwords.words('english'):
# # #             actions.append(word)
    
# # #     # Level 3: Specific Details (entities/numbers)
# # #     details = []
# # #     try:
# # #         for chunk in nltk.ne_chunk(pos_tags):
# # #             if hasattr(chunk, 'label'):
# # #                 details.append(' '.join(c[0] for c in chunk))
# # #     except:
# # #         details = [word for word, pos in pos_tags if pos in ['CD', 'NNP']][:10]
    
# # #     return {
# # #         'central': central_theme,
# # #         'topics': list(set(topics))[:5],
# # #         'actions': list(set(actions))[:8],
# # #         'details': list(set(details))[:10]
# # #     }

# # # # --------------------------
# # # # 3. NetworkX Graph with Fallbacks
# # # # --------------------------
# # # def create_networkx_mindmap(text):
# # #     elements = extract_semantic_elements(text)
# # #     G = nx.DiGraph()
    
# # #     # Central Node
# # #     G.add_node(elements['central'], level=0, color='#4E79A7', size=3500)
    
# # #     # Level 1: Topics
# # #     for i, topic in enumerate(elements['topics']):
# # #         G.add_node(topic, level=1, color='#F28E2B', size=2500)
# # #         G.add_edge(elements['central'], topic)
        
# # #         # Level 2: Actions
# # #         related_actions = elements['actions'][i*2:(i+1)*2]
# # #         for j, action in enumerate(related_actions):
# # #             action_node = f"{topic[:2]}_a{j}"
# # #             G.add_node(action_node, label=action, level=2, color='#59A14F', size=2000)
# # #             G.add_edge(topic, action_node)
            
# # #             # Level 3: Details
# # #             related_details = elements['details'][j*2:(j+1)*2]
# # #             for k, detail in enumerate(related_details):
# # #                 detail_node = f"{topic[:2]}_d{k}"
# # #                 G.add_node(detail_node, label=str(detail), level=3, color='#B07AA1', size=1500)
# # #                 G.add_edge(action_node, detail_node)
    
# # #     # Draw graph
# # #     plt.figure(figsize=(14, 10))
# # #     pos = nx.multipartite_layout(G, subset_key="level")
# # #     colors = [G.nodes[n]['color'] for n in G.nodes()]
# # #     sizes = [G.nodes[n]['size'] for n in G.nodes()]
# # #     labels = {n: G.nodes[n].get('label', n) for n in G.nodes()}
    
# # #     nx.draw(G, pos, labels=labels, node_color=colors, node_size=sizes,
# # #             font_size=10, edge_color='gray', width=1.5)
# # #     plt.title("Semantic Network Mindmap")
# # #     plt.axis('off')
# # #     plt.tight_layout()
# # #     return plt

# # # # --------------------------
# # # # 4. Graphviz Graph with Fallbacks
# # # # --------------------------
# # # def create_graphviz_mindmap(text):
# # #     elements = extract_semantic_elements(text)
# # #     dot = graphviz.Digraph(engine='dot')
    
# # #     # Graph style
# # #     dot.attr(rankdir='TB', size='12,12', ratio='auto')
    
# # #     # Central Node
# # #     dot.node('central', elements['central'], 
# # #              shape='doublecircle', style='filled',
# # #              fillcolor='#4E79A7', fontsize='14')
    
# # #     # Level 1: Topics
# # #     for i, topic in enumerate(elements['topics']):
# # #         dot.node(f'topic_{i}', topic, 
# # #                 shape='ellipse', style='filled',
# # #                 fillcolor='#F28E2B', fontsize='12')
# # #         dot.edge('central', f'topic_{i}')
        
# # #         # Level 2: Actions
# # #         for j, action in enumerate(elements['actions'][i*2:(i+1)*2]):
# # #             dot.node(f'action_{i}_{j}', action,
# # #                     shape='box', style='filled',
# # #                     fillcolor='#59A14F', fontsize='11')
# # #             dot.edge(f'topic_{i}', f'action_{i}_{j}')
            
# # #             # Level 3: Details
# # #             for k, detail in enumerate(elements['details'][j*2:(j+1)*2]):
# # #                 dot.node(f'detail_{i}_{j}_{k}', str(detail),
# # #                         shape='note', style='filled',
# # #                         fillcolor='#B07AA1', fontsize='10')
# # #                 dot.edge(f'action_{i}_{j}', f'detail_{i}_{j}_{k}')
    
# # #     return dot

# # # # --------------------------
# # # # 5. Streamlit App (Sequential Display)
# # # # --------------------------
# # # st.title("🎙️ Audio to Mindmap Converter")
# # # st.write("Generates hierarchical mindmaps from audio content")

# # # uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3"])
# # # if uploaded_file:
# # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
# # #         tmp_file.write(uploaded_file.read())
# # #         audio_path = tmp_file.name
    
# # #     # Transcription
# # #     with st.spinner("Processing audio content..."):
# # #         try:
# # #             model = whisper.load_model("base")
# # #             result = model.transcribe(audio_path)
# # #             text = result["text"]
# # #             st.subheader("Transcript")
# # #             st.text_area("", text, height=200)
# # #         except Exception as e:
# # #             st.error(f"Transcription failed: {str(e)}")
# # #             st.stop()
    
# # #     # NetworkX Mindmap
# # #     st.subheader("Interactive Network Visualization")
# # #     with st.spinner("Building semantic network..."):
# # #         try:
# # #             nx_plot = create_networkx_mindmap(text)
# # #             st.pyplot(nx_plot)
# # #         except Exception as e:
# # #             st.error(f"Network visualization error: {str(e)}")
    
# # #     # Graphviz Mindmap
# # #     st.subheader("Printable Hierarchy Diagram")
# # #     with st.spinner("Creating printable diagram..."):
# # #         try:
# # #             gv_graph = create_graphviz_mindmap(text)
# # #             st.graphviz_chart(gv_graph)
# # #         except Exception as e:
# # #             st.error(f"Diagram creation error: {str(e)}")
# # #             st.info("Ensure Graphviz is installed and in PATH")

# # # # --------------------------
# # # # 6. Installation Instructions
# # # # --------------------------
# # # st.subheader("⚙️ Complete Setup Guide")
# # # st.code("""
# # # # Core packages
# # # pip install streamlit openai-whisper torch nltk matplotlib networkx graphviz textblob scikit-learn

# # # # NLTK data (run in Python)
# # # import nltk
# # # nltk.download('punkt')
# # # nltk.download('stopwords')
# # # nltk.download('averaged_perceptron_tagger')
# # # nltk.download('maxent_ne_chunker')
# # # nltk.download('words')

# # # # TextBlob data
# # # python -m textblob.download_corpora

# # # # Graphviz (Windows):
# # # 1. Download from https://graphviz.org/download/
# # # 2. Install to default location
# # # 3. Add to PATH: C:\Program Files\Graphviz\bin\
# # # 4. Restart computer
# # # """)



# # import os
# # import re
# # import streamlit as st
# # import whisper
# # import torch
# # import numpy as np
# # import nltk
# # import tempfile
# # from collections import defaultdict
# # import matplotlib.pyplot as plt
# # import networkx as nx
# # from transformers import BertTokenizer, BertModel
# # from sklearn.metrics.pairwise import cosine_similarity
# # from sentence_transformers import SentenceTransformer
# # import torch.optim as optim

# # # --------------------------
# # # 1. NLTK & Model Setup
# # # --------------------------
# # def setup_environment():
# #     try:
# #         nltk.download('punkt', quiet=True)
# #         nltk.download('stopwords', quiet=True)
# #         return True
# #     except Exception as e:
# #         st.error(f"Setup failed: {str(e)}")
# #         return False

# # if not setup_environment():
# #     st.stop()

# # # Load BERT model
# # @st.cache_resource
# # def load_bert():
# #     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# #     model = BertModel.from_pretrained('bert-base-uncased')
# #     return tokenizer, model

# # tokenizer, bert_model = load_bert()

# # # --------------------------
# # # 2. Advanced Relationship Extraction
# # # --------------------------
# # def get_bert_embeddings(text, tokenizer, model):
# #     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
# #     with torch.no_grad():
# #         outputs = model(**inputs)
# #     return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# # class RelationshipExtractor:
# #     def __init__(self, bert_model, tokenizer):
# #         self.bert_model = bert_model
# #         self.tokenizer = tokenizer
# #         self.reward_weights = np.ones(768)  # Initialize with uniform weights
    
# #     def extract_relationships(self, sentences, threshold=0.85):
# #         # Get BERT embeddings
# #         embeddings = np.array([get_bert_embeddings(s, self.tokenizer, self.bert_model) for s in sentences])
        
# #         # Calculate similarity matrix
# #         sim_matrix = cosine_similarity(embeddings)
        
# #         # Apply IRL to adjust weights
# #         self._train_irl(embeddings)
        
# #         # Get final relationships with learned weights
# #         relationships = []
# #         for i in range(len(sentences)):
# #             for j in range(i+1, len(sentences)):
# #                 weighted_sim = np.dot(self.reward_weights, embeddings[i] * embeddings[j])
# #                 if weighted_sim > threshold:
# #                     relationships.append((sentences[i], sentences[j], weighted_sim))
        
# #         return relationships
    
# #     def _train_irl(self, embeddings, n_iterations=50, lr=0.01):
# #         optimizer = optim.Adam([torch.tensor(self.reward_weights, requires_grad=True)], lr=lr)
        
# #         for _ in range(n_iterations):
# #             # Calculate reward using current weights
# #             rewards = np.array([np.dot(self.reward_weights, emb) for emb in embeddings])
            
# #             # Normalize rewards
# #             norm_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
# #             # Calculate loss (inverse reinforcement learning)
# #             loss = -torch.mean(torch.tensor(norm_rewards))
            
# #             # Update weights
# #             optimizer.zero_grad()
# #             loss.backward()
# #             optimizer.step()
            
# #             # Update numpy array
# #             with torch.no_grad():
# #                 self.reward_weights = torch.clamp(optimizer.param_groups[0]['params'][0], 0, 1).numpy()

# # # --------------------------
# # # 3. Hierarchical Mindmap Construction
# # # --------------------------
# # def build_mindmap(text):
# #     # Process text
# #     sentences = [s for s in nltk.sent_tokenize(text) if len(s.split()) > 4]  # Only meaningful sentences
    
# #     if len(sentences) < 3:
# #         return None
    
# #     # Extract relationships
# #     extractor = RelationshipExtractor(bert_model, tokenizer)
# #     relationships = extractor.extract_relationships(sentences)
    
# #     # Build graph
# #     G = nx.Graph()
    
# #     # Add nodes with importance scores
# #     importance = defaultdict(int)
# #     for rel in relationships:
# #         importance[rel[0]] += rel[2]
# #         importance[rel[1]] += rel[2]
    
# #     # Central node (most important sentence)
# #     central_node = max(importance.items(), key=lambda x: x[1])[0]
# #     G.add_node(central_node, size=3000, color='#4E79A7')
    
# #     # Add relationships
# #     for rel in relationships:
# #         if rel[0] == central_node or rel[1] == central_node:
# #             other_node = rel[1] if rel[0] == central_node else rel[0]
# #             G.add_node(other_node, size=2000 + importance[other_node]*100, color='#F28E2B')
# #             G.add_edge(central_node, other_node, weight=rel[2])
            
# #             # Second level connections
# #             for rel2 in relationships:
# #                 if (rel2[0] == other_node and rel2[1] != central_node) or \
# #                    (rel2[1] == other_node and rel2[0] != central_node):
# #                     second_node = rel2[1] if rel2[0] == other_node else rel2[0]
# #                     G.add_node(second_node, size=1500 + importance[second_node]*80, color='#59A14F')
# #                     G.add_edge(other_node, second_node, weight=rel2[2])
    
# #     return G

# # # --------------------------
# # # 4. Visualization
# # # --------------------------
# # def visualize_mindmap(G):
# #     if not G:
# #         return None
    
# #     plt.figure(figsize=(16, 12))
# #     pos = nx.spring_layout(G, k=0.5, seed=42)
    
# #     # Draw nodes
# #     node_colors = [G.nodes[n]['color'] for n in G.nodes()]
# #     node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
    
# #     nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    
# #     # Draw edges with weights
# #     edge_weights = [G.edges[e]['weight']*3 for e in G.edges()]
# #     nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color='#888888', alpha=0.6)
    
# #     # Draw labels with word wrapping
# #     labels = {}
# #     for node in G.nodes():
# #         words = node.split()
# #         wrapped_label = '\n'.join([' '.join(words[i:i+8]) for i in range(0, len(words), 8)])
# #         labels[node] = wrapped_label
    
# #     nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
    
# #     plt.title("BERT+IRL Semantic Mindmap", pad=20)
# #     plt.axis('off')
# #     return plt

# # # --------------------------
# # # 5. Streamlit App
# # # --------------------------
# # st.title("🧠 BERT+IRL Audio Mindmap Generator")

# # uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3"])
# # if uploaded_file:
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
# #         tmp_file.write(uploaded_file.read())
# #         audio_path = tmp_file.name
    
# #     # Transcription
# #     with st.spinner("Processing audio content..."):
# #         try:
# #             model = whisper.load_model("base")
# #             result = model.transcribe(audio_path)
# #             text = result["text"]
# #             st.subheader("Transcript")
# #             st.text_area("", text, height=200)
# #         except Exception as e:
# #             st.error(f"Transcription failed: {str(e)}")
# #             st.stop()
    
# #     # Mindmap Generation
# #     with st.spinner("Building semantic relationships (BERT+IRL)..."):
# #         try:
# #             G = build_mindmap(text)
# #             if G:
# #                 plt = visualize_mindmap(G)
# #                 st.pyplot(plt)
# #             else:
# #                 st.warning("Not enough meaningful sentences to build mindmap")
# #         except Exception as e:
# #             st.error(f"Mindmap creation error: {str(e)}")

# # # --------------------------
# # # 6. Installation
# # # --------------------------
# # st.subheader("⚙️ Setup Instructions")
# # st.code("""
# # # Core packages
# # pip install streamlit openai-whisper torch numpy nltk matplotlib networkx scikit-learn sentence-transformers transformers

# # # NLTK data
# # python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
# # """)

# import os
# import re
# import streamlit as st
# import whisper
# import torch
# import numpy as np
# import nltk
# import tempfile
# from collections import defaultdict
# import matplotlib.pyplot as plt
# import networkx as nx
# import graphviz
# from transformers import BertTokenizer, BertModel
# import torch.optim as optim
# from sentence_transformers import SentenceTransformer

# # --------------------------
# # 1. Environment Setup
# # --------------------------
# def setup_environment():
#     try:
#         nltk.download('punkt', quiet=True)
#         nltk.download('stopwords', quiet=True)
#         return True
#     except Exception as e:
#         st.error(f"Setup failed: {str(e)}")
#         return False

# if not setup_environment():
#     st.stop()

# # --------------------------
# # 2. Enhanced BERT+IRL Model
# # --------------------------
# @st.cache_resource
# def load_models():
#     # Load both BERT and SentenceTransformer as fallback
#     bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     bert_model = BertModel.from_pretrained('bert-base-uncased')
#     st_model = SentenceTransformer('all-MiniLM-L6-v2')
#     return bert_tokenizer, bert_model, st_model

# class RelationshipExtractor:
#     def __init__(self, bert_model, tokenizer, st_model):
#         self.bert_model = bert_model
#         self.tokenizer = tokenizer
#         self.st_model = st_model
#         self.reward_weights = torch.nn.Parameter(torch.ones(768, requires_grad=True))
        
#     def extract_relationships(self, sentences, threshold=0.75):
#         try:
#             # Try BERT first
#             embeddings = self._get_bert_embeddings(sentences)
#         except:
#             # Fallback to SentenceTransformer
#             embeddings = torch.tensor(self.st_model.encode(sentences))
        
#         self._train_irl(embeddings)
#         return self._get_relationships(sentences, embeddings, threshold)
    
#     def _get_bert_embeddings(self, sentences):
#         inputs = self.tokenizer(sentences, return_tensors='pt', 
#                               padding=True, truncation=True, max_length=512)
#         with torch.no_grad():
#             outputs = self.bert_model(**inputs)
#         return outputs.last_hidden_state.mean(dim=1)
    
#     def _train_irl(self, embeddings, n_iterations=50, lr=0.01):
#         optimizer = optim.Adam([self.reward_weights], lr=lr)
#         for _ in range(n_iterations):
#             weights = torch.softmax(self.reward_weights, dim=0)
#             rewards = torch.matmul(embeddings, weights.unsqueeze(1)).squeeze()
#             norm_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
#             loss = -torch.mean(norm_rewards)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
    
#     def _get_relationships(self, sentences, embeddings, threshold):
#         relationships = []
#         with torch.no_grad():
#             weights = torch.softmax(self.reward_weights, dim=0)
#             for i in range(len(sentences)):
#                 for j in range(i+1, len(sentences)):
#                     sim = torch.dot(weights, embeddings[i] * embeddings[j]).item()
#                     if sim > threshold:
#                         relationships.append((sentences[i], sentences[j], sim))
#         return relationships

# # --------------------------
# # 3. Dual Mindmap Generation
# # --------------------------
# def build_mindmaps(text):
#     sentences = [s.strip() for s in nltk.sent_tokenize(text) if 5 <= len(s.split()) <= 30]
#     if len(sentences) < 3:
#         return None, None
    
#     bert_tokenizer, bert_model, st_model = load_models()
#     extractor = RelationshipExtractor(bert_model, bert_tokenizer, st_model)
#     relationships = extractor.extract_relationships(sentences)
    
#     if not relationships:
#         return None, None
    
#     # Create NetworkX graph
#     nx_graph = nx.Graph()
#     importance = defaultdict(float)
#     for rel in relationships:
#         importance[rel[0]] += rel[2]
#         importance[rel[1]] += rel[2]
    
#     central_node = max(importance.items(), key=lambda x: x[1])[0]
#     nx_graph.add_node(central_node, level=0, size=3000, color='#4E79A7')
    
#     # Add relationships with hierarchy
#     for rel in relationships:
#         if rel[0] == central_node or rel[1] == central_node:
#             other = rel[1] if rel[0] == central_node else rel[0]
#             nx_graph.add_node(other, level=1, size=2000 + importance[other]*100, color='#F28E2B')
#             nx_graph.add_edge(central_node, other, weight=rel[2])
            
#             # Second level connections
#             for rel2 in [r for r in relationships if r != rel]:
#                 if (rel2[0] == other and rel2[1] != central_node) or (rel2[1] == other and rel2[0] != central_node):
#                     second_node = rel2[1] if rel2[0] == other else rel2[0]
#                     nx_graph.add_node(second_node, level=2, size=1500 + importance[second_node]*80, color='#59A14F')
#                     nx_graph.add_edge(other, second_node, weight=rel2[2])
    
#     # Create Graphviz graph
#     gv_graph = graphviz.Digraph(engine='dot')
#     gv_graph.attr(rankdir='TB', size='15,15', ratio='auto')
#     gv_graph.node('central', central_node, shape='doublecircle', style='filled', fillcolor='#4E79A7', fontsize='14')
    
#     for node in nx_graph.nodes():
#         if node != central_node:
#             level = nx_graph.nodes[node]['level']
#             if level == 1:
#                 gv_graph.node(node[:8], node, shape='ellipse', style='filled', fillcolor='#F28E2B')
#                 gv_graph.edge('central', node[:8])
#             elif level == 2:
#                 gv_graph.node(node[:8], node, shape='box', style='filled', fillcolor='#59A14F')
#                 # Find parent
#                 for neighbor in nx_graph.neighbors(node):
#                     if nx_graph.nodes[neighbor]['level'] == 1:
#                         gv_graph.edge(neighbor[:8], node[:8])
    
#     return nx_graph, gv_graph

# # --------------------------
# # 4. Visualization Functions
# # --------------------------
# def draw_networkx_graph(G):
#     plt.figure(figsize=(16, 12))
#     pos = nx.multipartite_layout(G, subset_key="level")
#     nx.draw(G, pos, 
#            with_labels=True,
#            node_size=[G.nodes[n]['size'] for n in G.nodes()],
#            node_color=[G.nodes[n]['color'] for n in G.nodes()],
#            edge_color='#888888',
#            width=[G.edges[e]['weight'] for e in G.edges()],
#            font_size=10)
#     plt.title("Semantic Relationship Network", pad=20)
#     plt.axis('off')
#     return plt

# # --------------------------
# # 5. Streamlit App
# # --------------------------
# st.title("🧠 Advanced Audio Mindmap Generator")

# uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3"])
# if uploaded_file:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
#         tmp_file.write(uploaded_file.read())
#         audio_path = tmp_file.name
    
#     # Transcription
#     with st.spinner("Processing audio content..."):
#         try:
#             model = whisper.load_model("base")
#             result = model.transcribe(audio_path)
#             text = result["text"]
#             st.subheader("Transcript")
#             st.text_area("", text, height=200)
#         except Exception as e:
#             st.error(f"Transcription failed: {str(e)}")
#             st.stop()
    
#     # Mindmap Generation
#     with st.spinner("Building semantic relationships..."):
#         try:
#             nx_graph, gv_graph = build_mindmaps(text)
            
#             if nx_graph and gv_graph:
#                 st.subheader("Interactive Network Visualization")
#                 plt = draw_networkx_graph(nx_graph)
#                 st.pyplot(plt)
                
#                 st.subheader("Printable Hierarchy Diagram")
#                 st.graphviz_chart(gv_graph)
#             else:
#                 st.warning("Insufficient meaningful relationships found. Try longer or more detailed audio.")
#         except Exception as e:
#             st.error(f"Mindmap generation error: {str(e)}")

# # --------------------------
# # 6. Installation
# # --------------------------
# st.subheader("⚙️ Complete Setup Guide")
# st.code("""
# pip install streamlit openai-whisper torch numpy nltk matplotlib networkx graphviz transformers sentence-transformers
# python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# # For Graphviz:
# # Windows: Download from graphviz.org and add to PATH
# # Mac: brew install graphviz
# # Linux: sudo apt install graphviz
# """)



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

# Load English language model for spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.error("Please install spaCy English model: python -m spacy download en_core_web_sm")
    st.stop()

# --------------------------
# 1. Text Processing
# --------------------------
def process_text(text):
    """Extract sentences and BERT embeddings"""
    try:
        nltk.download("punkt", quiet=True)
        sentences = nltk.sent_tokenize(text)
    except:
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Remove very short/long sentences
    sentences = [s.strip() for s in sentences if 5 <= len(s.split()) <= 30]
    
    if len(sentences) < 2:
        return None, None
    
    # Get BERT embeddings
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    
    def get_bert_embeddings(sentence):
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    embeddings = [get_bert_embeddings(sent) for sent in sentences]
    return sentences, embeddings

# --------------------------
# 2. Relationship Extraction
# --------------------------
def extract_relationships(sentences, embeddings):
    """Find semantic relationships between sentences"""
    relationships = []
    threshold = 0.65  # Start with lower threshold
    
    # Try with decreasing thresholds if few relationships found
    for _ in range(3):
        relationships = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                if sim > threshold:
                    relationships.append((sentences[i], sentences[j], sim))
        
        if len(relationships) >= max(3, len(sentences)//2):  # Adaptive minimum
            break
        threshold -= 0.05
    
    return relationships if relationships else None

def extract_key_elements(text):
    """Extract keywords and entities using spaCy"""
    doc = nlp(text)
    
    # Extract key phrases (noun chunks)
    key_phrases = list(set([chunk.text for chunk in doc.noun_chunks if 2 <= len(chunk.text.split()) <= 4]))
    
    # Extract named entities
    entities = {
        'people': list(set([ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG"]])),
        'numbers': list(set([ent.text for ent in doc.ents if ent.label_ in ["MONEY", "PERCENT", "DATE", "QUANTITY"]]))
    }
    
    return key_phrases, entities

# --------------------------
# 3. Mindmap Generation
# --------------------------
def create_mindmap(text):
    # Process text and get relationships
    sentences, embeddings = process_text(text)
    if not sentences:
        return None, None
    
    relationships = extract_relationships(sentences, embeddings)
    if not relationships:
        return None, None
    
    key_phrases, entities = extract_key_elements(text)
    
    # Create NetworkX graph
    nx_graph = nx.Graph()
    
    # Central node (most connected sentence)
    connection_counts = defaultdict(int)
    for rel in relationships:
        connection_counts[rel[0]] += 1
        connection_counts[rel[1]] += 1
    central_node = max(connection_counts.items(), key=lambda x: x[1])[0]
    nx_graph.add_node(central_node, level=0, type='central', size=3000)
    
    # Add relationships
    for rel in relationships:
        if rel[0] == central_node or rel[1] == central_node:
            other_node = rel[1] if rel[0] == central_node else rel[0]
            nx_graph.add_node(other_node, level=1, type='sentence', size=2000)
            nx_graph.add_edge(central_node, other_node, weight=rel[2])
    
    # Add key phrases and entities
    for phrase in key_phrases[:10]:  # Top 10 phrases
        nx_graph.add_node(phrase, level=2, type='phrase', size=1500)
        # Connect to most relevant sentence
        best_sent = None
        best_sim = 0
        for node in nx_graph.nodes():
            if nx_graph.nodes[node]['type'] == 'sentence' and phrase.lower() in node.lower():
                nx_graph.add_edge(node, phrase, weight=0.8)
                best_sent = None
                break
        if best_sent:
            nx_graph.add_edge(best_sent, phrase, weight=0.7)
    
    # Create Graphviz graph
    gv_graph = graphviz.Digraph(engine='dot')
    gv_graph.attr(rankdir='TB', size='12,12')
    
    # Add nodes with styling
    for node in nx_graph.nodes():
        node_type = nx_graph.nodes[node]['type']
        if node_type == 'central':
            gv_graph.node(str(hash(node)), node, shape='doublecircle', style='filled', fillcolor='#4E79A7')
        elif node_type == 'sentence':
            gv_graph.node(str(hash(node)), node, shape='ellipse', style='filled', fillcolor='#F28E2B')
        else:
            gv_graph.node(str(hash(node)), node, shape='box', style='filled', fillcolor='#59A14F')
    
    # Add edges
    for edge in nx_graph.edges():
        gv_graph.edge(str(hash(edge[0])), str(hash(edge[1])))
    
    return nx_graph, gv_graph

# --------------------------
# 4. Visualization
# --------------------------
def draw_networkx_graph(G):
    plt.figure(figsize=(16, 12))
    
    # Prepare node colors and sizes
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

# --------------------------
# 5. Streamlit App
# --------------------------
st.title("🧠 Audio to Mindmap Converter")

uploaded_file = st.file_uploader("Upload audio file (WAV/MP3)", type=["wav", "mp3"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        audio_path = tmp_file.name
    
    # Transcription
    with st.spinner("Transcribing audio..."):
        try:
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            text = result["text"]
            st.subheader("Transcript")
            st.text_area("", text, height=200)
        except Exception as e:
            st.error(f"Transcription failed: {str(e)}")
            st.stop()
    
    # Mindmap Generation
    with st.spinner("Building semantic mindmap..."):
        nx_graph, gv_graph = create_mindmap(text)
        
        if nx_graph and gv_graph:
            st.subheader("Interactive Network View")
            plt = draw_networkx_graph(nx_graph)
            st.pyplot(plt)
            
            st.subheader("Printable Hierarchy View")
            st.graphviz_chart(gv_graph)
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

# For Graphviz:
# Windows: Download from graphviz.org and add to PATH
# Mac: brew install graphviz
# Linux: sudo apt install graphviz
""")