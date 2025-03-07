import os
import io
import tempfile
import numpy as np
import librosa
import warnings
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
import streamlit as st
import imageio_ffmpeg
import datetime
import soundfile as sf

# Import transformers pipeline for Q&A and summarization
from transformers import pipeline

warnings.filterwarnings('ignore')
nltk.download('punkt', quiet=True)

# --- Debug prints (using print) before any st commands ---
print("Using ffmpeg at:", imageio_ffmpeg.get_ffmpeg_exe())
try:
    test_audio = AudioSegment.silent(duration=1000)  # 1 sec silence
    test_audio.export("test.wav", format="wav")
    print("Test conversion succeeded. 'test.wav' created.")
except Exception as e:
    print("Test conversion failed:", e)

# --- Set page configuration (must be the very first st command) ---
st.set_page_config(page_title="Audio Reader & Q&A 🤖", layout="wide", initial_sidebar_state="expanded")

# Instantiate the dedicated QA model outside the class (to avoid re-loading)
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# -----------------------------------------------
# AudioReader class – processes audio and provides Q&A over its transcript
# -----------------------------------------------
class AudioReader:
    def __init__(self, max_chunks=70):
        self.audio_file = None         # Path of the temporary file
        self.audio_data = None
        self.sample_rate = None
        self.duration = None
        self.transcript = ""
        self.recognizer = sr.Recognizer()
        self.max_chunks = max_chunks
        self.wav_bytes = None          # In-memory WAV data

    def load_audio(self, file_obj, file_ext):
        """
        Save the uploaded Streamlit file to a temporary location and load it using librosa.
        """
        try:
            temp_path = os.path.join(tempfile.gettempdir(), file_obj.name)
            with open(temp_path, "wb") as f:
                f.write(file_obj.getbuffer())
            self.audio_file = temp_path
            self.audio_data, self.sample_rate = librosa.load(self.audio_file, sr=None)
            self.duration = librosa.get_duration(y=self.audio_data, sr=self.sample_rate)
            st.write(f"🎧 Audio loaded: {self.duration:.2f} sec at {self.sample_rate} Hz")
            return True
        except Exception as e:
            st.write(f"❌ Error loading audio: {e}")
            return False

    def transcribe_audio_chunks(self, min_silence_len=500, silence_thresh=-40, keep_silence=250):
        """
        Convert non-WAV files to WAV if necessary, split the audio into chunks and transcribe each chunk.
        """
        if not self.audio_file.lower().endswith('.wav'):
            try:
                st.write("🔄 Converting audio to WAV...")
                audio = AudioSegment.from_file(self.audio_file)  # Let it infer format
                wav_io = io.BytesIO()
                audio.export(wav_io, format="wav")
                wav_io.seek(0)
                self.wav_bytes = wav_io.getvalue()
                st.write("✅ Audio converted to WAV (in-memory).")
            except Exception as e:
                st.write(f"❌ Error converting audio: {e}")
                return False
        else:
            try:
                with open(self.audio_file, "rb") as f:
                    self.wav_bytes = f.read()
            except Exception as e:
                st.write(f"❌ Error reading WAV file: {e}")
                return False

        # Use in-memory WAV bytes for transcription.
        wav_io = io.BytesIO(self.wav_bytes)
        try:
            sound = AudioSegment.from_file(wav_io, format="wav")
        except Exception as e:
            st.write(f"❌ Error loading WAV audio: {e}")
            return False

        st.write("✂️ Splitting audio into chunks...")
        chunks = split_on_silence(sound, min_silence_len=min_silence_len,
                                    silence_thresh=silence_thresh, keep_silence=keep_silence)
        if not chunks:
            st.write("❌ No chunks created; check your silence settings.")
            return False

        whole_transcript = ""
        num_chunks = min(len(chunks), self.max_chunks)
        st.write(f"⌛ Processing {num_chunks} out of {len(chunks)} chunks (max_chunks={self.max_chunks}).")
        for i in range(num_chunks):
            chunk = chunks[i]
            chunk_io = io.BytesIO()
            chunk.export(chunk_io, format="wav")
            chunk_io.seek(0)
            with sr.AudioFile(chunk_io) as source:
                audio_chunk = self.recognizer.record(source)
                try:
                    chunk_text = self.recognizer.recognize_google(audio_chunk)
                    st.write(f"💬 Chunk {i+1} transcription: {chunk_text[:50]}...")
                    whole_transcript += chunk_text + " "
                except Exception as e:
                    st.write(f"❌ Error transcribing chunk {i+1}: {e}")
        self.transcript = whole_transcript.strip()
        st.write(f"📝 Combined transcript length: {len(self.transcript)} characters")
        # Clean up temporary file
        if os.path.exists(self.audio_file):
            os.remove(self.audio_file)
        return True

    def analyze_content(self):
        """
        Analyze the transcript to extract basic information about its content.
        Extracts key topics using noun phrases.
        """
        if not self.transcript:
            st.write("❌ No transcript available for analysis.")
            return None
        blob = TextBlob(self.transcript)
        noun_phrases = blob.noun_phrases
        # Count frequency of each noun phrase
        freq = {}
        for phrase in noun_phrases:
            freq[phrase] = freq.get(phrase, 0) + 1
        # Sort noun phrases by frequency
        sorted_phrases = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        top_phrases = [phrase for phrase, count in sorted_phrases[:5]]
        st.write("✅ Content analysis complete.")
        return top_phrases

    def summarize_transcript(self):
        """
        Generate a summary of the transcript using a summarization model.
        To keep latency low, if the transcript is too long, only a portion is summarized.
        """
        if not self.transcript:
            return "No transcript available."
        try:
            # Instantiate a lightweight summarizer
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            # Limit input length to avoid overloading the model (approx. 1000 characters)
            text = self.transcript
            if len(text) > 1000:
                text = text[:1000]
            summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            return f"Error in summarization: {e}"

    def answer_question(self, question):
        """
        Answer a user's question using a two-step approach:
        1. Select the two most relevant sentences from the transcript.
        2. Feed this shorter context to the QA model.
        """
        if not self.transcript:
            return "No transcript available to search for an answer."
        
        sentences = sent_tokenize(self.transcript)
        q_tokens = set(word_tokenize(question.lower()))
        # Compute word-overlap for each sentence
        scored_sentences = []
        for sentence in sentences:
            s_tokens = set(word_tokenize(sentence.lower()))
            overlap = len(q_tokens.intersection(s_tokens))
            scored_sentences.append((overlap, sentence))
        scored_sentences = sorted(scored_sentences, key=lambda x: x[0], reverse=True)
        # Select top 2 sentences with non-zero overlap
        selected = [s for overlap, s in scored_sentences if overlap > 0][:2]
        context = " ".join(selected)
        # Fallback if no overlap found
        if not context:
            context = self.transcript[:1000]
        try:
            result = qa_model(question=question, context=context)
            answer = result.get("answer", "I couldn't find an answer in the audio content.")
            score = result.get("score", 0)
            return f"{answer}"
        except Exception as e:
            return f"An error occurred during Q&A: {e}"

# -------------------------------------------
# Streamlit Interface
# -------------------------------------------
st.markdown("""
    <style>
      .reportview-container, .main .block-container {
          background-color: #000;
          color: #e0e0e0;
      }
      .stSidebar {
          background-color: #1e1e1e;
      }
      .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar p {
          color: #f0f0f0;
      }
      .chat-message {
          background-color: #121212;
          border-radius: 8px;
          padding: 10px 15px;
          margin: 5px 0;
          color: #e0e0e0;
      }
      .chat-user {
          font-weight: bold;
          color: #ffcc00;
          margin-bottom: 5px;
      }
      div.stButton > button {
          box-shadow: 0 4px 6px rgba(0,0,0,0.5) !important;
      }
      .stSidebar .stButton button {
          background-color: #d3d3d3 !important;
          color: black !important;
      }
    </style>
""", unsafe_allow_html=True)

# Sidebar: History and Controls
if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.header("🗂️ History & Controls")
if st.session_state.history:
    for entry in st.session_state.history:
        st.sidebar.markdown(entry, unsafe_allow_html=True)
else:
    st.sidebar.info("No uploads yet.")

if st.sidebar.button("🔄 Reset History"):
    st.session_state.history = []
    st.sidebar.success("History reset!")

if st.sidebar.button("🗂️ Create New Folder"):
    folder_name = f"reports_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(folder_name, exist_ok=True)
    st.sidebar.success(f"New folder created: {folder_name}")

st.sidebar.markdown("### More Info")
st.sidebar.info("Time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
st.sidebar.markdown("Sources: Librosa, SpeechRecognition, TextBlob, NLTK, Transformers")

# Main Title and Chat Container
st.title("Audio Reader & Q&A 🤖")
with st.container():
    st.markdown("""
        <div class="chat-message">
            <div class="chat-user">Bot 🤖:</div>
            Welcome! Please upload an audio file to extract its content and chat about it.
        </div>
    """, unsafe_allow_html=True)

# File uploader – pass the uploaded file directly
uploaded_file = st.file_uploader("📤 Upload your audio file", type=["wav", "mp3", "ogg"])

# Additional buttons for resetting analysis and clearing chat
col1, col2 = st.columns(2)
with col1:
    reset_button = st.button("❌ Reset Analysis")
with col2:
    clear_chat = st.button("🧹 Clear Chat")

if uploaded_file is not None:
    st.markdown(f"""
        <div class="chat-message">
            <div class="chat-user">User 🧑:</div>
            Uploaded file: <strong>{uploaded_file.name}</strong>
        </div>
    """, unsafe_allow_html=True)

    with st.spinner("Processing your audio file... ⏳"):
        reader = AudioReader()
        if reader.load_audio(uploaded_file, os.path.splitext(uploaded_file.name)[1][1:].lower()):
            if reader.transcribe_audio_chunks():
                topics = reader.analyze_content()
                summary = reader.summarize_transcript()
                st.markdown("### Audio Details")
                st.write(f"**Duration:** {reader.duration:.2f} sec")
                st.write(f"**Transcript length:** {len(reader.transcript)} characters")
                st.write("**Summary:** " + summary)
                if topics:
                    st.write("**Key Topics:** " + ", ".join(topics))
                else:
                    st.write("No key topics could be extracted.")
                # Save transcript in session state for Q&A
                st.session_state["transcript"] = reader.transcript
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                history_entry = f"""
                    <div class="chat-message">
                        [{timestamp}] File: <strong>{uploaded_file.name}</strong>
                    </div>
                """
                st.session_state.history.append(history_entry)
            else:
                st.error("❌ Error transcribing audio.")
        else:
            st.error("❌ Error loading audio.")

# Chatbox for Q&A on audio content
st.markdown("### Ask a Question about the Audio")
question = st.text_input("Type your question here:")
if st.button("Submit Question") and question:
    if "transcript" in st.session_state:
        reader = AudioReader()
        # Set the transcript from session state
        reader.transcript = st.session_state["transcript"]
        answer = reader.answer_question(question)
        st.markdown(f"""
            <div class="chat-message">
                <div class="chat-user">Answer Bot 🤖:</div>
                {answer}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.error("No transcript available. Please upload an audio file first.")

if reset_button:
    st.session_state.history = []
    if "transcript" in st.session_state:
        del st.session_state["transcript"]
    st.write("Analysis has been reset. Please upload a new file.")
if clear_chat:
    st.session_state.history = []
    st.write("Chat cleared.")