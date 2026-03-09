import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from music21 import stream, note

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="AI Text-to-Music", layout="centered")

st.markdown("""
<style>
body {
    background-color: #0b0f1a;
}
.main {
    background: linear-gradient(180deg, #0b0f1a 0%, #090d18 100%);
}
h1 {
    text-align: center;
    color: #a78bfa;
    font-weight: 700;
}
.subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 30px;
}
.stTextArea textarea {
    background-color: #111827;
    color: white;
    border-radius: 12px;
}
.stButton>button {
    background: linear-gradient(90deg,#7c3aed,#4f46e5);
    color: white;
    border-radius: 12px;
    height: 50px;
    width: 100%;
    font-size: 16px;
}
.result-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 14px;
    margin-top: 25px;
}
.genre-badge {
    background: #6d28d9;
    padding: 6px 14px;
    border-radius: 20px;
    color: white;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load Assets
# -------------------------------------------------
@st.cache_resource
def load_assets():
    text_model = load_model("models/text_lstm_genre_model.keras")
    midi_model = load_model("models/midi_lstm_model.keras")

    with open("models/text_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("models/genre_encoder.pkl", "rb") as f:
        genre_encoder = pickle.load(f)

    with open("models/int_to_note.pkl", "rb") as f:
        int_to_note = pickle.load(f)

    X_midi = np.load("models/X_midi.npy")

    return text_model, midi_model, tokenizer, genre_encoder, int_to_note, X_midi


text_model, midi_model, tokenizer, genre_encoder, int_to_note, X_midi = load_assets()

# -------------------------------------------------
# Genre Prediction
# -------------------------------------------------
def predict_genre(text):
    text = text.lower()

    # Keyword-based detection for reliable genre prediction
    if "rock" in text:
        return "Rock"
    elif "jazz" in text:
        return "Jazz"
    elif "classical" in text:
        return "Classical"
    elif "edm" in text or "electronic" in text:
        return "EDM"
    elif "hip hop" in text or "hip-hop" in text or "rap" in text:
        return "Hip-Hop"
    elif "ambient" in text or "relaxing" in text:
        return "Ambient"

    # Fallback to trained model if no keyword found
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=50)
    pred = text_model.predict(padded, verbose=0)

    return genre_encoder.inverse_transform([np.argmax(pred)])[0]

# -------------------------------------------------
# Seed Selection
# -------------------------------------------------
def get_seed():
    idx = np.random.randint(0, len(X_midi))
    return X_midi[idx].reshape(1, X_midi.shape[1], 1)

# -------------------------------------------------
# Music Generation
# -------------------------------------------------
def generate_music(seed, length=300):
    pattern = seed.copy()
    generated = []

    for _ in range(length):
        pred = midi_model.predict(pattern, verbose=0)[0]
        index = np.random.choice(len(pred), p=pred)
        generated.append(int_to_note[index])
        pattern = np.append(pattern[:,1:,:], [[[index]]], axis=1)

    return generated

# -------------------------------------------------
# Save MIDI
# -------------------------------------------------
def save_midi(notes, filename):
    midi_stream = stream.Stream()

    for n in notes:
        try:
            midi_stream.append(note.Note(n))
        except:
            pass

    midi_stream.write("midi", fp=filename)

# -------------------------------------------------
# UI Layout
# -------------------------------------------------
st.markdown("<h1>🎵 AI Text-to-Music Generator</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Describe music → AI composes it.</p>", unsafe_allow_html=True)

user_input = st.text_area("Describe your music")

if st.button("✨ Generate Music"):

    if user_input.strip() == "":
        st.warning("Please enter a description.")
        st.stop()

    with st.spinner("Composing your music..."):
        genre = predict_genre(user_input)
        seed = get_seed()
        notes = generate_music(seed)

        filename = f"generated_{genre}.mid"
        save_midi(notes, filename)

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    st.markdown("**Predicted Genre**")
    st.markdown(f'<span class="genre-badge">{genre}</span>', unsafe_allow_html=True)

    # Download only (no audio player)
    with open(filename, "rb") as file:
        st.download_button(
            "⬇ Download Generated MIDI",
            data=file,
            file_name=filename,
            mime="audio/midi"
        )

    st.markdown("</div>", unsafe_allow_html=True)