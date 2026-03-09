# 🎵 AI Text-to-Music Generation using LSTM

## 📌 Overview
This project implements an **AI-powered Text-to-Music Generation system** that converts natural language descriptions into musical compositions. The system first predicts the **music genre from a textual prompt** and then generates a **MIDI music sequence using a Long Short-Term Memory (LSTM) neural network** trained on symbolic music data.

The generated output is a **playable MIDI file** that can be downloaded or used in digital audio workstations such as **FL Studio, Ableton, MuseScore, or GarageBand**.

---

# 🚀 Features

- 🎼 Generate music from **natural language descriptions**
- 🤖 **Deep learning-based genre classification**
- 🎹 **LSTM-based sequential music generation**
- 📂 Automatic **MIDI file generation**
- 🌐 **Streamlit web interface**
- 🎧 Downloadable **MIDI compositions**

---

# 🧠 System Architecture

The system consists of two main stages:

1. **Text-to-Genre Prediction**
2. **Genre-based Music Generation using LSTM**

```
User Text Input
       ↓
Text Preprocessing
       ↓
Genre Prediction Model
       ↓
Predicted Genre
       ↓
Seed Selection (MIDI Dataset)
       ↓
LSTM Music Generation
       ↓
Note Sequence Mapping
       ↓
MIDI File Creation
       ↓
Generated MIDI File
```

---

# 📂 Project Structure

```
AI-Text-to-Music-Generation
│
├── models
│   ├── text_lstm_genre_model.keras
│   ├── midi_lstm_model.keras
│   ├── text_tokenizer.pkl
│   ├── genre_encoder.pkl
│   ├── int_to_note.pkl
│   ├── note_to_int.pkl
│   ├── X_midi.npy
│   ├── X_text.npy
│   ├── y_genre.npy
│   ├── y_midi.npy
│
├── notebook
│   ├── 01_text_preprocessing.ipynb
│   ├── 02_text_feature_extraction.ipynb
│   ├── 03_midi_preprocessing.ipynb
│   ├── 04_midi_sequence_creation.ipynb
│   ├── 05_midi_lstm_training.ipynb
│   ├── 06_music_generation.ipynb
│   ├── 07_text_to_music_generation.ipynb
│   ├── 08_text_lstm_training.ipynb
│   ├── 09_evaluation.ipynb
│
├── app.py
├── generated_Rock.mid
├── generated_Classical.mid
└── README.md
```

---

# ⚙️ Technologies Used

### Programming Language
- Python

### Frameworks
- Streamlit

### Libraries
- TensorFlow / Keras
- NumPy
- Pandas
- Pickle
- Music21
- Scikit-learn

---

# 🤖 Deep Learning Models

## 1️⃣ Text Genre Classification Model

Architecture:

```
Embedding Layer (64)
↓
GlobalAveragePooling1D
↓
Dense Layer (64)
↓
Dropout
↓
Softmax Output Layer (6 Genres)
```

Predicted Genres:

- Rock
- Jazz
- Classical
- EDM
- Hip-Hop
- Ambient

---

## 2️⃣ LSTM Music Generation Model

The music generation model uses **Long Short-Term Memory (LSTM)** networks to learn sequential patterns from MIDI note sequences.

### Parameters

```
Sequence Length : 100
Generation Length : 300
Sampling Method : Probability Sampling
```

The model predicts the **next note in the sequence** based on previously generated notes.

---

# 🎼 MIDI Generation Process

1. Generated note indices are mapped using:

```
int_to_note.pkl
```

2. Notes are converted into **music21 Note objects**

3. A **music stream** is created

4. The stream is exported as a **.mid file**

Example output:

```
generated_Rock.mid
generated_Jazz.mid
generated_EDM.mid
```

---

# 🌐 Running the Application

## 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

## 3️⃣ Open in Browser

```
http://localhost:8501
```

Enter a text prompt and generate music automatically.

---

# 🧪 Example Inputs

```
Generate a fast rock guitar track
Create a smooth jazz saxophone melody
Compose a classical piano piece
Produce an energetic EDM dance beat
Generate a hip hop rap beat
Create a relaxing ambient soundscape
```

---

# ⚠️ Possible Errors

### Invalid Prompt
Occurs when the input is empty or lacks musical context.

### Invalid Note Token
Occurs when a generated token cannot be mapped to a valid musical note.

### Instrument Mapping Error
Occurs when an unsupported instrument is used in the music21 library.

### File Write Permission Error
Occurs when the MIDI file cannot be written due to system permissions.

---

# 📊 Evaluation

The system evaluates generated music using feature-based metrics:

- BPM (Tempo)
- Note Density
- Pitch Range
- Pitch Standard Deviation

These metrics help verify that the generated music aligns with **genre characteristics**.

---

# 👨‍💻 Authors

Mini Project – III-II

- Praharsha Kandukuri
- Pavan Srinivas
- Isaac Aaron

---

# 📜 License

This project is developed for **academic and research purposes**.