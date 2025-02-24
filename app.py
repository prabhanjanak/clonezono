import streamlit as st
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device
import tempfile
import os

# Load Zonos Model
st.session_state["model"] = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

# Streamlit UI
st.title("🎙️ AI Voice Cloning & Enhancement")
st.write("Upload a voice sample and adjust settings to enhance the generated speech.")

uploaded_file = st.file_uploader("Upload your voice sample (MP3 or WAV)", type=["mp3", "wav"])

# Emotion sliders
st.subheader("Emotion Enhancement")
happiness = st.slider("Happiness", 0.0, 1.0, 0.1)
sadness = st.slider("Sadness", 0.0, 1.0, 0.05)
disgust = st.slider("Disgust", 0.0, 1.0, 0.05)
fear = st.slider("Fear", 0.0, 1.0, 0.05)
surprise = st.slider("Surprise", 0.0, 1.0, 0.05)
anger = st.slider("Anger", 0.0, 1.0, 0.05)
neutral = st.slider("Neutral", 0.0, 1.0, 0.2)

# Unconditional Toggles
st.subheader("Unconditional Toggles")
unconditional_speaker = st.checkbox("Unconditional Speaker")
unconditional_emotion = st.checkbox("Unconditional Emotion")

text_input = st.text_area("Enter text to generate speech")

def process_audio(uploaded_file):
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            temp_wav.write(uploaded_file.read())
            return temp_wav.name
    return None

if st.button("Generate Cloned Voice") and uploaded_file and text_input:
    audio_path = process_audio(uploaded_file)
    wav, sr = torchaudio.load(audio_path)
    speaker = st.session_state["model"].make_speaker_embedding(wav, sr)
    
    cond_dict = make_cond_dict(
        text=text_input, 
        speaker=speaker if not unconditional_speaker else None,
        emotion=None if unconditional_emotion else {
            "happiness": happiness,
            "sadness": sadness,
            "disgust": disgust,
            "fear": fear,
            "surprise": surprise,
            "anger": anger,
            "neutral": neutral
        },
        language="en-us"
    )
    conditioning = st.session_state["model"].prepare_conditioning(cond_dict)
    codes = st.session_state["model"].generate(conditioning)
    
    output_wav = st.session_state["model"].autoencoder.decode(codes).cpu()
    output_path = "generated_voice.wav"
    torchaudio.save(output_path, output_wav[0], st.session_state["model"].autoencoder.sampling_rate)
    
    st.audio(output_path, format="audio/wav")
    st.success("Voice cloning completed! Download your file below.")
    st.download_button("Download Cloned Voice", output_path)
