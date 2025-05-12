import streamlit as st

# video download
import requests
def download_video(url, filename="https://www.youtube.com/watch?app=desktop&v=ZeJM7QmufO8"):
    r = requests.get(url)
    with open(filename, "wb") as f:
        f.write(r.content)
    return filename

# Audio extract using (ffmpeg)
import os
def extract_audio(video_path, audio_path="extracted_audio.wav"):
    os.system(f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path}")
    return audio_path

# accent prediction function
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch
import torchaudio

model_name = "facebook/wav2vec2-large-960h-lv60-self"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

def predict_accent(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    inputs = processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predicted_id = torch.argmax(logits, dim=-1).item()
    confidence = torch.softmax(logits, dim=-1).max().item()

    # class
    accent_classes = ["American", "British", "Australian"]
    predicted_class = accent_classes[predicted_id % len(accent_classes)]

    return predicted_class, round(confidence * 100, 2)


st.title("Accent Detection Tool")
video_url = st.text_input("https://www.youtube.com/watch?app=desktop&v=ZeJM7QmufO8")

if st.button("Analyze"):
    if video_url:
        video_path = download_video(video_url)
        audio_path = extract_audio(video_path)
        accent, score = predict_accent(audio_path)
        
        st.success(f"Detected Accent: {accent}")
        st.info(f"Confidence Score: {score}%")