import streamlit as st
import os
import tempfile
import json
import urllib.request
from urllib.parse import urlparse
import yt_dlp
from pydub import AudioSegment
from speechbrain.pretrained.interfaces import foreign_class

# Constants
TEMP_DIR = "/tmp"  # Streamlit Cloud only allows writing to /tmp
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize model (cached to avoid reloading)
@st.cache_resource
def load_model():
    return foreign_class(
        source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
        savedir=os.path.join(TEMP_DIR, "model")  # Save to temp dir
    )

classifier = load_model()

def extract_loom_id(url):
    parsed_url = urlparse(url)
    return parsed_url.path.split("/")[-1]

def download_loom_video(url, filename):
    try:
        video_id = extract_loom_id(url)
        request = urllib.request.Request(
            url=f"https://www.loom.com/api/campaigns/sessions/{video_id}/transcoded-url",
            headers={"User-Agent": "Mozilla/5.0"},
            method="POST"
        )
        with urllib.request.urlopen(request) as response:
            content = json.loads(response.read().decode("utf-8"))
            urllib.request.urlretrieve(content["url"], filename)
        return filename
    except Exception as e:
        raise RuntimeError(f"Loom download failed: {str(e)}")

def download_youtube_audio(url):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(TEMP_DIR, 'yt_audio.%(ext)s'),
            'quiet': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_path = ydl.prepare_filename(info).replace('.webm', '.wav').replace('.m4a', '.wav')
        return audio_path
    except Exception as e:
        raise RuntimeError(f"YouTube download failed: {str(e)}")

def download_direct_video(url):
    try:
        with urllib.request.urlopen(url) as response:
            if response.status != 200:
                raise RuntimeError("Failed to download video")
            video_path = os.path.join(TEMP_DIR, "direct_video.mp4")
            with open(video_path, 'wb') as f:
                f.write(response.read())
            return video_path
    except Exception as e:
        raise RuntimeError(f"Video download failed: {str(e)}")

def extract_audio(video_path):
    try:
        audio = AudioSegment.from_file(video_path)
        wav_path = os.path.join(TEMP_DIR, os.path.basename(video_path).replace(".mp4", ".wav"))
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        raise RuntimeError(f"Audio extraction failed: {str(e)}")

def get_speech_segments(audio_path, min_silence_len=700, silence_thresh=-40, duration=10000):
    try:
        audio = AudioSegment.from_wav(audio_path)
        nonsilent_ranges = AudioSegment.silent.detect_nonsilent(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )
        
        if not nonsilent_ranges:
            raise RuntimeError("No speech segments detected")
            
        start_ms = nonsilent_ranges[0][0]
        end_ms = min(start_ms + duration, len(audio))
        segment = audio[start_ms:end_ms]
        
        segment_path = os.path.join(TEMP_DIR, "temp_segment.wav")
        segment.export(segment_path, format="wav")
        return segment_path
    except Exception as e:
        raise RuntimeError(f"Speech detection failed: {str(e)}")

def classify_audio(wav_path):
    try:
        segment_path = get_speech_segments(wav_path)
        out_prob, score, index, label = classifier.classify_file(segment_path)
        return label[0], float(score[0]) * 100
    except Exception as e:
        raise RuntimeError(f"Classification failed: {str(e)}")

def cleanup_files(*paths):
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except:
            pass

# Streamlit UI
st.title("Accent Classifier for English Speakers")

with st.form("video_input_form"):
    video_url = st.text_input("Enter video URL (YouTube, Loom, or direct MP4)")
    uploaded_file = st.file_uploader("Or upload a video file", type=["mp4", "mov", "avi"])
    submit_button = st.form_submit_button("Process")

if submit_button:
    video_path, wav_path = None, None
    try:
        with st.spinner('Processing...'):
            if video_url:
                if any(x in video_url for x in ["youtube.com", "youtu.be"]):
                    wav_path = download_youtube_audio(video_url)
                elif "loom.com" in video_url:
                    video_path = os.path.join(TEMP_DIR, "loom_video.mp4")
                    download_loom_video(video_url, video_path)
                    wav_path = extract_audio(video_path)
                elif video_url.endswith(".mp4"):
                    video_path = download_direct_video(video_url)
                    wav_path = extract_audio(video_path)
                else:
                    st.error("Unsupported URL format")
            elif uploaded_file:
                video_path = os.path.join(TEMP_DIR, uploaded_file.name)
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                wav_path = extract_audio(video_path)
            else:
                st.error("Please provide input")

            if wav_path:
                label, confidence = classify_audio(wav_path)
                st.success(f"**Detected Accent:** {label}")
                st.info(f"**Confidence:** {confidence:.2f}%")
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        cleanup_files(video_path, wav_path)
