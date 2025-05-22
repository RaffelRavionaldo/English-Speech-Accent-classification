import streamlit as st
import os
import tempfile
import torch
import json
import urllib.request
from urllib.parse import urlparse
from moviepy import VideoFileClip, AudioFileClip
from speechbrain.pretrained.interfaces import foreign_class
import yt_dlp
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# Load model once
classifier = foreign_class(
    source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier"
)

def extract_loom_id(url):
    parsed_url = urlparse(url)
    return parsed_url.path.split("/")[-1]

def download_loom_video(url, filename):
    try:
        video_id = extract_loom_id(url)
        request = urllib.request.Request(
            url=f"https://www.loom.com/api/campaigns/sessions/{video_id}/transcoded-url",
            headers={},
            method="POST"
        )
        response = urllib.request.urlopen(request)
        body = response.read()
        content = json.loads(body.decode("utf-8"))
        video_url = content["url"]
        urllib.request.urlretrieve(video_url, filename)
        return filename
    except Exception as e:
        raise RuntimeError(f"Failed to download video from Loom: {e}")

def download_youtube_audio(url):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'yt_audio.%(ext)s',
            'quiet': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '64',
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        audioclip = AudioFileClip("yt_audio.mp3")
        wav_path = "output.wav"
        audioclip.write_audiofile(wav_path, logger=None)
        audioclip.close()
        os.remove("yt_audio.mp3")
        return wav_path
    except Exception as e:
        raise RuntimeError(f"Failed to download from YouTube: {e}")

def download_direct_video(url):
    try:
        response = urllib.request.urlopen(url)
        if response.status != 200:
            raise RuntimeError("Failed to download video.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(response.read())
            return temp_file.name
    except Exception as e:
        raise RuntimeError(f"Failed to download video : {e}")

def extract_audio(video_path):
    try:
        clip = VideoFileClip(video_path)
        # audio_clip = clip.audio.subclip(0, min(duration, clip.duration))  # ambil 10 detik awal atau durasi video kalau kurang
        wav_path = video_path.replace(".mp4", ".wav")
        clip.audio.write_audiofile(wav_path)
        return wav_path
    except Exception as e:
        raise RuntimeError(f"Fail to extract the video : {e}")
    
def get_speech_segments(audio_path, min_silence_len=700, silence_thresh=-40, duration=10000):
    """
    Get speech segments with absolute position
    Detects non-silent parts in audio with precise timing
    """
    audio = AudioSegment.from_wav(audio_path)
    total_duration = len(audio)

    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )

    start_ms, original_end_ms = nonsilent_ranges[0]
    end_ms = min(start_ms + duration, total_duration)

    segment = audio[start_ms:end_ms]
    temp_path = "temp_first_segment.wav"
    segment.export(temp_path, format="wav")

    return temp_path

def classify_audio(wav_path):
    out_prob, score, index, label = classifier.classify_file(get_speech_segments(wav_path))
    confidence = float(score[0]) * 100  # convert tensor to float
    return label, confidence

def delete_file(path):
    try:
        os.remove(path)
    except:
        pass

# Streamlit UI
st.title("Accent Classifier for English Speakers")

with st.form("Input your video (it can be video link or upload)"):
    video_url = st.text_input(
        "Enter video URL (YouTube, Loom, or .mp4)",
        disabled=st.session_state.processing
    )

    uploaded_file = st.file_uploader(
        "Or upload a video file (mp4, mov, or mkv)",
        type=["mp4", "mov", "avi"],
        disabled=st.session_state.processing
    )

    if st.form_submit_button("Process"):
        video_path = None
        wav_path = None

        try:
            with st.spinner('Processing video... Please wait'):
                if video_url:
                    if "youtube.com" in video_url or "youtu.be" in video_url:
                        wav_path = download_youtube_audio(video_url)
                    elif "loom.com" in video_url:
                        video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                        download_loom_video(video_url, video_path)
                        wav_path = extract_audio(video_path)
                    elif video_url.endswith(".mp4"):
                        video_path = download_direct_video(video_url)
                        wav_path = extract_audio(video_path)
                    else:
                        st.error("URL Format unrecognized.")
                elif uploaded_file is not None:
                    video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                    with open(video_path, "wb") as f:
                        f.write(uploaded_file.read())
                    wav_path = extract_audio(video_path)
                else:
                    st.error("Please upload a file or link")

                if wav_path:
                    label, confidence = classify_audio(wav_path)
                    st.success(f"Video Accent: **{label}**")
                    st.info(f"Confidence Score: **{confidence:.2f}%**")
                else:
                    st.error("Error processing video")

        except Exception as e:
            st.error(str(e))
        finally:
            delete_file(wav_path)
            delete_file(video_path)