# English-Speech-Accent-classification
Using speechbrain models for detecting English speaker accent

# How to install
I use Windows 11, Anaconda, Python 3.11, and PyTorch 2.6.0+cu12.4. You can use other PyTorch versions according to your CUDA version, read this https://pytorch.org/get-started/locally/

and before install it, first you need to install the ffmepg too, you can download it via this link : https://ffmpeg.org/download.html

```
#create venv in anaconda
conda create --name speech_accent python=3.11
conda activate speech_accent

#clone this repo
git clone https://github.com/RaffelRavionaldo/English-Speech-Accent-classification.git
cd English-Speech-Accent-classification

# install pytorch according to your CUDA version, but if you want have a same pytorch version like me (on local)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# if you don't have GPU (just CPU), use this
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# install the requirements
pip install -r requirements.txt
```

# How to run/use this app

1. On your anaconda, run this syntax: `streamlit run app_2.py`
2. After you run it, your default browser will open a streamlit UI like the image below :

![Screenshot 2025-05-23 084751](https://github.com/user-attachments/assets/586dfaef-662c-40b7-acce-4a023e9ccfc6)

3. Input your video link in text input or upload a video from your computer, for this test I am using a video YouTube (https://www.youtube.com/watch?v=0nLIGzLOXgg)
4. Click process button and wait until the output show like below image (I'm still a newbie on streamlit, so when the "system" is predicting the accent of the video, you can still enter another input and click process, this should be disabled while the process is still running but I don't know how yet)

![Screenshot 2025-05-23 085147](https://github.com/user-attachments/assets/8e85514b-56ed-4012-aefc-404c1f3d6313)

# Flow of this app
1. After the app receives your video or video link and you click process, the video will be processed. If it's the video link, it will be downloaded and saved in the temp folder
2. Extract audio from the video, detect the non-silent part of the video, and get 10 seconds of it. I only take 10 seconds for the speed of detection. If we classify all of the voices from the video, it will depend on the video length.
3. Send the audio to speech accent model, i use this model : https://huggingface.co/Jzuluaga/accent-id-commonaccent_xlsr-en-english
4. Give the output (an accent and the confidence score)
