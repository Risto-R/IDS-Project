# Introduction to Data Science 2019 Project
## Speaker Recognition - Gender and accent classification based on the speaker's voice

For this project we used datasets from [Common Voice by Mozilla](https://voice.mozilla.org)
Datasets consisted of tsv files and mp3 audio clips.

To extract features from audio files we used LibROSA and the code used is in the [Notebook](Notebook.ipynb) and in [SpeechFeatures.py](SpeechFeatures.py)
[SpeechFeatures.py](SpeechFeatures.py) reads in the tsv file, gets the features and outputs them into a csv file (Files can be specified in the code)

Even though features can be extracted from .mp3 files there might be errors and in our case there was so the .mp3 files needed to be converted to .WAV file format. For this ffmpeg or avconv is needed (We used FFMPEG) and the code for that: [ConvertToWAV.py](ConvertToWAV.py)

Datasets with extracted features: [Google Drive](https://drive.google.com/drive/folders/10N8crdpFxvhx3oCJk_jVDxM0qI4NVcB2)

Poster for presentation: [Poster](Poster.pptx)

