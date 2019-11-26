import pandas as pd
import librosa
import numpy as np
import os

df = pd.read_csv("Datasets/validated_RU.tsv", sep="\t")
df = df.dropna(subset=["gender"])
df = df[df.gender != "other"]
df = df[["path", "age", "gender"]]
# New Columns
column_chroma_stft = []
column_rms = []
column_spec_cent = []
column_spec_bw = []
column_rolloff = []
column_zcr = []
column_mfcc = []

# Get all audio file paths and names
WAV_FilePaths = []
WAV_FileNames = []
dir = os.fsdecode("Audio_WAV")
for path in os.listdir(dir):
    src = os.path.abspath(os.path.join(dir, path))
    name = path[:-3] + str("mp3")
    WAV_FilePaths.append(src)
    WAV_FileNames.append(name)

# Remove all entries from dataframe where we don't have the given audio file
df = df[df["path"].isin(WAV_FileNames)]
dictionary = dict(zip(WAV_FileNames, WAV_FilePaths))

# Get speech features, add them to the dataframe and export to csv file
for index, row in df.iterrows():
        src = dictionary.get(row.path)
        y, sr = librosa.load(src)
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        rms = np.mean(librosa.feature.rms(y=y))
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr))
        column_chroma_stft.append(chroma_stft)
        column_rms.append(rms)
        column_spec_cent.append(spec_cent)
        column_spec_bw.append(spec_bw)
        column_rolloff.append(rolloff)
        column_zcr.append(zcr)
        column_mfcc.append(mfcc)

# Add columns to dataframe
df = df.assign(chroma_stft=column_chroma_stft)
df = df.assign(rms=column_rms)
df = df.assign(spec_cent=column_spec_cent)
df = df.assign(spec_bw=column_spec_bw)
df = df.assign(rolloff=column_rolloff)
df = df.assign(zcr=column_zcr)
df = df.assign(mfcc=column_mfcc)
df.to_csv("Datasets/validated_RU_SpeechFeatures.csv", encoding="utf-8", index=False)
