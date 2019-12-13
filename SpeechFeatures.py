import pandas as pd
import librosa
import numpy as np
import os

df = pd.read_csv("Datasets/DATASET_NAME.tsv", sep="\t")
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
column_mfcc1 = []
column_mfcc2 = []
column_mfcc3 = []
column_mfcc4 = []
column_mfcc5 = []
column_mfcc6 = []
column_mfcc7 = []
column_mfcc8 = []
column_mfcc9 = []
column_mfcc10 = []
column_mfcc11 = []
column_mfcc12 = []
column_mfcc13 = []
column_mfcc14 = []
column_mfcc15 = []
column_mfcc16 = []
column_mfcc17 = []
column_mfcc18 = []
column_mfcc19 = []
column_mfcc20 = []

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
kokku = len(df)
print(kokku)
countnr = 0
print(df["gender"].value_counts())
# Get speech features, add them to the dataframe and export to csv file
for index, row in df.iterrows():
        print(countnr)
        src = dictionary.get(row.path)
        y, sr = librosa.load(src)
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        rms = np.mean(librosa.feature.rms(y=y))
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        column_mfcc1.append(np.mean(mfcc[0]))
        column_mfcc2.append(np.mean(mfcc[1]))
        column_mfcc3.append(np.mean(mfcc[2]))
        column_mfcc4.append(np.mean(mfcc[3]))
        column_mfcc5.append(np.mean(mfcc[4]))
        column_mfcc6.append(np.mean(mfcc[5]))
        column_mfcc7.append(np.mean(mfcc[6]))
        column_mfcc8.append(np.mean(mfcc[7]))
        column_mfcc9.append(np.mean(mfcc[8]))
        column_mfcc10.append(np.mean(mfcc[9]))
        column_mfcc11.append(np.mean(mfcc[10]))
        column_mfcc12.append(np.mean(mfcc[11]))
        column_mfcc13.append(np.mean(mfcc[12]))
        column_mfcc14.append(np.mean(mfcc[13]))
        column_mfcc15.append(np.mean(mfcc[14]))
        column_mfcc16.append(np.mean(mfcc[15]))
        column_mfcc17.append(np.mean(mfcc[16]))
        column_mfcc18.append(np.mean(mfcc[17]))
        column_mfcc19.append(np.mean(mfcc[18]))
        column_mfcc20.append(np.mean(mfcc[19]))
        column_chroma_stft.append(chroma_stft)
        column_rms.append(rms)
        column_spec_cent.append(spec_cent)
        column_spec_bw.append(spec_bw)
        column_rolloff.append(rolloff)
        column_zcr.append(zcr)
        countnr += 1

# Add columns to dataframe
df = df.assign(chroma_stft=column_chroma_stft)
df = df.assign(rms=column_rms)
df = df.assign(spec_cent=column_spec_cent)
df = df.assign(spec_bw=column_spec_bw)
df = df.assign(rolloff=column_rolloff)
df = df.assign(zcr=column_zcr)
df = df.assign(mfcc1=column_mfcc1)
df = df.assign(mfcc2=column_mfcc2)
df = df.assign(mfcc3=column_mfcc3)
df = df.assign(mfcc4=column_mfcc4)
df = df.assign(mfcc5=column_mfcc5)
df = df.assign(mfcc6=column_mfcc6)
df = df.assign(mfcc7=column_mfcc7)
df = df.assign(mfcc8=column_mfcc8)
df = df.assign(mfcc9=column_mfcc9)
df = df.assign(mfcc10=column_mfcc10)
df = df.assign(mfcc11=column_mfcc11)
df = df.assign(mfcc12=column_mfcc12)
df = df.assign(mfcc13=column_mfcc13)
df = df.assign(mfcc14=column_mfcc14)
df = df.assign(mfcc15=column_mfcc15)
df = df.assign(mfcc16=column_mfcc16)
df = df.assign(mfcc17=column_mfcc17)
df = df.assign(mfcc18=column_mfcc18)
df = df.assign(mfcc19=column_mfcc19)
df = df.assign(mfcc20=column_mfcc20)


df.to_csv("Datasets/Dataset_NAME.csv", encoding="utf-8", index=False)
