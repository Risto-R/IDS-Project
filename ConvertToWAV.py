import pydub
import os
mp3_folder = "Audio_MP3" # Location of mp3 audio files
wav_folder = "Audio_WAV/" # Location where wav files will be saved
# Path to FFMPEG bin
os.environ["PATH"] += os.pathsep + 'FFMPEG/bin'

dir = os.fsdecode(mp3_folder)
for path in os.listdir(dir):
    src = os.path.abspath(os.path.join(dir, path))
    dst = wav_folder + path[:-3] + str("wav")
    sound = pydub.AudioSegment.from_mp3(src)
    sound.export(dst, format("wav"))
