import pydub
import ffmpeg
import os
mp3_folder = "data/clips/" # Location of mp3 audio files
wav_folder = "data/wav_clips/" # Location where wav files will be saved
os.environ["PATH"] += os.pathsep + 'FFMPEG/bin' # Path to FFMPEG bin

dir = os.fsdecode(mp3_folder)
for path in os.listdir(dir):
    try:
        src = os.path.abspath(os.path.join(dir, path))
        dst = wav_folder + path[:-3] + str("wav")
        sound = pydub.AudioSegment.from_mp3(src)
        sound.export(dst, format("wav"))
        os.remove(src)
    except:
        print("Broken file " + path)

