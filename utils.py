# Utils and helper functions
import soundfile
import numpy as np
import librosa
import glob
import os
from sklearn.model_selection import train_test_split


def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
    return result

# We will use the following emotions only
ALLOWED_EMOTIONS = [
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
]
# Allows us to load the data and features from the disk
def load_data(test_size=0.2):
    X, y = [], []
    for folder in os.listdir("data/train/"):
        for file in glob.glob("data/train/" + folder + "/*.wav"):
            # get the basename of the audio file
            basename = os.path.basename(file)
            # extract the emotion from the label
            print("Extracting features from file {}".format(basename))
            emotion = basename.split("-")[-1].split(".")[0].split("_")[-1]
            if emotion not in ALLOWED_EMOTIONS:
                continue
            # extract features
            features = extract_feature(file, mfcc=True, chroma=True, mel=True)
            # append to data
            X.append(features)
            y.append(emotion)
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)

# allows us to load the validation data
def load_validation_data():
    X, y = [], []
    for folder in os.listdir("data/validation/"):
        for file in glob.glob("data/validation/" + folder + "/*.wav"):
            # get the basename of the audio file
            basename = os.path.basename(file)
            # extract the emotion from the label
            print("Extracting features from file {}".format(basename))
            emotion = basename.split("-")[-1].split(".")[0].split("_")[-1]
            if emotion not in ALLOWED_EMOTIONS:
                continue
            # extract features
            features = extract_feature(file, mfcc=True, chroma=True, mel=True)
            # append to data
            X.append(features)
            y.append(emotion)
    return np.array(X), y

