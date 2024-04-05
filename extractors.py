import numpy as np
import librosa
from normalizers import noise

def extract_features (data, samplerate) :
    # ZCR extraction
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    # Chroma_stft extraction
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=samplerate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    # MFCC extraction
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=samplerate).T, axis=0)
    result = np.hstack((result, mfcc))

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    # MelSpectogram extraction
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=samplerate).T, axis=0)
    result = np.hstack((result, mel))

    return result

def get_feautures (path) :
    data, samplerate = librosa.load(path, duration=2.5, offset=0.6)

    res1 = extract_features(data, samplerate)
    result = np.array(res1)

    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2)) # stacking vertically
    
    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3)) # stacking vertically
    
    return result

