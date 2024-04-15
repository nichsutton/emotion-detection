import numpy as np
import librosa

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data) 
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data):
    return librosa.effects.time_stretch(data, rate=0.8)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

def extract_features (data, samplerate):
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

def get_feautures (path, sample_rate):
    data, samplerate = librosa.load(path, duration=2.5, offset=0.6)

    res1 = extract_features(data, samplerate)
    result = np.array(res1)

    noise_data = noise(data)
    res2 = extract_features(noise_data, samplerate)
    result = np.vstack((result, res2)) # stacking vertically
    
    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch, sample_rate)
    result = np.vstack((result, res3)) # stacking vertically
    
    return result

