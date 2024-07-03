import wave
import numpy as np
import cupy as cp
import librosa
from scipy.io.wavfile import write, read
from python_speech_features import mfcc
from itertools import zip_longest
from cuml.metrics import pairwise_distances

def read_wav(file):
    print(f"Reading {file}...")
    with wave.open(file, 'rb') as wf:
        params = wf.getparams()
        frames = wf.readframes(params.nframes)
        audio = np.frombuffer(frames, dtype=np.int16)
        audio = audio.reshape(-1, params.nchannels)
        return audio, params

def resample_audio(audio, original_sr, target_sr):
    audio = audio.astype(np.float32)
    resampled_audio = np.vstack([librosa.resample(audio[:, i], orig_sr=original_sr, target_sr=target_sr) for i in range(audio.shape[1])]).T
    return resampled_audio, target_sr

def zero_crossings(audio):
    print("Calculating zero crossings...")
    audio = cp.array(audio)
    crossings = cp.where(cp.diff(cp.sign(audio)))[0]
    return crossings.get()

def slice_audio(audio, zero_crossings, num_crossings):
    print("Slicing audio into segments...")
    slices = []
    for i in range(0, len(zero_crossings), num_crossings):
        start = zero_crossings[i] if i < len(zero_crossings) else None
        end = zero_crossings[i + num_crossings] if i + num_crossings < len(zero_crossings) else None
        if start is not None and end is not None:
            slices.append(audio[start:end])
    return slices

def windowed_features(slices, samplerate):
    features = []
    for i, sl in enumerate(slices):
        sl = np.array(sl, dtype=np.float32)
        mfccs = librosa.feature.mfcc(y=sl, sr=samplerate, n_mfcc=13, n_fft=len(sl))
        features.append(mfccs)
        if i % 9 == 0:
            print(f"Calculated MFCC {i + 1}/{len(slices)}")
    return features

def match_slices(features1, features2):
    print("Matching slices using MFCC...")
    best_matches = []
    max_len = max([f.shape[1] for f in features1 + features2])

    # Pad features to the same length
    padded_features1 = [np.pad(f, ((0, 0), (0, max_len - f.shape[1])), 'constant') for f in features1]
    padded_features2 = [np.pad(f, ((0, 0), (0, max_len - f.shape[1])), 'constant') for f in features2]

    padded_features1 = cp.array(padded_features1).reshape(len(padded_features1), -1)
    padded_features2 = cp.array(padded_features2).reshape(len(padded_features2), -1)

    distances = pairwise_distances(padded_features1, padded_features2, metric='euclidean').get()

    best_matches = np.argmin(distances, axis=1)
    return best_matches

def reconstruct_audio(matched_indices, original_slices):
    print("Reconstructing audio...")
    reconstructed = []
    for idx in matched_indices:
        if idx < len(original_slices):
            reconstructed.append(original_slices[idx])
    return np.concatenate(reconstructed)

def save_wav(file, audio, params):
    print(f"Saving output to {file}...")
    audio = audio.astype(np.int16)
    with wave.open(file, 'wb') as wf:
        wf.setparams(params)
        wf.writeframes(audio.tobytes())

def pad_audio_to_match_length(audio1, audio2):
    print("Padding audio to match lengths...")
    max_length = max(len(audio1), len(audio2))
    padded_audio1 = np.pad(audio1, (0, max_length - len(audio1)), 'constant')
    padded_audio2 = np.pad(audio2, (0, max_length - len(audio2)), 'constant')
    return padded_audio1, padded_audio2

def process_wav_files(file1, file2, output_file, num_crossings):
    audio1, params1 = read_wav(file1)
    audio2, params2 = read_wav(file2)
    samplerate1, _ = read(file1)
    samplerate2, _ = read(file2)

    if samplerate1 != samplerate2:
        audio2, samplerate2 = resample_audio(audio2, samplerate2, samplerate1)

    # Process the first channel
    zero_crossings1_ch1 = zero_crossings(audio1[:, 0])
    zero_crossings2_ch1 = zero_crossings(audio2[:, 0])

    slices1_ch1 = slice_audio(audio1[:, 0], zero_crossings1_ch1, num_crossings)
    slices2_ch1 = slice_audio(audio2[:, 0], zero_crossings2_ch1, num_crossings)

    windowed_features1_ch1 = windowed_features(slices1_ch1, samplerate1)
    windowed_features2_ch1 = windowed_features(slices2_ch1, samplerate1)


    matched_slices1_ch1 = match_slices(windowed_features1_ch1, windowed_features2_ch1)

    reconstructed_ch1 = reconstruct_audio(matched_slices1_ch1, slices2_ch1)

    if params1.nchannels == 2:
        # Process the second channel
        zero_crossings1_ch2 = zero_crossings(audio1[:, 1])
        zero_crossings2_ch2 = zero_crossings(audio2[:, 1])

        slices1_ch2 = slice_audio(audio1[:, 1], zero_crossings1_ch2, num_crossings)
        slices2_ch2 = slice_audio(audio2[:, 1], zero_crossings2_ch2, num_crossings)

        windowed_features1_ch2 = windowed_features(slices1_ch2, samplerate1)
        windowed_features2_ch2 = windowed_features(slices2_ch2, samplerate1)

        matched_slices1_ch2 = match_slices(windowed_features1_ch2, windowed_features2_ch2)

        reconstructed_ch2 = reconstruct_audio(matched_slices1_ch2, slices2_ch2)

        # Ensure both channels have the same length
        reconstructed_ch1, reconstructed_ch2 = pad_audio_to_match_length(reconstructed_ch1, reconstructed_ch2)

        audio1_reconstructed = np.column_stack((reconstructed_ch1, reconstructed_ch2))
    else:
        audio1_reconstructed = reconstructed_ch1

    save_wav(output_file, audio1_reconstructed, params1)
    print("Processing complete.")

# Example usage:
file2= 'input1.wav'
file1 = 'input2.wav'
output_file = 'output.wav'
num_crossings = 256

process_wav_files(file1, file2, output_file, num_crossings)
