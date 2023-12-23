import cv2
import numpy as np
import librosa

def solution(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    n_fft = 2048
    hop_length = 512
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, fmax=22000)
    mean_spec = np.mean(spec)
    std_spec = np.std(spec)
    mean_threshold = 2.0
    std_deviation_threshold = 5.0
    if (mean_spec > mean_threshold and std_spec > std_deviation_threshold):
        class_name = 'metal'
    else:
        class_name = 'cardboard'
    ############################
    ############################
    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    
    return class_name
