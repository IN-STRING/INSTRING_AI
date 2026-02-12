import librosa
import numpy as np

def estimate_temp(file_path):
    y, sr = librosa.load(file_path)
    
    duration = librosa.get_duration(y=y, sr=sr)
    hop_length = len(y) // 5
    
    tempos = []
    for i in range(5):
        start = i * hop_length
        end = (i + 1) * hop_length
        y_chunk = y[start:end]
        
        onset_env = librosa.onset.onset_strength(y=y_chunk, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

        t_val = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        tempos.append(t_val)
    
    final_tempo = np.median(tempos)
    
    if final_tempo < 60:
        final_tempo *= 2
    elif final_tempo > 180:
        final_tempo /= 2
        
    return final_tempo

def tempo_label(tempo):
    if tempo < 80:
        return 0  # slow
    elif tempo <= 120:
        return 1  # mid
    else:
        return 2  # fast