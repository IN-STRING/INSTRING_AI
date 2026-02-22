import librosa
import numpy as np

def estimate_temp(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)

    duration = librosa.get_duration(y=y, sr=sr)
    
    if duration >= 6:
        n_chunks = 5
    elif duration >= 3:
        n_chunks = 3
    else:
        n_chunks = 1

    chunk_length = len(y) // n_chunks
    tempos = []

    for i in range(n_chunks):
        start = i * chunk_length
        end = (i + 1) * chunk_length
        y_chunk = y[start:end]

        if len(y_chunk) < sr: 
            continue

        onset_env = librosa.onset.onset_strength(y=y_chunk, sr=sr)
        tempo, _ = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr
        )

        t_val = float(tempo)
        tempos.append(t_val)

    if len(tempos) == 0:
        return 0

    final_tempo = np.median(tempos)

    if final_tempo < 60:
        final_tempo *= 2
    elif final_tempo > 180:
        final_tempo /= 2

    return final_tempo