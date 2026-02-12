import torch
import librosa
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from model import CNNLSTM
from analyzer import estimate_temp, tempo_label

# --- 설정 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "best_model.pt")
CLASS_NAMES = ["arpeggio", "fingerstyle", "stroke"]
TEMPO_NAMES = ["slow", "mid", "fast"]

# 2. 모델 로드 함수
def load_trained_model():
    model = CNNLSTM().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def preprocess_for_model(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram = librosa.power_to_db(spectrogram)

    input_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return input_tensor


def analyze_guitar_performance(audio_file_name):
    file_path = os.path.join(BASE_DIR, "test", audio_file_name)
    
    if not os.path.exists(file_path):
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return

    print(f"분석 중: {audio_file_name}...")

    model = load_trained_model()
    input_data = preprocess_for_model(file_path)
    
    with torch.no_grad():
        output = model(input_data)
        style_idx = torch.argmax(output, dim=1).item()
        style_name = CLASS_NAMES[style_idx]

    bpm = estimate_temp(file_path)
    t_label_idx = tempo_label(bpm)
    t_label_name = TEMPO_NAMES[t_label_idx]

    print("\n" + "="*30)
    print(f"주법 결과: {style_name}")
    print(f"템포 결과: {bpm:.0f} BPM ({t_label_name})")
    print("="*30)

if __name__ == "__main__":
    analyze_guitar_performance("test4.mp3")