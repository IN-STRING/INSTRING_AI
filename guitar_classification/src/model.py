import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(
        self,
        n_classes=3,
        n_mels=128
    ):
        super().__init__()

        # 1️⃣ CNN: 주파수/시간 국소 패턴 추출
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        # Mel(128) → 64 → 32
        self.lstm_input_size = 32 * (n_mels // 4)

        # 2️⃣ LSTM: 시간 흐름 학습
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        # 3️⃣ FC: 최종 분류
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        """
        x: (B, 1, 128, T)
        """
        x = self.cnn(x)
        # x: (B, 32, 32, T')

        B, C, F, T = x.shape

        # LSTM 입력 형태로 변환
        x = x.permute(0, 3, 1, 2)      # (B, T', C, F)
        x = x.contiguous().view(B, T, C * F)  # (B, T', feature)

        # LSTM
        out, _ = self.lstm(x)          # (B, T', 128)

        # 마지막 타임스텝만 사용
        out = out[:, -1, :]            # (B, 128)

        out = self.fc(out)             # (B, 3)
        return out


if __name__ == "__main__":
    model = CNNLSTM()
    dummy = torch.randn(1, 1, 128, 173)
    out = model(dummy)
    print(out.shape)