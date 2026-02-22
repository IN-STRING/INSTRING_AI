import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(
        self,
        n_classes=3,
        n_mels=128
    ):
        super().__init__()

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

        self.lstm_input_size = 32 * (n_mels // 4)

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        """
        x: (B, 1, 128, T)
        """
        x = self.cnn(x)

        B, C, F, T = x.shape

        x = x.permute(0, 3, 1, 2) 
        x = x.contiguous().view(B, T, C * F)  

        # LSTM
        out, _ = self.lstm(x)      

        # 마지막 타임스텝만 사용
        out = out[:, -1, :]    

        out = self.fc(out)      
        return out


if __name__ == "__main__":
    model = CNNLSTM()
    dummy = torch.randn(1, 1, 128, 173)
    out = model(dummy)
    print(out.shape)