import torch
import torch.nn as nn

# ==========================================
# Helper: ShallowNet encoder for raw EEG
# ==========================================
class shallownet(nn.Module):
    def __init__(self, out_dim, C, T):
        super(shallownet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (C, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.Dropout(0.5),
        )
        self.out = nn.Linear(1040 * (T // 200), out_dim)

    def forward(self, x):  # input: (batch, 1, C, T)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


# ==========================================
# GLMNet for raw EEG
# ==========================================
class glfnet(nn.Module):
    def __init__(self, out_dim, emb_dim, C, T):
        super(glfnet, self).__init__()
        self.globalnet = shallownet(emb_dim, C, T)

        # Occipital channels (EEG 10-20 system, channels 50–61 in 62-channel cap)
        self.occipital_index = list(range(50, 62))
        self.occipital_localnet = shallownet(emb_dim, 12, T)

        self.out = nn.Linear(emb_dim * 2, out_dim)

    def forward(self, x):  # input: (batch, 1, C, T)
        global_feature = self.globalnet(x)
        global_feature = global_feature.view(x.size(0), -1)

        occipital_x = x[:, :, self.occipital_index, :]
        occipital_feature = self.occipital_localnet(occipital_x)

        out = self.out(torch.cat((global_feature, occipital_feature), 1))
        return out


# ==========================================
# Helper: MLP encoder for DE/PSD features
# ==========================================
class mlpnet(nn.Module):
    def __init__(self, out_dim, input_dim):
        super(mlpnet, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):  # input: (batch, C, F)
        out = self.net(x)
        return out


# ==========================================
# GLMNet for DE/PSD features
# ==========================================
class glfnet_mlp(nn.Module):
    def __init__(self, out_dim, emb_dim, input_dim):
        super(glfnet_mlp, self).__init__()
        self.globalnet = mlpnet(emb_dim, input_dim)

        # Occipital subset for DE/PSD (12 channels × F features each)
        self.occipital_index = list(range(50, 62))
        self.occipital_localnet = mlpnet(emb_dim, 12 * (input_dim // 62))

        self.out = nn.Linear(emb_dim * 2, out_dim)

    def forward(self, x):  # input: (batch, C, F)
        global_feature = self.globalnet(x)
        occipital_x = x[:, self.occipital_index, :]
        occipital_feature = self.occipital_localnet(occipital_x)

        out = self.out(torch.cat((global_feature, occipital_feature), 1))
        return out
