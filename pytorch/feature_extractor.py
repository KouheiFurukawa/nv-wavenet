import torch
from torch import nn
import torchvision.models as models


class ResidualConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1):
        super(ResidualConv1d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(out_channel),
            nn.GELU(),
            nn.Conv1d(out_channel, out_channel, kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(out_channel)
        )

    def forward(self, x):
        return x + self.net(x)


class DynamicEncoder(nn.Module):
    def __init__(self, embed_dim=512, mode='dynamic'):
        super(DynamicEncoder, self).__init__()
        self.net = []
        self.net.append(nn.Sequential(
            nn.Conv1d(1 if mode == 'dynamic' else 512, 64, 3, padding=1),
            nn.Conv1d(64, 64, 9, stride=4, padding=4),
            nn.Dropout(p=0.5),
            nn.GELU()
        ))
        self.net.append(nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1),
            nn.Conv1d(128, 128, 9, stride=4, padding=4),
            nn.Dropout(p=0.5),
            nn.GELU()
        ))
        self.net.append(nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            ResidualConv1d(256, 256, 3, padding=1),
            nn.Conv1d(256, 256, 3, stride=2, padding=1),
            nn.Dropout(p=0.5),
            nn.GELU()
        ))
        self.net.append(nn.Sequential(
            nn.Conv1d(256, 256, 3, padding=1),
            ResidualConv1d(256, 256, 3, padding=1),
            nn.Conv1d(256, 256, 3, stride=2, padding=1),
            nn.Dropout(p=0.5),
            nn.GELU()
        ))
        self.net.append(nn.Sequential(
            nn.Conv1d(256, 512, 3, padding=1),
            ResidualConv1d(512, 512, 3, padding=1),
            nn.Conv1d(512, 512, 3, stride=2, padding=1),
            nn.Dropout(p=0.5),
            nn.GELU(),
        ))
        self.net.append(nn.Sequential(
            ResidualConv1d(512, 512, 3, padding=1),
            ResidualConv1d(512, 512, 3, padding=1),
            nn.Conv1d(512, 512, 3, stride=2, padding=1),
            nn.Dropout(p=0.5),
            nn.GELU(),
            nn.Conv1d(512, embed_dim, 1, 1, 0)
        ))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(1)
        x = self.net(x)
        return x


class StaticEncoder(nn.Module):

    def __init__(self, base_model, out_dim):
        super(StaticEncoder, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=True)}

        self.backbone = self._get_basemodel(base_model)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                        bias=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, out_dim)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        model = self.resnet_dict[model_name]
        return model

    def forward(self, x):
        return self.backbone(x)