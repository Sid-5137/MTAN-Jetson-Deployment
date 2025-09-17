import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List

# -------------------------------
# Utility Modules
# -------------------------------
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x


# -------------------------------
# Task Attention Module
# -------------------------------
class TaskAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_tasks=2, reduction=8):
        super().__init__()
        self.shared_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.shared_bn = nn.BatchNorm2d(out_channels)

        self.task_attentions = nn.ModuleList([
            ChannelAttention(out_channels, reduction=reduction) for _ in range(num_tasks)
        ])

        self.task_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
                nn.Conv2d(out_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_tasks)
        ])

    def forward(self, x):
        shared = F.relu(self.shared_bn(self.shared_conv(x)), inplace=True)
        task_features = []
        for att, conv in zip(self.task_attentions, self.task_convs):
            task_features.append(conv(att(shared)))
        return task_features


# -------------------------------
# Encoder
# -------------------------------
class MobileNetV3Encoder(nn.Module):
    def __init__(self, backbone: str = "mobilenetv3_large", pretrained=True):
        super().__init__()
        if backbone == "mobilenetv3_small":
            mobilenet = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.layer_indices = [1, 3, 8, 11]  # Output channels: [16, 24, 48, 96]
        elif backbone == "mobilenetv3_large":
            mobilenet = models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.layer_indices = [2, 4, 7, 13]  # Output channels: [16, 24, 40, 80, 160]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.features = mobilenet.features

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layer_indices:
                features.append(x)
        return features


# -------------------------------
# Decoder Block
# -------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if skip_channels != out_channels else nn.Identity()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        skip = self.skip_conv(skip)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
        return self.conv(torch.cat([x, skip], dim=1))


# -------------------------------
# MTAN Network
# -------------------------------
class MTANNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config.model.num_classes_seg
        self.use_aux_loss = config.model.use_aux_loss

        # ---------------- Encoder ----------------
        self.encoder = MobileNetV3Encoder(
            backbone=config.model.backbone,
            pretrained=getattr(config.model, "pretrained", True)
        )
        encoder_channels = config.model.encoder_channels  # [24, 40, 80, 160]
        decoder_channels = config.model.decoder_channels  # [128, 64, 32, 16]

        # ---------------- Attention Modules ----------------
        # Use top 3 encoder features for task attention
        self.task_attention_modules = nn.ModuleList([
            TaskAttentionModule(encoder_channels[1], decoder_channels[0], num_tasks=2),  # 40 -> 128
            TaskAttentionModule(encoder_channels[2], decoder_channels[1], num_tasks=2),  # 80 -> 64
            TaskAttentionModule(encoder_channels[3], decoder_channels[2], num_tasks=2)   # 160 -> 32
        ])

        # ---------------- Decoders ----------------
        # After reversing seg_feats: [0]=32ch, [1]=64ch, [2]=128ch
        # DecoderBlock(in_channels, skip_channels, out_channels)
        self.seg_decoders = nn.ModuleList([
            DecoderBlock(decoder_channels[2], decoder_channels[1], decoder_channels[1]),  # in=32, skip=64->64, conv_in=32+64=96, out=64
            DecoderBlock(decoder_channels[1], decoder_channels[0], decoder_channels[2]),  # in=64, skip=128->32, conv_in=64+32=96, out=32
            DecoderBlock(decoder_channels[2], encoder_channels[0], decoder_channels[3])  # in=32, skip=24->16, conv_in=32+16=48, out=16
        ])
        self.depth_decoders = nn.ModuleList([
            DecoderBlock(decoder_channels[2], decoder_channels[1], decoder_channels[1]),  # in=32, skip=64->64, conv_in=32+64=96, out=64
            DecoderBlock(decoder_channels[1], decoder_channels[0], decoder_channels[2]),  # in=64, skip=128->32, conv_in=64+32=96, out=32
            DecoderBlock(decoder_channels[2], encoder_channels[0], decoder_channels[3])  # in=32, skip=24->16, conv_in=32+16=48, out=16
        ])

        # ---------------- Prediction Heads ----------------
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_channels[3], self.num_classes, 1),
            nn.Upsample(scale_factor=4, mode='nearest')
        )
        self.depth_head = nn.Sequential(
            nn.Conv2d(decoder_channels[3], 1, 1),
            nn.Upsample(scale_factor=4, mode='nearest'),
            nn.Sigmoid()
        )
        self.depth_offset = 0.01

        # Optional auxiliary segmentation head
        if self.use_aux_loss:
            self.aux_head = nn.Sequential(
                nn.Conv2d(encoder_channels[0], self.num_classes, 1),  # Use first encoder feature (24 channels)
                nn.Upsample(scale_factor=4, mode='nearest')  # Reduced from 8x to 4x to avoid OOM
            )

        self._initialize_weights()

    def forward(self, x):
        enc_feats = self.encoder(x)  # 4 encoder outputs: [24, 40, 80, 160] channels

        # ---------------- Task Attention ----------------
        # Use top 3 encoder features for attention (40, 80, 160 channels)
        top_enc_feats = enc_feats[1:]  # enc_feats[1], enc_feats[2], enc_feats[3]
        seg_feats, depth_feats = [], []
        for enc_feat, task_attn in zip(top_enc_feats, self.task_attention_modules):
            t_feats = task_attn(enc_feat)
            seg_feats.append(t_feats[0])
            depth_feats.append(t_feats[1])

        # Reverse to decode from high-level -> low-level
        seg_feats = list(reversed(seg_feats))
        depth_feats = list(reversed(depth_feats))

        # ---------------- Decoder ----------------
        # Segmentation
        seg_x = seg_feats[0]
        seg_x = self.seg_decoders[0](seg_x, seg_feats[1])
        seg_x = self.seg_decoders[1](seg_x, seg_feats[2])
        seg_x = self.seg_decoders[2](seg_x, enc_feats[0])  # skip connection with first encoder feature

        # Depth
        depth_x = depth_feats[0]
        depth_x = self.depth_decoders[0](depth_x, depth_feats[1])
        depth_x = self.depth_decoders[1](depth_x, depth_feats[2])
        depth_x = self.depth_decoders[2](depth_x, enc_feats[0])  # skip connection with first encoder feature

        # ---------------- Heads ----------------
        seg_out = self.seg_head(seg_x)
        depth_out = self.depth_head(depth_x)
        depth_out = torch.clamp(depth_out + self.depth_offset, min=self.depth_offset, max=1.0)

        outputs = {"segmentation": seg_out, "depth": depth_out}

        # Auxiliary head
        if self.use_aux_loss:
            aux_out = self.aux_head(enc_feats[0])  # Use first encoder feature for aux
            outputs["aux"] = aux_out

        return outputs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # More conservative initialization to prevent NaN/Inf
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m.weight, 'data'):
                    m.weight.data *= 0.1  # Scale down weights
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # Very conservative
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_model_size(self):
        param_size, buffer_size = 0, 0
        for p in self.parameters(): param_size += p.nelement() * p.element_size()
        for b in self.buffers(): buffer_size += b.nelement() * b.element_size()
        return (param_size + buffer_size) / 1024 / 1024

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

def create_mtan_model(config):
    model = MTANNetwork(config)
    total, trainable = model.count_parameters()
    print(f"MTAN Model Stats:\n  Total params: {total:,}\n  Trainable: {trainable:,}\n  Size: {model.get_model_size():.2f} MB")
    return model
