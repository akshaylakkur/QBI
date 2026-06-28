"""
Self-contained implementation of SegResNetV2 for CryoET object detection.
Faithfully reproduces the architecture from the 1st-place Kaggle solution.
"""

from typing import List, Optional, Tuple, Union
import math

import torch
from torch import nn, Tensor


# ─── Helper layers ───────────────────────────────────────────────────────


class Conv3dNormAct(nn.Module):
    """Conv3d + optional norm + activation (matching MONAI's get_conv_layer style)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class ResBlock3d(nn.Module):
    """
    MONAI-style ResBlock for 3D (pre-activation):
        norm1 -> act -> conv1 -> norm2 -> act -> conv2 -> + (skip)
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm: str = "GROUP",
        num_groups: int = 8,
        act: str = "RELU",
        act_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        if act_kwargs is None:
            act_kwargs = {"inplace": True}

        self.norm1 = self._get_norm(norm, spatial_dims, in_channels, num_groups)
        self.act1 = self._get_act(act, act_kwargs)
        self.conv1 = Conv3dNormAct(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = self._get_norm(norm, spatial_dims, in_channels, num_groups)
        self.act2 = self._get_act(act, act_kwargs)
        self.conv2 = Conv3dNormAct(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def _get_norm(norm: str, spatial_dims: int, channels: int, num_groups: int) -> nn.Module:
        if norm.upper() == "GROUP":
            return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        elif norm.upper() == "BATCH":
            return nn.BatchNorm3d(channels)
        elif norm.upper() == "INSTANCE":
            return nn.InstanceNorm3d(channels)
        else:
            raise ValueError(f"Unknown norm: {norm}")

    @staticmethod
    def _get_act(act: str, kwargs: dict) -> nn.Module:
        if act.upper() == "RELU":
            return nn.ReLU(**kwargs)
        elif act.upper() == "SILU":
            return nn.SiLU(**kwargs)
        elif act.upper() == "LEAKYRELU":
            return nn.LeakyReLU(**kwargs)
        else:
            raise ValueError(f"Unknown act: {act}")

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)
        x = x + identity
        return x


# ─── SegResNet Backbone ──────────────────────────────────────────────────


class SegResNetBackbone(nn.Module):
    """
    SegResNet backbone (no VAE).
    Matches the architecture from the Kaggle 1st-place solution.

    Args:
        spatial_dims: 3 for 3D.
        init_filters: Number of output channels for initial convolution.
        in_channels: Number of input channels.
        out_channels: Number of output channels for the final conv.
        blocks_down: Number of ResBlocks per down-sampling level.
        blocks_up: Number of ResBlocks per up-sampling level.
        dropout_prob: Dropout probability (applied after init conv).
        act: Activation type.
        norm: Normalization type.
        num_groups: Number of groups for GroupNorm.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 32,
        in_channels: int = 1,
        out_channels: int = 105,
        blocks_down: Tuple[int, ...] = (1, 2, 2, 4),
        blocks_up: Tuple[int, ...] = (1, 1, 1),
        dropout_prob: Optional[float] = 0.2,
        act: str = "RELU",
        norm: str = "GROUP",
        num_groups: int = 8,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act
        self.norm = norm
        self.num_groups = num_groups

        # Initial convolution
        self.convInit = Conv3dNormAct(in_channels, init_filters, kernel_size=3, stride=1, padding=1)

        # Down layers
        self.down_layers = nn.ModuleList()
        for i, n_blocks in enumerate(blocks_down):
            layer_in_channels = init_filters * (2**i)
            if i > 0:
                # Strided conv for down-sampling
                pre_conv = Conv3dNormAct(
                    layer_in_channels // 2, layer_in_channels, kernel_size=3, stride=2, padding=1
                )
            else:
                pre_conv = nn.Identity()

            blocks = [pre_conv]
            for _ in range(n_blocks):
                blocks.append(
                    ResBlock3d(spatial_dims, layer_in_channels, norm=norm, num_groups=num_groups, act=act)
                )
            self.down_layers.append(nn.Sequential(*blocks))

        # Up layers
        self.up_layers = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = init_filters * (2 ** (n_up - i))
            # Up-sampling: 1x1x1 conv + upscale
            up_sample = nn.Sequential(
                Conv3dNormAct(sample_in_channels, sample_in_channels // 2, kernel_size=1, stride=1, padding=0),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            )
            self.up_samples.append(up_sample)

            # ResBlocks after up-sampling
            up_layer_blocks = []
            for _ in range(blocks_up[i]):
                up_layer_blocks.append(
                    ResBlock3d(spatial_dims, sample_in_channels // 2, norm=norm, num_groups=num_groups, act=act)
                )
            self.up_layers.append(nn.Sequential(*up_layer_blocks))

        # Final conv: norm -> act -> 1x1x1 conv
        act_fn = ResBlock3d._get_act(act, {"inplace": True})
        self.conv_final = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=init_filters),
            act_fn,
            Conv3dNormAct(init_filters, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )

        if dropout_prob is not None and dropout_prob > 0:
            self.dropout = nn.Dropout3d(dropout_prob)
        else:
            self.dropout = None

    def encode(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        x = self.convInit(x)
        if self.dropout is not None:
            x = self.dropout(x)

        down_x = []
        for down in self.down_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def decode(self, x: Tensor, down_x: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
        feature_maps = []
        # down_x after reverse: [stride2, stride4, stride8, stride16]
        # We use down_x[1], down_x[2], down_x[3] (skip stride2, matching original)
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)
            feature_maps.append(x)

        x = self.conv_final(x)
        return x, feature_maps

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        x, down_x = self.encode(x)
        # Reverse: now down_x[0] = stride2, down_x[1] = stride4, down_x[2] = stride8, down_x[3] = stride16
        down_x.reverse()
        x, feature_maps = self.decode(x, down_x)
        return x, feature_maps


# ─── Detection Head ──────────────────────────────────────────────────────


class ObjectDetectionHead(nn.Module):
    """
    Detection head that predicts class logits and offset vectors.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        stride: int,
        intermediate_channels: int = 64,
        offset_intermediate_channels: int = 32,
        use_offset_head: bool = True,
    ):
        super().__init__()
        self.use_offset_head = use_offset_head
        self.stride = stride

        # Classification stem
        self.cls_stem = nn.Sequential(
            nn.Conv3d(in_channels, intermediate_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.InstanceNorm3d(intermediate_channels),
            nn.Conv3d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.InstanceNorm3d(intermediate_channels),
        )

        self.cls_head = nn.Conv3d(intermediate_channels, num_classes, kernel_size=1, padding=0)

        if use_offset_head:
            self.offset_stem = nn.Sequential(
                nn.Conv3d(in_channels, offset_intermediate_channels, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
                nn.InstanceNorm3d(offset_intermediate_channels),
                nn.Conv3d(offset_intermediate_channels, offset_intermediate_channels, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
                nn.InstanceNorm3d(offset_intermediate_channels),
            )

            self.offset_head = nn.Conv3d(offset_intermediate_channels, 3, kernel_size=1, padding=0)
            nn.init.zeros_(self.offset_head.weight)
            nn.init.constant_(self.offset_head.bias, 0)

        # Initialize classification head
        nn.init.zeros_(self.cls_head.weight)
        nn.init.constant_(self.cls_head.bias, -4)

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        logits = self.cls_head(self.cls_stem(features))

        if self.use_offset_head:
            offsets = self.offset_head(self.offset_stem(features)).tanh() * self.stride
        else:
            offsets = torch.zeros_like(logits[:, 0:3, ...])

        return logits, offsets


# ─── Full Detection Model ────────────────────────────────────────────────


class SegResNetForObjectDetectionV2(nn.Module):
    """
    SegResNetV2 for object detection.
    Matches the architecture from the Kaggle 1st-place solution checkpoint.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 105,
        init_filters: int = 32,
        blocks_down: Tuple[int, ...] = (1, 2, 2, 4),
        blocks_up: Tuple[int, ...] = (1, 1, 1),
        dropout_prob: float = 0.2,
        head_dropout_prob: float = 0.0,
        num_classes: int = 6,
        use_stride4: bool = False,
        use_stride2: bool = True,
        use_offset_head: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_stride4 = use_stride4
        self.use_stride2 = use_stride2
        self.use_offset_head = use_offset_head

        self.backbone = SegResNetBackbone(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=init_filters,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            dropout_prob=dropout_prob,
            act="RELU",
            norm="GROUP",
            num_groups=8,
        )

        self.dropout = nn.Dropout3d(head_dropout_prob) if head_dropout_prob > 0 else nn.Identity()

        if use_stride2:
            self.head2 = ObjectDetectionHead(
                in_channels=64,
                num_classes=num_classes,
                stride=2,
                intermediate_channels=48,
                offset_intermediate_channels=16,
                use_offset_head=use_offset_head,
            )

        if use_stride4:
            self.head4 = ObjectDetectionHead(
                in_channels=128,
                num_classes=num_classes,
                stride=4,
                intermediate_channels=64,
                offset_intermediate_channels=32,
                use_offset_head=use_offset_head,
            )

    def forward(
        self, volume: Tensor, is_tracing: bool = False
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Forward pass for inference.

        Returns:
            logits: List of [B, C, D, H, W] tensors (one per stride)
            offsets: List of [B, 3, D, H, W] tensors (one per stride)
        """
        _, feature_maps = self.backbone(volume)
        # feature_maps[-3] = stride 4, feature_maps[-2] = stride 2
        fm4, fm2 = feature_maps[-3], feature_maps[-2]

        logits = []
        offsets = []

        if self.use_stride4:
            l4, o4 = self.head4(self.dropout(fm4))
            logits.append(l4)
            offsets.append(o4)

        if self.use_stride2:
            l2, o2 = self.head2(self.dropout(fm2))
            logits.append(l2)
            offsets.append(o2)

        # Apply sigmoid for inference
        logits = [l.sigmoid() for l in logits]

        return logits, offsets

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> "SegResNetForObjectDetectionV2":
        """
        Load model from a Lightning checkpoint.
        Handles the 'model.' prefix in state dict keys.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["state_dict"]

        # Determine config from state dict
        has_head4 = any("head4" in k for k in state_dict)
        has_head2 = any("head2" in k for k in state_dict)

        # Infer num_classes from head2.cls_head.weight
        num_classes = 6
        for k, v in state_dict.items():
            if "head2.cls_head.weight" in k:
                num_classes = v.shape[0]
                break

        # Infer init_filters from convInit
        init_filters = 32
        for k, v in state_dict.items():
            if "convInit.conv.weight" in k:
                init_filters = v.shape[0]
                break

        # Infer out_channels from conv_final
        out_channels = 105
        for k, v in state_dict.items():
            if "conv_final.2.conv.weight" in k:
                out_channels = v.shape[0]
                break

        model = cls(
            in_channels=1,
            out_channels=out_channels,
            init_filters=init_filters,
            num_classes=num_classes,
            use_stride4=has_head4,
            use_stride2=has_head2,
            use_offset_head=True,
            dropout_prob=0.2,
            head_dropout_prob=0.0,
        )

        # Strip 'model.' prefix from keys
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[6:]] = v
            elif k in ("thresholds", "per_class_scores"):
                continue
            else:
                new_state_dict[k] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"Warning: Missing keys: {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys: {unexpected}")

        model = model.to(device)
        model.eval()
        return model


# ─── Decoding & NMS ──────────────────────────────────────────────────────


def anchors_for_offsets_feature_map(offsets: Tensor, stride: int) -> Tensor:
    """Generate anchor grid for a given feature map."""
    z, y, x = torch.meshgrid(
        torch.arange(offsets.size(-3), device=offsets.device),
        torch.arange(offsets.size(-2), device=offsets.device),
        torch.arange(offsets.size(-1), device=offsets.device),
        indexing="ij",
    )
    anchors = torch.stack([x, y, z], dim=0)
    anchors = anchors.float().add_(0.5).mul_(stride)
    anchors = anchors[None, ...].repeat(offsets.size(0), 1, 1, 1, 1)
    return anchors


def decode_detections(
    logits: Union[Tensor, List[Tensor]],
    offsets: Union[Tensor, List[Tensor]],
    strides: Union[int, List[int]],
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Decode detections from logits and offsets.

    Returns:
        logits_flat: [B, N, C]
        centers_flat: [B, N, 3]
        anchors_flat: [B, N, 3]
    """
    if torch.is_tensor(logits):
        logits = [logits]
    if torch.is_tensor(offsets):
        offsets = [offsets]
    if isinstance(strides, int):
        strides = [strides]

    anchors = [anchors_for_offsets_feature_map(off, s) for off, s in zip(offsets, strides)]

    logits_flat_list = []
    centers_flat_list = []
    anchors_flat_list = []

    for logit, offset, anchor in zip(logits, offsets, anchors):
        centers = anchor + offset
        B, C, D, H, W = logit.shape
        logits_flat_list.append(logit.reshape(B, C, -1).transpose(1, 2))  # [B, D*H*W, C]
        centers_flat_list.append(centers.reshape(B, 3, -1).transpose(1, 2))  # [B, D*H*W, 3]
        anchors_flat_list.append(anchor.reshape(B, 3, -1).transpose(1, 2))  # [B, D*H*W, 3]

    logits_flat = torch.cat(logits_flat_list, dim=1)
    centers_flat = torch.cat(centers_flat_list, dim=1)
    anchors_flat = torch.cat(anchors_flat_list, dim=1)

    return logits_flat, centers_flat, anchors_flat


def keypoint_similarity(pts1: Tensor, pts2: Tensor, sigmas: Tensor) -> Tensor:
    """Compute OKS (Object Keypoint Similarity) between two sets of keypoints."""
    d = ((pts1 - pts2) ** 2).sum(dim=-1, keepdim=False)
    e = d / (2 * sigmas**2)
    return torch.exp(-e)


def gaussian_blur_3d(x: Tensor, kernel_size: int, sigma: float) -> Tensor:
    """3D Gaussian blur."""
    kd = kh = kw = kernel_size
    z = torch.linspace(-(kd // 2), kd // 2, steps=kd)
    y = torch.linspace(-(kh // 2), kh // 2, steps=kh)
    x_ = torch.linspace(-(kw // 2), kw // 2, steps=kw)
    zz, yy, xx = torch.meshgrid(z, y, x_, indexing="ij")
    kernel_3d = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    kernel_3d = kernel_3d / kernel_3d.sum()
    kernel_3d = kernel_3d.to(x.device).to(x.dtype)

    C = x.shape[1]
    kernel_3d = kernel_3d.view(1, 1, *kernel_3d.shape).repeat(C, 1, 1, 1, 1)
    return torch.nn.functional.conv3d(x, weight=kernel_3d, padding=kernel_size // 2, groups=C)


def centernet_heatmap_nms(scores: Tensor, kernel: Union[int, Tuple[int, int, int]] = 3) -> Tensor:
    """CenterNet-style NMS: keep only local maxima."""
    if isinstance(kernel, int):
        kernel = (kernel, kernel, kernel)
    pad = (kernel[0] - 1) // 2, (kernel[1] - 1) // 2, (kernel[2] - 1) // 2
    maxpool = torch.nn.functional.max_pool3d(scores, kernel_size=kernel, padding=pad, stride=1)
    mask = scores == maxpool
    return scores * mask


@torch.no_grad()
def decode_detections_with_nms(
    scores: List[Tensor],
    offsets: List[Tensor],
    strides: List[int],
    min_score: Union[float, List[float]],
    class_sigmas: List[float],
    iou_threshold: float = 0.25,
    use_single_label_per_anchor: bool = True,
    use_centernet_nms: bool = False,
    pre_nms_top_k: Optional[int] = None,
    class_map_gaussian_smoothing_kernel: int = 0,
    centernet_nms_kernel: Union[int, Tuple[int, int, int]] = 3,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Decode detections with per-class NMS.

    Returns:
        final_centers: [N, 3] (x, y, z) in voxel coordinates
        final_labels: [N]
        final_scores: [N]
    """
    num_classes = scores[0].shape[0]

    # Normalize min_score
    min_score_arr = torch.as_tensor(min_score, dtype=torch.float32)
    if min_score_arr.numel() == 1:
        min_score_arr = min_score_arr.expand(num_classes)

    # Optional Gaussian smoothing
    if class_map_gaussian_smoothing_kernel > 0:
        scores = [
            gaussian_blur_3d(s.unsqueeze(0), kernel_size=class_map_gaussian_smoothing_kernel, sigma=1.0).squeeze(0)
            for s in scores
        ]

    # Optional CenterNet NMS
    if use_centernet_nms:
        scores = [centernet_heatmap_nms(s.unsqueeze(0), kernel=centernet_nms_kernel).squeeze(0) for s in scores]

    # Decode all detections
    scores_flat, centers_flat, _ = decode_detections(
        [s.unsqueeze(0) for s in scores],
        [o.unsqueeze(0) for o in offsets],
        strides,
    )
    scores_flat = scores_flat.squeeze(0)  # [N, C]
    centers_flat = centers_flat.squeeze(0)  # [N, 3]

    labels_of_max_score = scores_flat.argmax(dim=1)

    final_labels_list = []
    final_scores_list = []
    final_centers_list = []

    for class_index in range(num_classes):
        sigma_value = float(class_sigmas[class_index])
        score_threshold = float(min_score_arr[class_index])

        score_mask = scores_flat[:, class_index] >= score_threshold

        if use_single_label_per_anchor:
            class_mask = labels_of_max_score.eq(class_index)
            mask = class_mask & score_mask
        else:
            mask = score_mask

        if not mask.any():
            continue

        class_scores = scores_flat[mask, class_index]
        class_centers = centers_flat[mask]

        if pre_nms_top_k is not None and len(class_scores) > pre_nms_top_k:
            class_scores, sort_idx = torch.topk(class_scores, pre_nms_top_k, largest=True)
            class_centers = class_centers[sort_idx]
        else:
            class_scores, sort_idx = class_scores.sort(descending=True)
            class_centers = class_centers[sort_idx]

        # Greedy NMS
        keep_indices = []
        suppressed = torch.zeros_like(class_scores, dtype=torch.bool)

        for i in range(class_scores.size(0)):
            if suppressed[i]:
                continue
            keep_indices.append(i)
            iou = keypoint_similarity(
                class_centers[i : i + 1, :], class_centers, torch.tensor([sigma_value], device=class_centers.device)
            )
            suppressed |= iou.squeeze(0) > iou_threshold

        keep_indices = torch.tensor(keep_indices, dtype=torch.long, device=class_scores.device)
        final_labels_list.append(torch.full((keep_indices.numel(),), class_index, dtype=torch.long, device=class_scores.device))
        final_scores_list.append(class_scores[keep_indices])
        final_centers_list.append(class_centers[keep_indices])

    if final_labels_list:
        final_labels = torch.cat(final_labels_list, dim=0)
        final_scores = torch.cat(final_scores_list, dim=0)
        final_centers = torch.cat(final_centers_list, dim=0)
    else:
        final_labels = torch.empty((0,), dtype=torch.long, device=scores_flat.device)
        final_scores = torch.empty((0,), device=scores_flat.device)
        final_centers = torch.empty((0, 3), device=scores_flat.device)

    return final_centers, final_labels, final_scores
