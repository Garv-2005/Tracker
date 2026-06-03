import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision.models import alexnet, AlexNet_Weights, resnet50, ResNet50_Weights

# Reuse compiled backbones across tracker re-inits (e.g. after re-detection).
_FEATURE_EXTRACTOR_CACHE: dict[tuple[str, str], nn.Module] = {}


def _cache_key(model_name: str, device: torch.device) -> tuple[str, str]:
    return (model_name.lower(), str(device))


def crop_and_pad(img, center, crop_sz, out_sz, pad_val=114):
    """
    Crops a square patch centered at `center` with side `crop_sz`,
    pads with `pad_val` if outside image bounds, and resizes to `out_sz`.
    """
    h, w = img.shape[:2]
    cx, cy = center

    # Half crop size
    half_sz = crop_sz / 2.0
    x1 = int(round(cx - half_sz))
    y1 = int(round(cy - half_sz))
    x2 = x1 + int(round(crop_sz))
    y2 = y1 + int(round(crop_sz))

    # Pad margins
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    # Valid crop coords
    x1_in = max(0, x1)
    y1_in = max(0, y1)
    x2_in = min(w, x2)
    y2_in = min(h, y2)

    patch = img[y1_in:y2_in, x1_in:x2_in]

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        patch = cv2.copyMakeBorder(
            patch, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(pad_val, pad_val, pad_val)
        )

    if patch.shape[0] != out_sz or patch.shape[1] != out_sz:
        patch = cv2.resize(patch, (out_sz, out_sz))

    return patch

class PyTorchSiameseTracker:
    def __init__(self, model_name="dasiamrpn", device="cpu"):
        self.device = torch.device(device)
        self.model_name = model_name.lower()
        cache_key = _cache_key(self.model_name, self.device)

        cached = _FEATURE_EXTRACTOR_CACHE.get(cache_key)
        if cached is not None:
            self.feature_extractor = cached
        else:
            self.feature_extractor = self._build_feature_extractor()
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.feature_extractor = self._maybe_compile(self.feature_extractor)
            _FEATURE_EXTRACTOR_CACHE[cache_key] = self.feature_extractor

        self.z_feat = None
        self.target_sz = None
        self.center_pos = None

        if self.model_name in ("ostrack", "mixformer"):
            self.exemplar_size = 128
            self.instance_size = 256
            self.stride = 16.0
        else:
            self.exemplar_size = 127
            self.instance_size = 255
            self.stride = 16.0

        self.context_amount = 0.5
        self.scales = [0.95, 1.0, 1.05]
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _build_feature_extractor(self) -> nn.Module:
        if self.model_name in ("dasiamrpn", "dasiamrpm"):
            weights = AlexNet_Weights.DEFAULT
            backbone = alexnet(weights=weights).features.eval().to(self.device)
            return backbone[:12]
        if self.model_name == "siamrpnpp":
            weights = ResNet50_Weights.DEFAULT
            resnet = resnet50(weights=weights).eval().to(self.device)
            return nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
            )
        if self.model_name in ("ostrack", "mixformer"):
            from torchvision.models import vit_b_16, ViT_B_16_Weights

            print(
                f"[PyTorchTracker] Loading ViT-B-16 weights for {self.model_name.upper()}..."
            )
            return vit_b_16(weights=ViT_B_16_Weights.DEFAULT).eval().to(self.device)
        raise ValueError(f"Unknown PyTorch model name: {self.model_name}")

    def _maybe_compile(self, model: nn.Module) -> nn.Module:
        if not (hasattr(torch, "compile") and self.device.type == "cuda"):
            return model
        try:
            print(f"[PyTorchTracker] Compiling feature extractor for {self.device.type}...")
            compiled_model = torch.compile(model)
            with torch.no_grad():
                dummy_in = torch.randn(1, 3, 224, 224, device=self.device)
                if self.model_name in ("ostrack", "mixformer"):
                    _ = self.extract_vit_features(dummy_in, 128)
                else:
                    _ = compiled_model(dummy_in)
            print("[PyTorchTracker] Model compiled successfully.")
            return compiled_model
        except Exception as e:
            print(f"[PyTorchTracker] torch.compile failed (falling back to uncompiled): {e}")
            return model

    def _preprocess(self, patch):
        # Convert BGR to RGB, scale to [0,1], normalize, and convert to Tensor
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        patch_norm = (patch_rgb - self.mean) / self.std
        tensor = torch.from_numpy(patch_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    def extract_vit_features(self, tensor, size):
        # Resize input to multiple of 16 (vit_b_16 patch size)
        target_size = (size // 16) * 16
        if tensor.shape[2] != target_size or tensor.shape[3] != target_size:
            tensor = F.interpolate(tensor, size=(target_size, target_size), mode='bilinear', align_corners=False)
            
        n, c, h, w = tensor.shape
        p = self.feature_extractor.patch_size
        h_feat, w_feat = h // p, w // p
        
        # Bypass the strict 224x224 assertion by calling conv_proj directly
        x = self.feature_extractor.conv_proj(tensor)  # (n, hidden_dim, h_feat, w_feat)
        x = x.flatten(2).transpose(1, 2)  # (n, h_feat * w_feat, hidden_dim)
        
        batch_class_token = self.feature_extractor.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # Interpolate position embedding if size differs from 224x224 (197 tokens)
        pos_embed = self.feature_extractor.encoder.pos_embedding
        if pos_embed.shape[1] != x.shape[1]:
            cls_pos = pos_embed[:, :1, :]
            spatial_pos = pos_embed[:, 1:, :]
            dim = spatial_pos.shape[2]
            spatial_pos = spatial_pos.transpose(1, 2).reshape(1, dim, 14, 14)
            spatial_pos = F.interpolate(spatial_pos, size=(h_feat, w_feat), mode='bicubic', align_corners=False)
            spatial_pos = spatial_pos.reshape(1, dim, -1).transpose(1, 2)
            pos_embed = torch.cat([cls_pos, spatial_pos], dim=1)
            
        x = x + pos_embed
        x = self.feature_extractor.encoder.ln(self.feature_extractor.encoder.layers(x))
        
        # Reshape spatial tokens back to grid
        spatial_tokens = x[:, 1:, :]
        feat_map = spatial_tokens.transpose(1, 2).reshape(n, -1, h_feat, w_feat)
        return feat_map

    def extract_ostrack_features(self, z_tensor, x_batch):
        hz, wz = 8, 8
        hx, wx = 16, 16
        
        if z_tensor.shape[2] != 128 or z_tensor.shape[3] != 128:
            z_tensor = F.interpolate(z_tensor, size=(128, 128), mode='bilinear', align_corners=False)
        if x_batch.shape[2] != 256 or x_batch.shape[3] != 256:
            x_batch = F.interpolate(x_batch, size=(256, 256), mode='bilinear', align_corners=False)
            
        # Patch projection
        z_proj = self.feature_extractor.conv_proj(z_tensor) # (1, 768, 8, 8)
        x_proj = self.feature_extractor.conv_proj(x_batch) # (3, 768, 16, 16)
        
        z_tokens = z_proj.flatten(2).transpose(1, 2) # (1, 64, 768)
        x_tokens = x_proj.flatten(2).transpose(1, 2) # (3, 256, 768)
        
        n = x_tokens.shape[0]
        z_tokens = z_tokens.expand(n, -1, -1) # (3, 64, 768)
        
        # Load and split position embedding
        pos_embed = self.feature_extractor.encoder.pos_embedding
        cls_pos = pos_embed[:, :1, :]
        spatial_pos = pos_embed[:, 1:, :] # (1, 196, 768)
        dim = spatial_pos.shape[2]
        spatial_pos_grid = spatial_pos.transpose(1, 2).reshape(1, dim, 14, 14)
        
        # Interpolate position embedding for z (8x8)
        z_pos = F.interpolate(spatial_pos_grid, size=(hz, wz), mode='bicubic', align_corners=False)
        z_pos = z_pos.reshape(1, dim, -1).transpose(1, 2).expand(n, -1, -1)
        z_tokens = z_tokens + z_pos
        
        # Interpolate position embedding for x (16x16)
        x_pos = F.interpolate(spatial_pos_grid, size=(hx, wx), mode='bicubic', align_corners=False)
        x_pos = x_pos.reshape(1, dim, -1).transpose(1, 2).expand(n, -1, -1)
        x_tokens = x_tokens + x_pos
        
        # Concatenate template and search tokens
        joint_tokens = torch.cat([z_tokens, x_tokens], dim=1) # (3, 320, 768)
        
        # Add class token
        batch_class_token = self.feature_extractor.class_token.expand(n, -1, -1)
        joint_tokens = torch.cat([batch_class_token, joint_tokens], dim=1) # (3, 321, 768)
        
        # Class token pos embedding addition
        joint_tokens = joint_tokens + torch.cat([cls_pos.expand(n, -1, -1), torch.zeros(n, 320, dim, device=self.device)], dim=1)
        
        # Run transformer blocks
        joint_tokens = self.feature_extractor.encoder.ln(self.feature_extractor.encoder.layers(joint_tokens))
        
        # Extract search features (last 256 tokens)
        search_feats = joint_tokens[:, 1 + hz*wz:, :] # (3, 256, 768)
        feat_map = search_feats.transpose(1, 2).reshape(n, -1, hx, wx)
        
        # Extract template features (tokens 1 to 1+64)
        temp_feats = joint_tokens[:, 1 : 1 + hz*wz, :] # (3, 64, 768)
        # Take the first element of batch (scale 0) for template features
        z_feat = temp_feats[0:1].transpose(1, 2).reshape(1, -1, hz, wz)
        
        return feat_map, z_feat

    def init(self, frame, bbox):
        x, y, w, h = bbox
        self.center_pos = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float32)
        self.target_sz = np.array([w, h], dtype=np.float32)

        # Compute exemplar patch size with context
        context_sz = self.context_amount * sum(self.target_sz)
        s_z = np.sqrt((self.target_sz[0] + context_sz) * (self.target_sz[1] + context_sz))

        z_crop = crop_and_pad(frame, self.center_pos, s_z, self.exemplar_size)
        z_tensor = self._preprocess(z_crop)

        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                if self.model_name == "ostrack":
                    # Store raw preprocessed z_tensor to perform joint projection later
                    self.z_feat = z_tensor
                elif self.model_name == "mixformer":
                    self.z_feat = self.extract_vit_features(z_tensor, self.exemplar_size)
                    self.z_feat = F.normalize(self.z_feat, p=2, dim=1)
                else:
                    self.z_feat = self.feature_extractor(z_tensor)
                    # Normalize along channel dimension for cosine similarity cross-correlation
                    self.z_feat = F.normalize(self.z_feat, p=2, dim=1)

    def update(self, frame, predicted_center=None):
        if self.z_feat is None:
            raise RuntimeError("Tracker not initialised. Call init() first.")

        # Compute baseline search patch size
        context_sz = self.context_amount * sum(self.target_sz)
        s_z = np.sqrt((self.target_sz[0] + context_sz) * (self.target_sz[1] + context_sz))
        s_x = s_z * (self.instance_size / self.exemplar_size)

        crop_center = np.array(predicted_center, dtype=np.float32) if predicted_center is not None else self.center_pos

        # Crop search patches at multiple scales
        x_crops = []
        for scale in self.scales:
            crop_sz = s_x * scale
            x_crop = crop_and_pad(frame, crop_center, crop_sz, self.instance_size)
            x_crops.append(self._preprocess(x_crop))

        # Batch forward pass
        x_batch = torch.cat(x_crops, dim=0)
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                if self.model_name == "ostrack":
                    x_feats, z_feat_dyn = self.extract_ostrack_features(self.z_feat, x_batch)
                    x_feats = F.normalize(x_feats, p=2, dim=1)
                    z_feat_dyn = F.normalize(z_feat_dyn, p=2, dim=1)
                    response = F.conv2d(x_feats, z_feat_dyn)
                elif self.model_name == "mixformer":
                    x_feats = self.extract_vit_features(x_batch, self.instance_size)
                    x_feats = F.normalize(x_feats, p=2, dim=1)
                    response = F.conv2d(x_feats, self.z_feat)
                else:
                    x_feats = self.feature_extractor(x_batch)
                    x_feats = F.normalize(x_feats, p=2, dim=1)
                    # Cosine similarity cross-correlation
                    # x_feats: (3, C, Hx, Wx), self.z_feat: (1, C, Hz, Wz)
                    response = F.conv2d(x_feats, self.z_feat)  # Output shape: (3, 1, H_out, W_out)
            response = response.squeeze(1).cpu().numpy()  # Shape: (3, H_out, W_out)

        # Apply Hanning window to penalize large displacements
        _, r_h, r_w = response.shape
        window = np.outer(np.hanning(r_h), np.hanning(r_w))

        # Normalize response maps jointly across all scales to preserve relative match strengths
        r_min, r_max = response.min(), response.max()
        if r_max > r_min:
            normalized_response = (response - r_min) / (r_max - r_min)
        else:
            normalized_response = response

        penalty_weight = 0.3
        scale_penalty = 0.975  # Penalize scale changes to prevent runaway box shrinking/growing
        penalized_response = []
        for i in range(len(self.scales)):
            resp = normalized_response[i]
            # Combine response with window penalty
            resp = (1.0 - penalty_weight) * resp + penalty_weight * window
            # Apply scale penalty for scale changes
            if self.scales[i] != 1.0:
                resp = resp * scale_penalty
            penalized_response.append(resp)
        penalized_response = np.stack(penalized_response, axis=0)

        # Find maximum response across scales and positions
        scale_idx, max_r, max_c = np.unravel_index(
            np.argmax(penalized_response), penalized_response.shape
        )

        # Calculate displacement from the center of the response map
        disp_feat = np.array([max_c - (r_w - 1) / 2.0, max_r - (r_h - 1) / 2.0], dtype=np.float32)
        disp_crop = disp_feat * self.stride

        # Map crop displacement to image coordinates
        best_scale = self.scales[scale_idx]
        best_crop_sz = s_x * best_scale
        disp_img = disp_crop * (best_crop_sz / self.instance_size)

        # Update state
        self.center_pos = crop_center + disp_img
        self.target_sz *= best_scale

        # Keep size reasonable
        fh, fw = frame.shape[:2]
        self.target_sz[0] = max(10, min(self.target_sz[0], fw))
        self.target_sz[1] = max(10, min(self.target_sz[1], fh))

        # Compute bounding box
        x = int(round(self.center_pos[0] - self.target_sz[0] / 2.0))
        y = int(round(self.center_pos[1] - self.target_sz[1] / 2.0))
        w = int(round(self.target_sz[0]))
        h = int(round(self.target_sz[1]))

        # Clamp box to image boundaries
        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = max(1, min(w, fw - x))
        h = max(1, min(h, fh - y))

        max_score = float(response[scale_idx, max_r, max_c])
        return (x, y, w, h), max_score
