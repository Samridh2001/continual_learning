import torch
import torchvision.transforms as T

class MultiViewAugmenter:
    """
    In-memory multi-view perturbation engine for activation covariance probing.
    Operates directly on normalized GPU batch tensors to avoid disk I/O.
    """
    def __init__(self, device='cuda'):
        self.device = device
        # ImageNet normalization constants
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        # Perturbation 1: Appearance / Color perturbation (Color Jitter)
        self.color_jitter = T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)

        # Perturbation 2: High-frequency texture perturbation (Gaussian Blur)
        self.gaussian_blur = T.GaussianBlur(kernel_size=(7, 7), sigma=(2.0, 3.5))

    def denormalize(self, x):
        """Reverts normalized tensor (B, 3, H, W) back to [0, 1] range."""
        return torch.clamp(x * self.std + self.mean, 0.0, 1.0)

    def normalize(self, x):
        """Applies ImageNet normalization to [0, 1] tensor."""
        return (x - self.mean) / self.std

    def get_color_view(self, x):
        """
        Strips dominant chromatic shortcuts by converting to 3-channel grayscale
        and applying strong color jitter.
        """
        img = self.denormalize(x)
        # Luminance grayscale conversion
        gray = 0.2989 * img[:, 0:1] + 0.5870 * img[:, 1:2] + 0.1140 * img[:, 2:3]
        gray_3ch = gray.repeat(1, 3, 1, 1)
        jittered = self.color_jitter(gray_3ch)
        return self.normalize(jittered)

    def get_blur_view(self, x):
        """
        Strips high-frequency pixel textures and borders via Gaussian smoothing.
        """
        img = self.denormalize(x)
        blurred = self.gaussian_blur(img)
        return self.normalize(blurred)
