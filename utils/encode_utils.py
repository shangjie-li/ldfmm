import numpy as np


def angle_to_bin(angle):
    """

    Args:
        angle: float, angle in rad

    Returns:
        bin_id: int, bin id (0 to 11)
        residual_angle: float, angle in rad

    """
    angle_per_bin = 2 * np.pi / 12
    shifted_angle = (angle + angle_per_bin / 2) % (2 * np.pi)
    bin_id = int(shifted_angle / angle_per_bin)
    residual_angle = shifted_angle - (bin_id * angle_per_bin + angle_per_bin / 2)

    return bin_id, residual_angle


def draw_umich_gaussian(heatmap, center, radius, k=1.0):
    """

    Args:
        heatmap: ndarray of float, [C, H, W], heatmap (0 to 1)
        center: ndarray, [2], (u, v)
        radius: int, pixel length
        k: float

    Returns:
        heatmap: ndarray of float, [C, H, W], heatmap (0 to 1)

    """
    def gaussian2d(shape, sigma=1.0):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    radius = int(radius)
    diameter = 2 * radius + 1
    gaussian = gaussian2d((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap
