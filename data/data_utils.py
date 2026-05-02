from typing import Tuple, Union
import numpy as np
import cv2
import torch

def discretize(x: Union[np.ndarray, float], num_discrete: int, discrete_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    lo, hi = discrete_range
    x = (x - lo) / (hi - lo)
    x = x * (num_discrete - 1)
    out = np.round(x)
    x = out / (num_discrete - 1)
    x = x * (hi - lo) + lo
    return x, out

DISCRETIZE_VALUE = [12, 4, 5] # angle, scale, translation
# DISCRETIZE_VALUE = [360, 40, 20] # angle, scale, translation

# transform: angle, scale, translation are normalized to [-1, 1]
def generate_rand_transform(image, mode=0):
    # angle: 0-360
    if mode == 0 or mode == 1:
        real_angle = np.random.rand() * 360
    else:
        real_angle = 0.0
    real_angle, encode_angle = discretize(real_angle, DISCRETIZE_VALUE[0] + 1, (0, 360))
    encode_angle = encode_angle - DISCRETIZE_VALUE[0] if encode_angle > DISCRETIZE_VALUE[0] / 2.0 else encode_angle
    encode_angle = encode_angle / (DISCRETIZE_VALUE[0] / 2.0) * np.pi
    real_angle_str = f'{real_angle} degrees'
    # scale: 1-5 & 1/(1-5)
    if mode == 0 or mode == 2:
        scale = np.random.rand(2) * 4 + 1
        real_scale, encode_scale = discretize(scale, DISCRETIZE_VALUE[1] + 1, (1, 5))
        if np.random.rand() < 0.5:
            real_scale_str = f'1/{real_scale[0]} along x-axis and 1/{real_scale[1]} along y-axis'
            real_scale = 1 / real_scale
            encode_scale = -encode_scale
        else:
            real_scale_str = f'{real_scale[0]} along x-axis and {real_scale[1]} along y-axis'
            encode_scale = encode_scale
    else:
        real_scale_str = f'1.0 along x-axis and 1.0 along y-axis'
        real_scale = np.ones(2)
        encode_scale = np.zeros(2)
    encode_scale = encode_scale / DISCRETIZE_VALUE[1] * 3

    # translation: 0-1
    if mode == 0 or mode == 3:
        translation = np.random.rand(2)
        real_translation, encode_translation = discretize(translation, DISCRETIZE_VALUE[2] + 1, (0, 1))
        real_translation_str = f'{real_translation[0].round(1)} units along x-axis and {real_translation[1].round(1)} units along y-axis'
    else:
        real_translation_str = f'0.0 units along x-axis and 0.0 units along y-axis'
        real_translation = np.zeros(2)
        encode_translation = np.zeros(2)
    encode_translation[encode_translation > DISCRETIZE_VALUE[2] / 2] -= DISCRETIZE_VALUE[2]
    encode_translation = encode_translation / (DISCRETIZE_VALUE[2] / 2.0) * np.pi
    images, coords = transform_by_matrix(image, real_angle, real_scale, real_translation)
    transformed_img = images[1]
    original_coord, transformed_coord = coords
    transform_info = {
        'transform': {
            'angle': real_angle,
            'scale': real_scale,
            'translation': real_translation
        },
        'transform_str': {
            'angle': real_angle_str,
            'scale': real_scale_str,
            'translation': real_translation_str,
        },
        'encode_transform': {
            'angle': encode_angle, # -pi ~ pi represent angle
            'scale': encode_scale, # -3 ~ 3 represent [1/5, 1] & [1, 5]
            'translation': encode_translation # -pi ~ pi represent [0.5, 1] & [0, 0.5]
        },
        'original_coord': original_coord,
        'transformed_coord': transformed_coord,
    }
    return transformed_img, transform_info

def transform_by_matrix(image, angle, scale, translation):

    def get_matrix(angle, scale, translation, for_img=False):
        original_h, original_w = image.shape[:2]
        # for cv2 test, need to repeat image incase black boarder
        # if for_img:
        #     pad_x, pad_y = 9, 9
        # else:
        #     pad_x, pad_y = 1, 1
        pad_x, pad_y = 1, 1
        scale_x, scale_y = scale
        offset_x, offset_y = translation
        offset_x = offset_x * original_w
        offset_y = offset_y * original_h

        h = original_h * pad_y
        w = original_w * pad_x

        center_x, center_y = w // 2, h // 2
            
        T0 = np.array([
            [1, 0, center_x],
            [0, 1, center_y],
            [0, 0, 1]
        ], dtype=np.float32)

        angle_rad = -angle / 180.0 * np.pi  # Convert to radians, in cv2, angle is negative clockwise
        alpha = np.cos(angle_rad)
        beta = np.sin(angle_rad)

        T = np.array([
            [alpha / scale_x, beta / scale_y, (offset_x * alpha / scale_x + offset_y * beta / scale_y)],
            [-beta / scale_x, alpha / scale_y, (-offset_x * beta  / scale_x + offset_y * alpha / scale_y)],
            [0, 0, 1]
        ], dtype=np.float32)

        T0_INV = np.array([
            [1, 0, -center_x],
            [0, 1, -center_y],
            [0, 0, 1]
        ], dtype=np.float32)

        m = T0 @ T @ T0_INV # first move to center, then transform, finally move back

        if for_img:
            transformed_image = np.tile(image, (pad_y, pad_x, 1))
            transformed_image = cv2.warpAffine(transformed_image, m[:2, :], (w, h), borderValue=(255,255,255))
            transformed_image = transformed_image[
                center_y - original_h // 2:center_y + original_h // 2,
                center_x - original_w // 2:center_x + original_w // 2
            ] # crop to original size
            return image, transformed_image
        else:
            c = np.array([0, 0, -1])
            original_coord = np.meshgrid(np.arange(original_w) / original_w, np.arange(original_h) / original_h)
            original_coord = np.stack(original_coord, axis=-1)
            original_coord = np.concatenate([original_coord, np.zeros((original_h, original_w, 1), dtype=np.float32)], axis=-1)
            # A_ori = original_coord * 2.0 - 1.0
            # B_ori = np.cross(c, A_ori)
            # C_ori = np.broadcast_to(c, A_ori.shape)

            transformed_coord = np.tile(original_coord, (pad_y, pad_x, 1))
            transformed_coord = cv2.warpAffine(transformed_coord, m[:2, :], (w, h), borderValue=(0.5,0.5,1))
            transformed_coord = transformed_coord[
                center_y - original_h // 2:center_y + original_h // 2,
                center_x - original_w // 2:center_x + original_w // 2
            ] # crop to original size
            # A_trans = transformed_coord * 2.0 - 1.0
            # B_trans = np.cross(c, A_trans)
            # C_trans = np.broadcast_to(c, A_trans.shape)

            # original_coord = np.concatenate([B_ori, C_ori], axis=-1)
            # transformed_coord = np.concatenate([B_trans, C_trans], axis=-1)

            original_coord = original_coord * 2.0 - 1.0
            transformed_coord = transformed_coord * 2.0 - 1.0
            return original_coord, transformed_coord

    images = get_matrix(angle, scale, translation, for_img=True)
    coords = get_matrix(angle, scale, translation, for_img=False)

    return images, coords

def rotate_image(image, angle):
    # rotate image from center
    original_h, original_w = image.shape[:2]
    pad_image = scale_image(image, (2, 2), resize=False)
    (h, w) = pad_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(pad_image, M, (w, h))
    return rotated_image[h//2-original_h//2:h//2+original_h//2, w//2-original_w//2:w//2+original_w//2]

def scale_image(image, scale, resize=True):
    original_h, original_w = image.shape[:2]
    scale_x, scale_y = scale
    pad_x = int(np.ceil(scale_x))
    pad_y = int(np.ceil(scale_y))
    tile_image = np.tile(image, (pad_y, pad_x, 1))
    if pad_x % 2 == 0:
        move_x = int(original_w / 2)
    else:
        move_x = original_w
    if pad_y % 2 == 0:
        move_y = int(original_h / 2)
    else:
        move_y = original_h
    image = np.zeros(tile_image.shape)
    image[:move_y, :move_x] = tile_image[-move_y:, -move_x:]
    image[move_y:, :move_x] = tile_image[:-move_y, -move_x:]
    image[:move_y, move_x:] = tile_image[-move_y:, :-move_x]
    image[move_y:, move_x:] = tile_image[:-move_y, :-move_x]

    h, w = image.shape[:2]
    new_region_x_min = int(w // 2 - original_w * scale_x // 2)
    new_region_y_min = int(h // 2 - original_h * scale_y // 2)
    new_region_x_max = int(new_region_x_min + original_w * scale_x)
    new_region_y_max = int(new_region_y_min + original_h * scale_y)
    image = image[new_region_y_min:new_region_y_max, new_region_x_min:new_region_x_max]
    if resize:
        image = cv2.resize(image, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
    return image

def translate_image(image, translation):
    # translate image from center
    (h, w) = image.shape[:2]
    translation_x, translation_y = translation
    image = np.roll(image, int(translation_x * w), axis=1)
    image = np.roll(image, int(translation_y * h), axis=0)
    return image

def get_bbox_from_mask(mask_tensor):
    """
    输入: mask_tensor, shape [1, 1024, 1024]
    输出: [h_min, h_max, w_min, w_max] (闭区间，包含边界索引)
    """
    # 1. 去掉 batch 维度，变成 [1024, 1024]
    # 如果不想 squeeze，也可以在 nonzero 后处理，但这样更直观
    _, h, w = mask_tensor.shape
    mask_2d = mask_tensor.squeeze(0) 
    
    # 2. 找到所有值为 1 的元素坐标
    # nonzero 返回 shape [N, 2]，每一行是 (h_idx, w_idx)
    nonzero_indices = torch.nonzero(mask_2d == 1)
    
    # 3. 异常处理：如果图中没有 1
    if nonzero_indices.shape[0] == 0:
        return None # 或者返回 [-1, -1, -1, -1]
    
    # 4. 分离 H 和 W 坐标
    h_indices = nonzero_indices[:, 0]
    w_indices = nonzero_indices[:, 1]
    
    # 5. 计算 min 和 max
    h_min = (h_indices.min().item() + 1) / h
    h_max = (h_indices.max().item()) / h
    w_min = (w_indices.min().item() + 1) / w
    w_max = (w_indices.max().item()) / w

    # points = [
    #     [h_min, w_min],
    #     [h_min, w_max],
    #     [h_max, w_min],
    #     [h_max, w_max],
    # ]
    points = [
        [h_min + (h_max - h_min) / 4, w_min + (w_max - w_min) / 4],
        [h_min + (h_max - h_min) / 4, w_max - (w_max - w_min) / 4],
        [h_max - (h_max - h_min) / 4, w_min + (w_max - w_min) / 4],
        [h_max - (h_max - h_min) / 4, w_max - (w_max - w_min) / 4],
    ]
    return points

def focus_area(mask):
    mask_img = mask.to('cpu').numpy()
    mask_img = mask_img.transpose(1, 2, 0)[:, :, 0]
    h, w = mask_img.shape
    ys, xs = np.where(mask_img > 0)
    y_min, y_max = np.min(ys), np.max(ys)
    x_min, x_max = np.min(xs), np.max(xs)

    y_len = y_max - y_min
    x_len = x_max - x_min
    max_len = int(max(y_len, x_len) * 1.5)
    max_len = min(max_len, w, h)
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2

    if max_len >= w:
        x_min = 0
        x_max = w
    else:
        x_min = x_center - max_len // 2
        x_max = x_center + max_len // 2
        if x_min < 0:
            x_min = 0
            x_max = max_len
        if x_max > w:
            x_max = w
            x_min = x_max - max_len
    if max_len >= h:
        y_min = 0
        y_max = h
    else:
        y_min = y_center - max_len // 2
        y_max = y_center + max_len // 2
        if y_min < 0:
            y_min = 0
            y_max = max_len
        if y_max > h:
            y_max = h
            y_min = y_max - max_len
        
    return (int(y_min), int(y_max), int(x_min), int(x_max)), (h, w)

def focus_image(images, region):
    y_min, y_max, x_min, x_max = region
    new_images = []
    for i in images:
        i = i[:, y_min:y_max, x_min:x_max]
        new_images.append(i)
    return new_images

def focus_coord(coord, region, original_size):
    y_min, y_max, x_min, x_max = region
    h, w = original_size
    new_coord = []
    for c in coord:
        y = (c[0] * h - y_min) / (y_max - y_min)
        x = (c[1] * w - x_min) / (x_max - x_min)
        new_coord.append([y, x])
    return new_coord

def debug_save_img(image, path):
    image = image.to('cpu').numpy().transpose(1, 2, 0)[:, :, ::-1]
    image = (image * 255).astype(np.uint8)
    cv2.imwrite(path, image)

DROP_OUT_COORDS = [
    [2.0, 2.0],
    [2.0, 3.0],
    [3.0, 2.0],
    [3.0, 3.0],
]


if __name__ == '__main__':
    image = np.zeros((16, 16, 3))
    _, transform_info = generate_rand_transform(image, mode=0)
    transform = transform_info['transform']
    encode_transform = transform_info['encode_transform']
    print(transform['angle'])
    print(encode_transform['angle'])
    print(transform['scale'])
    print(encode_transform['scale'])
    print(transform['translation'])
    print(encode_transform['translation'])
    print(_.shape)
    exit()
