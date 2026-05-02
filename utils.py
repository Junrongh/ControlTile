import time
import random
import torch
from lightning.pytorch.utilities import rank_zero_only
import threading

GLOBAL_LOG_STATE = {
    "exec_time": '',
    "memory_usage_train": '',
    "memory_usage_val": ''
}

def print_periodically(log_file):
    with open(log_file, 'w') as f:
        f.write('[exec_time]          : ' + GLOBAL_LOG_STATE['exec_time'] + '\n')
        f.write('[memory_usage_train] : ' + GLOBAL_LOG_STATE['memory_usage_train'] + '\n')
        f.write('[memory_usage_val]   : ' + GLOBAL_LOG_STATE['memory_usage_val'] + '\n')

class RepeatingTimer:
    def __init__(self, interval, *args, **kwargs):
        self.interval = interval
        self.function = print_periodically
        self.args = args
        self.kwargs = kwargs
        self.timer = None
        self.is_running = False

    def start(self):
        if not self.is_running:
            self.is_running = True
            self._run()

    def _run(self):
        self.function(*self.args, **self.kwargs)
        if self.is_running:
            self.timer = threading.Timer(self.interval, self._run)
            self.timer.daemon = True
            self.timer.start()

    def stop(self):
        self.is_running = False
        if self.timer:
            self.timer.cancel()

def exec_time(func):
    def new_func(*args, **args2):
        start_time = time.time()
        back = func(*args, **args2)
        now_time = time.time()
        print_str = "{:20s}".format(func.__name__) + " : " + "{:5.1f}".format((now_time - start_time) * 1000) + "ms"
        GLOBAL_LOG_STATE['exec_time'] = print_str
        print(print_str)
        return back
    return new_func

@rank_zero_only
def get_memory_usage(phase, models, optimizer):
    torch.cuda.synchronize()
    total = torch.cuda.memory_allocated() / 1024**2  # MB

    param_mems = 0
    optim_mems = 0
    grad_mems = 0
    for model in models:
        param_mems += sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        optim_mem = 0
        for param in model.parameters():
            if param in optimizer.state:
                optim_mem += sum(
                    state.numel() * state.element_size() 
                    for state in optimizer.state[param].values()
                )
            if param.grad is not None:
                grad_mems += param.grad.numel() * param.grad.element_size() / 1024**2
        optim_mem /= 1024**2
        optim_mems += optim_mem
    data_mem = total - param_mems - optim_mems - grad_mems
    print_str = f"[{phase}] Total: {total:.2f} MB | " + f"Model: {param_mems:.2f} MB | " + f"Grad: {grad_mems:.2f} MB | " + f"Optimizer: {optim_mems:.2f} MB | " + f"Data+Activation: {data_mem:.2f} MB"
    GLOBAL_LOG_STATE[f'memory_usage_{phase}'] = print_str
    return print_str

def meshgrid_from_points(points: torch.Tensor, target_H: int=32, target_W: int=32, H: int = 32, W: int = 32, additional_value=1.0) -> torch.Tensor:
    """
    points: [B, 4, 2], order ABCD, each point is (y, x) in [0, 1]
    return: [B, H*W, 2]
    """
    assert points.ndim == 3 and points.shape[1:] == (4, 2), "shape must be [B, 4, 2]"
    batch_size = points.shape[0]
    device = points.device
    dtype = points.dtype

    point_A = points[:, 0] # [B, 2]
    point_B = points[:, 1] # [B, 2]
    point_C = points[:, 2] # [B, 2]

    dir_x = point_B - point_A    # [B, 2]
    dir_y = point_C - point_A    # [B, 2]

    u = torch.linspace(0, W-1, W, device=device, dtype=dtype)
    v = torch.linspace(0, H-1, H, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(v, u, indexing='ij')  # [H, W], [H, W]

    uu = uu.view(1, H, W, 1)  # [1, H, W, 1]
    vv = vv.view(1, H, W, 1)  # [1, H, W, 1]

    A_ = point_A.clone()
    A_[:, 0] *= H - 1
    A_[:, 1] *= W - 1
    A_ = A_.to(device).view(batch_size, 1, 1, 2)

    dir_x_ = dir_x.view(batch_size, 1, 1, 2) # [B,1,1,2]
    dir_y_ = dir_y.view(batch_size, 1, 1, 2) # [B,1,1,2]

    pts = A_ + uu * dir_x_ + vv * dir_y_  # [B, H, W, 2]
    pts = pts.view(batch_size, H * W, 2)  # [B, 1024, 2] for 32x32

    additional_channel = torch.zeros_like(pts[:, :, 0]).unsqueeze(2) + additional_value  # [B, H*W, 1]
    pts = torch.cat((additional_channel, pts), dim=2)
    scale_H = target_H / H
    scale_W = target_W / W
    pts[:, :, 1] *= scale_H
    pts[:, :, 2] *= scale_W
    return pts.to('cpu')

def fisheye_meshgrid_from_points(
    points: torch.Tensor, 
    target_H: int = 32, 
    target_W: int = 32, 
    H: int = 32, 
    W: int = 32, 
    k: float = 0.5,
    additional_value=1.0
) -> torch.Tensor:
    """
    Generates a grid with fisheye distortion on a plane defined by four given points.

    points: [B, 4, 2], ordered A-B-C-D, where each point is (y, x) normalized between [0, 1].
    k: Distortion coefficient. k > 0 produces barrel distortion (fisheye).
    return: [B, H*W, 3] (includes an additional_value channel).
    """
    batch_size = points.shape[0]
    device = points.device
    dtype = points.dtype

    point_A = points[:, 0].view(batch_size, 1, 1, 2)
    dir_x = (points[:, 1] - points[:, 0]).view(batch_size, 1, 1, 2)
    dir_y = (points[:, 2] - points[:, 0]).view(batch_size, 1, 1, 2)

    u = torch.linspace(0, 1, W, device=device, dtype=dtype)
    v = torch.linspace(0, 1, H, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(v, u, indexing='ij')
    
    u_norm = uu * 2 - 1
    v_norm = vv * 2 - 1
    
    r_sq = u_norm**2 + v_norm**2
    distortion = (1 + k * r_sq)
    
    u_distorted = u_norm * distortion
    v_distorted = v_norm * distortion
    
    u_final = (u_distorted + 1) / 2
    v_final = (v_distorted + 1) / 2

    u_final = u_final.unsqueeze(0).unsqueeze(-1) # [1, H, W, 1]
    v_final = v_final.unsqueeze(0).unsqueeze(-1) # [1, H, W, 1]
    
    pts = point_A + u_final * dir_x + v_final * dir_y # [B, H, W, 2]
    
    pts = pts.view(batch_size, H * W, 2)
    
    pts[:, :, 0] *= (target_H - 1)
    pts[:, :, 1] *= (target_W - 1)

    add_chan = torch.full((batch_size, H * W, 1), additional_value, device=device, dtype=dtype)
    res = torch.cat((add_chan, pts), dim=2)

    return res.to('cpu')

import math

def swirl_meshgrid_from_points(
    points: torch.Tensor, 
    target_H: int = 32, 
    target_W: int = 32, 
    H: int = 32, 
    W: int = 32, 
    angle_max: float = 2.0,
    radius_max: float = 1.0,
    additional_value=1.0
) -> torch.Tensor:
    """
    Generates a grid with a swirl distortion on a plane defined by four given points.

    points: [B, 4, 2], ordered A-B-C-D, where each point is (y, x) normalized between [0, 1].
    angle_max: The maximum rotation in radians; positive values for clockwise, negative for counter-clockwise.
    """
    print("*" * 80)
    print("using swirl meshgrid from points!")
    batch_size = points.shape[0]
    device = points.device
    dtype = points.dtype

    point_A = points[:, 0].view(batch_size, 1, 1, 2)
    dir_x = (points[:, 1] - points[:, 0]).view(batch_size, 1, 1, 2)
    dir_y = (points[:, 2] - points[:, 0]).view(batch_size, 1, 1, 2)

    u = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    v = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(v, u, indexing='ij')

    r = torch.sqrt(uu**2 + vv**2)
    theta = torch.atan2(vv, uu)

    rotation_amount = angle_max * torch.clamp((radius_max - r) / radius_max, min=0)
    
    new_theta = theta + rotation_amount

    u_distorted = r * torch.cos(new_theta)
    v_distorted = r * torch.sin(new_theta)

    u_final = (u_distorted + 1) / 2
    v_final = (v_distorted + 1) / 2
    
    u_final = u_final.unsqueeze(0).unsqueeze(-1) # [1, H, W, 1]
    v_final = v_final.unsqueeze(0).unsqueeze(-1) # [1, H, W, 1]

    pts = point_A + u_final * dir_x + v_final * dir_y # [B, H, W, 2]
    pts = pts.view(batch_size, H * W, 2)

    pts[:, :, 0] *= (target_H - 1)
    pts[:, :, 1] *= (target_W - 1)

    add_chan = torch.full((batch_size, H * W, 1), additional_value, device=device, dtype=dtype)
    res = torch.cat((add_chan, pts), dim=2)

    return res.to('cpu')

def patch_condition(condition, coords, size):
    """
    condition: [B, C, H, W]
    coords: [B, 4, 2]
    size: float
    return: new_condition: [B, C, H * size, W * size], new_coords: [B, 4, 2]
    """
    if size == 1.0:
        return condition, coords
    b, _, h, w = condition.shape
    new_condition = []
    new_coords = []
    for i in range(b):
        condition_i = condition[i] # [C, H, W]
        coords_i = coords[i] # [4, 2]
        pos_x = int(random.random() * (1.0 - size) * w)
        pos_y = int(random.random() * (1.0 - size) * h)
        new_condition_i = condition_i[:, pos_y:pos_y + int(size * h), pos_x:pos_x + int(size * w)]

        pt1 = coords_i[0, :]
        pt2 = coords_i[1, :]
        pt3 = coords_i[2, :]
        pt4 = coords_i[3, :]

        delta_x = pt2 - pt1
        delta_y = pt3 - pt1

        new_pt1 = pt1 + delta_x * pos_x / w + delta_y * pos_y / h
        new_pt2 = new_pt1 + delta_x * size
        new_pt3 = new_pt1 + delta_y * size
        new_pt4 = new_pt1 + delta_x * size + delta_y * size
        new_coords_i = torch.stack([new_pt1, new_pt2, new_pt3, new_pt4], dim=0)

        new_condition.append(new_condition_i)
        new_coords.append(new_coords_i)

    new_condition = torch.stack(new_condition, dim=0) # [B, C, H, W]
    new_coords = torch.stack(new_coords, dim=0) # [B, 4, 2]

    return new_condition, new_coords
