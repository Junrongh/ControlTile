import os
import argparse
import torch

class Args:
    def __init__(self, args=None):

        if args.data_dir is None:
            raise ValueError('Specify `--data_dir`')

        if args.val_metadata is None:
            raise ValueError('Specify `--val_metadata `')

        self.stage = args.stage
        # model
        self.diffusion_model = args.diffusion_model
        self.output_name = args.output_name

        self.seed = args.seed
        devices = str(args.devices)
        if ',' in devices:
            self.devices: list = [int(device) for device in devices.split(',')]
        else:
            self.devices: list = [int(devices)]
        self.mix_precision = torch.bfloat16
        self.batch_size = args.batch_size

        self.inference_timesteps = args.inference_timesteps

        # test stage
        if self.stage == 'test':
            train_devices = str(args.train_devices)
            if ',' in train_devices:
                self.train_devices: list = [int(device) for device in train_devices.split(',')]
            else:
                self.train_devices: list = [int(train_devices)]
            self.img_output_folder = args.img_output_folder + '-s' + str(args.seed)
            self.ckpt = args.ckpt
            self.grid = args.grid
        else: # train stage
            self.model_output = args.model_output
            self.num_train_epochs = args.num_train_epochs
            self.checkpoints_total_limit = args.checkpoints_total_limit
            self.val_per_epoch = args.val_per_epoch
            self.learning_rate = args.learning_rate
            self.resume = args.resume
            if args.metadata is None:
                raise ValueError('Specify `--metadata`')

        self.attn_mask = args.attn_mask
        self.clip_text_pt = args.clip_text_pt
        self.t5_text_pt = args.t5_text_pt
        self.lora_rank = args.lora_rank
        self.lora_alpha = args.lora_alpha
        self.lora_modules = args.lora_modules
        self.logging_dir = args.logging_dir
        self.text_encoder = args.text_encoder

        # data
        self.data_dir = args.data_dir
        self.metadata = args.metadata
        self.val_metadata = args.val_metadata

        self.rgb_column = args.rgb_column
        self.mask_column = args.mask_column
        self.geometry_column = args.geometry_column
        self.irradiance_column = args.irradiance_column
        self.material_column = args.material_column
        self.condition_column = args.condition_column

        valid_resolution = False
        resolution = args.resolution
        if isinstance(resolution, str):
            if 'x' in resolution:
                resolution = resolution.split('x')
                self.resolution = [int(resolution[0].strip()), int(resolution[1].strip())]
                if self.resolution[0] % 8 == 0 and self.resolution[1] % 8 == 0:
                    valid_resolution = True
        else:
            self.resolution: int = int(args.resolution)
            if self.resolution % 8 == 0:
                valid_resolution = True
            self.resolution = [self.resolution, self.resolution]

        if not valid_resolution:
            raise ValueError(
                '`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder.'
            )

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--stage', type=str, default='train', choices=['train', 'test'], help='stage: train, test')
    # model
    parser.add_argument('--diffusion_model', type=str, default=None, help='Path to diffusion base model')
    parser.add_argument('--model_output', type=str, default=None, help='Path to save model checkpoint')
    parser.add_argument('--output_name', type=str, default='debug', help='Name of logger and ckpt file')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint or None')
    # data
    parser.add_argument('--data_dir', type=str, default=None, help=())
    parser.add_argument('--metadata', type=str, default=None, help='metadata.jsonl file')
    parser.add_argument('--val_metadata', type=str, default=None, help='metadata_val.jsonl file')
    parser.add_argument('--resolution', type=str, default='512', help=(
        'The resolution for input images, all the images in the train/validation dataset will be resized to this resolution'
        'Size could be described as height x width'
    ))
    parser.add_argument('--rgb_column', type=str, default='rgb')
    parser.add_argument('--mask_column', type=str, default='mask')
    parser.add_argument('--geometry_column', type=str, default='depth')
    parser.add_argument('--irradiance_column', type=str, default='irradiance')
    parser.add_argument('--material_column', type=str, default='mtl')
    parser.add_argument('--condition_column', type=str, default='condition')
    # training strategy
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--checkpoints_total_limit', type=int, default=None, help=('Max number of checkpoints to store')),
    parser.add_argument('--val_per_epoch', type=int, default=None, help=('check validation every N epochs')),

    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Initial learning rate')
    parser.add_argument('--logging_dir', type=str, default='logs', help=(
        '[TensorBoard](https://www.tensorflow.org/tensorboard) log directory'
    ))
    parser.add_argument('--grid', type=str, default='standard', choices=['standard', 'fisheye', 'swirl'], help='Grid type for RoPE')
    parser.add_argument('--attn_mask', type=bool, default=True, help='Use attention mask')
    parser.add_argument('--clip_text_pt', type=str, default=None, help='Path to clip text embeds')
    parser.add_argument('--t5_text_pt', type=str, default=None, help='Path to t5 text embeds')
    parser.add_argument('--text_encoder', type=bool, default=False, help='Use text encoder')
    
    parser.add_argument('--lora_rank', type=int, default=16, help=('LoRA rank'))
    parser.add_argument('--lora_alpha', type=int, default=32, help=('LoRA rank'))
    parser.add_argument('--lora_modules', type=str, default=None, help=('LoRA modules'))

    # inference parameters
    parser.add_argument('--inference_timesteps', type=int, default=50, help=())
    parser.add_argument('--ckpt', type=str, default=None, help=('Checkpoint path'))

    parser.add_argument('--devices', type=str, default='0', help='The device to use for this script.')
    parser.add_argument('--train_devices', type=str, default='0', help='The device to use for training.')
    parser.add_argument('--img_output_folder', type=str, default=None, help='Folder to save output images.')

    return parser
