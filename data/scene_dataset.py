import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import json
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms import Resize

from transformers import CLIPTokenizer, T5TokenizerFast
import lightning as L

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from data_utils import focus_area, focus_image, focus_coord, debug_save_img, DROP_OUT_COORDS, get_bbox_from_mask

from parse_args import Args

SCALE_LIST = [0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5]

def get_tensor(normalize=None):
    transform_list = [ transforms.ToTensor() ]
    if normalize is not None:
        for item in normalize:
            transform_list += [ transforms.Normalize(item[0], item[1]) ]
    return transforms.Compose(transform_list)

class MtlTransformDataset(data.Dataset):

    def __init__(self, args: Args, data_dir, metadata_file):
        self.args = args
        self.stage = args.stage
        self.data_dir = data_dir
        self.metadata_file = metadata_file

        if args.text_encoder:
            self.clip_tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
                args.diffusion_model,
                subfolder="tokenizer"
            )
            self.t5_tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained(
                args.diffusion_model,
                subfolder="tokenizer_2"
            )

        metadata = {
            'name': [],
            'tag': [],
            'mask': [],
            'geometry': [],
            'irradiance': [],
            'mtl': [],
            'gt': [],
            'coords': [],
            'condition': []
        }

        with open(f'{self.data_dir}/{metadata_file}') as f:
            json_list = list(f)
        for _, json_str in enumerate(json_list):
            item = json.loads(json_str)
            scene = item['scene']
            mask = self.data_dir + '/' + item[args.mask_column]
            gt = self.data_dir + '/' + item[args.rgb_column]
            depth = self.data_dir + '/' + item[args.geometry_column]
            irradiance = self.data_dir + '/' + item[args.irradiance_column]
            mtl = self.data_dir + '/' + item[args.material_column]
            condition = self.data_dir + '/' + item[args.condition_column]

            # image paths
            metadata['mask'].append(mask)
            metadata['geometry'].append(depth)
            metadata['irradiance'].append(irradiance)
            metadata['mtl'].append(mtl)
            metadata['gt'].append(gt)
            metadata['condition'].append(condition)

            # name = f'{scene}_' + mask.split('/')[-1].split('.')[0]
            name = f'{scene}_' + condition.split('/')[-1].split('.')[0]
            metadata['name'].append(name)
            metadata['tag'].append(item.get('tag', 'train'))
            metadata['coords'].append(item.get('points', None))

        self.dataset = metadata
        self.indices = list(range(len(self.dataset['name'])))

        self.load_config = {
            'mask': {
                'convert': "L",
                'normalization': None,
            },
            'irradiance': {
                'convert': "RGB",
                'normalization': [([0.5], [0.5])],
            },
            'geometry': {
                'convert': "RGB",
                'normalization': [([0.5], [0.5])],
            },
            'mtl': {
                'convert': "RGB",
                'normalization': [([0.5], [0.5])],
                # 'resolution': [512, 512]
            },
            'gt': {
                'convert': "RGB",
                'normalization': [([0.5], [0.5])],
            },
            'condition': {
                'convert': "RGB",
                'normalization': [([0.5], [0.5])],
            },
        }

    def load_pt(self, key, path):
        convert = self.load_config[key]['convert']
        img = Image.open(path).convert(convert)
        img = transforms.ToTensor()(img)
        return img


    def normalize_pt(self, key, pt: torch.Tensor):
        normalization = self.load_config[key]['normalization']
        resolution = self.load_config[key].get('resolution', self.args.resolution)
        normalize_list = [transforms.Resize(resolution, transforms.InterpolationMode.BILINEAR)]
        if normalization is not None:
            for item in normalization:
                normalize_list.append(transforms.Normalize(item[0], item[1]))
        normalize_list = transforms.Compose(normalize_list)
        pt = normalize_list(pt)
        return pt
    
    def irradiance_exposure(self, irradiance_tensor, exposure):
        return torch.pow(torch.pow(irradiance_tensor, 2.2) * 2**(exposure), 0.4545)

    def __getitem__(self, index):
        # channel concat condition
        # 0. inference: noisy latent; training: gt latent add noise
        # 1. mask, reshape, see FLUX-Fill
        # 2. masked rgb, vae encode
        # 3. depth, vae encode
        # 4. irradiance, vae encode
        # sequence concat condition
        # mtl, vae encode, different pe
        mask_path = self.dataset['mask'][index]
        geometry_path = self.dataset['geometry'][index]
        irradiance_path = self.dataset['irradiance'][index]
        mtl_path = self.dataset['mtl'][index]
        gt_path = self.dataset['gt'][index]
        condition_path = self.dataset['condition'][index]

        gt_tensor = self.load_pt('gt', gt_path)
        mask_tensor = self.load_pt('mask', mask_path)
        mask_tensor = (mask_tensor > 0.25) + 0.0
        masked_rgb_tensor = gt_tensor * (1.0 - mask_tensor)
        geometry_tensor = self.load_pt('geometry', geometry_path)
        irradiance_tensor = self.load_pt('irradiance', irradiance_path)
        if self.args.irradiance_exposure is not None:
            irradiance_tensor = self.irradiance_exposure(irradiance_tensor, self.args.irradiance_exposure)

        mtl_tensor = self.load_pt('mtl', mtl_path)
        condition_tensor = self.load_pt('condition', condition_path)

        coords = self.dataset['coords'][index]
        if coords is None:
            coords = get_bbox_from_mask(mask_tensor)

        coords = torch.tensor(coords)

        gt_tensor = self.normalize_pt('gt', gt_tensor)
        mask_tensor = self.normalize_pt('mask', mask_tensor)
        masked_rgb_tensor = self.normalize_pt('gt', masked_rgb_tensor)
        geometry_tensor = self.normalize_pt('geometry', geometry_tensor)
        irradiance_tensor = self.normalize_pt('irradiance', irradiance_tensor)
        mtl_tensor = self.normalize_pt('mtl', mtl_tensor)
        condition_tensor = self.normalize_pt('condition', condition_tensor)

        # rotate = self.dataset['rotate'][index]
        # rotate_str = f"{rotate} degrees"
        # scale_x = self.dataset['scale'][index][0]
        # scale_y = self.dataset['scale'][index][1]
        # scale_str = f"{scale_x} along x-axis and {scale_y} along y-axis"
        # trans_x = self.dataset['translate'][index][0]
        # trans_y = self.dataset['translate'][index][1]
        # trans_str = f"{trans_x} units along x-axis and {trans_y} units along y-axis" 

        # prompt = ["Transform it"]

        # encode_rotate = rotate - 360 if rotate > 180 else rotate
        # encode_rotate = encode_rotate / 180.0 * np.pi
        # encode_scale_x = (SCALE_LIST.index(scale_x) - 4) / 4.0 * 3.0
        # encode_scale_y = (SCALE_LIST.index(scale_y) - 4) / 4.0 * 3.0
        # encode_scale = np.array([encode_scale_x, encode_scale_y])
        # encode_trans_x = trans_x - 1.0 if trans_x > 0.5 else trans_x
        # encode_trans_y = trans_y - 1.0 if trans_y > 0.5 else trans_y
        # encode_trans = np.array([encode_trans_x, encode_trans_y]) / 0.5 * np.pi
        # encode_transform = {
        #     "rotate": encode_rotate,
        #     "scale": encode_scale,
        #     "trans": encode_trans
        # }
        # real_transform = {
        #     "rotate": rotate,
        #     "scale": [scale_x, scale_y],
        #     "trans": [trans_x, trans_y]
        # }
        
        # clip_text_inputs = self.clip_tokenizer(
        #     prompt,
        #     padding="max_length",
        #     max_length=self.clip_tokenizer.model_max_length,
        #     truncation=True,
        #     return_overflowing_tokens=False,
        #     return_length=False,
        #     return_tensors="pt"
        # ).input_ids

        # t5_text_inputs = self.t5_tokenizer(
        #     prompt,
        #     padding="max_length",
        #     max_length=512,
        #     truncation=True,
        #     return_length=False,
        #     return_overflowing_tokens=False,
        #     return_tensors="pt",
        # ).input_ids

        dic = {
            "name": self.dataset['name'][index],
            "tag": self.dataset['tag'][index],
            "masked_rgb": masked_rgb_tensor,
            "mask": mask_tensor,
            "geometry": geometry_tensor,
            "irradiance": irradiance_tensor,
            "mtl": mtl_tensor,
            "gt": gt_tensor,
            "coords": coords,
            "condition": condition_tensor,
            # "prompt": prompt,
            # "clip_text_inputs": clip_text_inputs,
            # "t5_text_inputs": t5_text_inputs,
            # "encode_transform": encode_transform,
            # "transform": real_transform
        }
        return dic

    def __len__(self):
        return len(self.indices)

    def sub_dataset(self, ratio):
        total_size = len(self.indices)
        train_size = int(total_size * ratio)
        start_point = np.random.randint(0, len(self.indices) - train_size)
        self.indices = self.indices[start_point:start_point + train_size]
        for i in self.dataset.keys():
            self.dataset[i] = self.dataset[i][start_point:start_point + train_size]


class DataModule(L.LightningDataModule):
    def __init__(self, args: Args):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size

    def setup(self, stage=None):
        train_dataset = MtlTransformDataset(self.args, self.args.data_dir, metadata_file=self.args.metadata)
        val_dataset = MtlTransformDataset(self.args, self.args.data_dir, metadata_file=self.args.val_metadata)

        # train_dataset.sub_dataset(0.1)

        self.datasets = {
            'train': train_dataset,
            'val': val_dataset,
        }

    def train_dataloader(self):
        self.args.train_data_size = len(self.datasets['train'])
        return data.DataLoader(
            self.datasets['train'],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
    
    def val_dataloader(self):
        self.args.val_data_size = len(self.datasets['val'])
        return data.DataLoader(
            self.datasets['val'],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

    def test_dataloader(self):
        return self.val_dataloader()

if __name__ == '__main__':
    import yaml
    from parse_args import get_parser, Args
    from utils import exec_time

    with open('config/config.yaml') as f:
        default_args = yaml.safe_load(f)
    parser = get_parser()
    parser.set_defaults(**default_args.get('debug'))
    args = Args(parser.parse_args())
    args.batch_size = 2
    
    dm = DataModule(args)
    dm.setup()
    val_dataloader = dm.val_dataloader()
    val_loader = iter(val_dataloader)
    @exec_time
    def test_iter_time():
        return val_loader.__next__()
    
    for i in range(10):
        batch = test_iter_time()
        print(batch['coords'])
        Image.fromarray(((batch['masked_rgb'][0].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255.0).astype(np.uint8)).save(f'output/debug/masked_rgb_{i}.png')
        Image.fromarray(((batch['gt'][0].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255.0).astype(np.uint8)).save(f'output/debug/rgb_{i}.png')
        Image.fromarray(((batch['mtl'][0].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255.0).astype(np.uint8)).save(f'output/debug/condition_{i}.png')
        exit()
    
# ========================================
# test_iter_time       : 861.2ms
# test_iter_time       : 729.5ms
# test_iter_time       : 856.9ms
# test_iter_time       : 858.3ms
# test_iter_time       : 722.3ms
# test_iter_time       : 827.4ms
# test_iter_time       : 854.9ms
# test_iter_time       : 867.6ms
# test_iter_time       : 937.8ms
# test_iter_time       : 860.8ms
# ========================================
