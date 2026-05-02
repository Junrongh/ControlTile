import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import cv2

from torchvision import transforms

import torch
import torch.utils.data as data
from torch.utils.data import Subset

from transformers import CLIPTokenizer, T5TokenizerFast

import lightning as L

from data_utils import generate_rand_transform
from parse_args import Args
# from utils import exec_time

def get_tensor(img, normalize=None):
    transform_list = [ transforms.ToTensor() ]
    if normalize is not None:
        for item in normalize:
            transform_list += [ transforms.Normalize(item[0], item[1]) ]
    processor = transforms.Compose(transform_list)
    tensor = processor(img)
    return tensor

class TextureTransformDataset(data.Dataset):
    def __init__(self, args: Args, stage, data_dir, metadata_file=None):
        self.args: Args = args
        self.stage = stage
        self.data_dir = data_dir
        self.metadata_file = metadata_file
        self.count = 0
        self.clip_tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
            args.diffusion_model,
            subfolder="tokenizer"
        )

        self.t5_tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained(
            args.diffusion_model,
            subfolder="tokenizer_2"
        )

        metadata = { 'name': [], 'condition': [], 'tag': [] }

        with open(f'{data_dir}/{metadata_file}') as f:
            json_list = list(f)
        for _, json_str in enumerate(json_list):
            item = json.loads(json_str)
            name = item['name']
            metadata['name'].append(name)
            metadata['condition'].append(f'{data_dir}/original/{name}.jpg')
            tag = item.get('tag', 'train')
            metadata['tag'].append(tag)
        self.dataset = metadata
        self.load_config = {
            'rgb': {
                'normalization': [([0.5], [0.5])],
                'vae': True
            },
            'condition': {
                'normalization': [([0.5], [0.5])],
                'vae': True
            }
        }
        self.indices = list(range(len(self.dataset['name'])))

    def load_pt(self, key, img):
        normalization = self.load_config[key]['normalization']
        img = cv2.resize(img, self.args.resolution)
        tensor = get_tensor(img, normalization) # b, h, w, c -> b, c, h, w
        return tensor

    def __getitem__(self, index):
        self.count += 1
        self.count = self.count % 3

        mtl_path = self.dataset['condition'][index]
        mtl_img = cv2.imread(mtl_path) # [0, 255]
        mtl_img = cv2.cvtColor(mtl_img, cv2.COLOR_BGR2RGB)
        # gt_img, transform_info = generate_rand_transform(mtl_img, mode=self.count+1) # [0, 255]
        gt_img, transform_info = generate_rand_transform(mtl_img, mode=0) # [0, 255]
        condition_tensor = self.load_pt('condition', mtl_img / 255.0) # [-1, 1]
        gt_tensor = self.load_pt('rgb', gt_img / 255.0) # [-1, 1]
        original_coord = get_tensor(transform_info['original_coord'], normalize=None) # b, h, w, 3 -> b, 3, h, w
        transform_coord = get_tensor(transform_info['transformed_coord'], normalize=None) # b, h, w, 3 -> b, 3, h, w

        # matrix_tensor = torch.tensor(transform_info['matrix']).view(2, 3)
        # transform_str = transform_info['transform_str']
        # prompt = f"Create an image maintaining the specified texture details. Apply a rotation of {transform_str['angle']} around the center point. Scale the image by a factor of {transform_str['scale']}. Translate the composition by {transform_str['translation']}"
        # prompt = f"Keep the texture details, rotate {transform_str['angle']} by center, scale {transform_str['scale']}, translate {transform_str['translation']}"
        # prompt = f"Apply 2D transformation to this image while maintaining the texture details unchanged."
        # prompt = [prompt]

        dic = {
            "name": self.dataset['name'][index],
            "tag": self.dataset['tag'][index],
            "target": gt_tensor,
            "condition": condition_tensor,
            # "matrix": matrix_tensor,
            # "encode_transform": transform_info['encode_transform'],
            "transform":transform_info['transform'],
            "original_coord": original_coord,
            "transformed_coord": transform_coord,
            # "prompt": prompt,
            # "clip_text_inputs": clip_text_inputs,
            # "t5_text_inputs": t5_text_inputs
        }
        return dic

    def __len__(self):
        return len(self.indices)

    def sub_dataset(self, ratio):
        total_size = len(self.indices) // 36
        train_size = int(total_size * ratio) * 36
        start_point = np.random.randint(0, len(self.indices) - train_size)
        self.indices = self.indices[start_point:start_point + train_size]
        for i in self.dataset.keys():
            self.dataset[i] = self.dataset[i][start_point:start_point + train_size]

class DataModule(L.LightningDataModule):
    def __init__(self, args: Args):
        super().__init__()
        self.args: Args = args

    def setup(self, stage=None):
        train_dataset = TextureTransformDataset(self.args, 'train', self.args.data_dir, metadata_file=self.args.metadata)
        # dataset.sub_dataset(0.1)

        val_dataset = TextureTransformDataset(self.args, 'val', self.args.data_dir, metadata_file=self.args.val_metadata)
        self.datasets = {
            'train': train_dataset,
            'val': val_dataset,
        }

    def train_dataloader(self):
        return data.DataLoader(
            self.datasets['train'],
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4
        )
    
    def val_dataloader(self):
        return data.DataLoader(
            self.datasets['val'],
            batch_size=4,
            shuffle=False,
            num_workers=4
        )

    def test_dataloader(self):
        return self.val_dataloader()


if __name__ == '__main__':
    import yaml
    from parse_args import get_parser, Args
    from utils import exec_time

    with open('config/toy_config.yaml') as f:
        default_args = yaml.safe_load(f)
    parser = get_parser()
    parser.set_defaults(**default_args.get('train'))
    args = Args(parser.parse_args())
    args.batch_size = 2
    
    dm = DataModule(args)
    dm.setup()
    val_dataloader = dm.val_dataloader()
    @exec_time
    def test_iter_time():
        return val_dataloader.__iter__().__next__()
    
    for i in range(10):
        data = test_iter_time()
        for k in ['target', 'condition', 'original_coord', 'transformed_coord']:
            d = data[k]
            for j in range(d.shape[0]):
                cv2.imwrite(f'output/images/debug/{k}_{i}_{j}.png', (d[j][:3, :, :].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)[:, :, ::-1] * 255.0)