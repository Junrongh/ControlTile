conda install -y nvidia:cuda-toolkit nvidia:cudnn
conda install -y lightning -c conda-forge
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# check cuda and cudnn version
python -c "import torch; print('[cuda version]: ' + torch.version.cuda); print('[cudnn version]: ' + str(torch.backends.cudnn.version()))"

pip install numpy opencv-python Pillow
pip install accelerate diffusers transformers
pip install datasets einops tensorboard
