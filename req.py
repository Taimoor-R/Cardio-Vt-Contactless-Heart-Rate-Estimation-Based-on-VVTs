import os
import subprocess
import sys

conda_env = 'CardioVt'
core_dependencies = ['python=3.9', 'numpy', 'scipy', 'pandas']
other_dependencies = ['h5py', 'yacs', 'scikit-image', 'matplotlib', 'opencv-python', 'PyYAML', 'scikit-learn', 'tensorboardX', 'tqdm']
pip_dependencies = ['mat73', 'torchvision', 'fvcore', 'simplejson', 'einops', 'timm', 'psutil', 'tensorboard', 'matplotlib', 'opencv-python']

# Get the path to the conda executable
conda_path = os.path.dirname(sys.executable)

# Create the conda environment
subprocess.call(f'{conda_path}/conda create -n {conda_env} {" ".join(core_dependencies)} -y', shell=True)

# Install core packages using conda
subprocess.call(f'{conda_path}/conda install -n {conda_env} {" ".join(core_dependencies)} --no-update-deps -y', shell=True)

# Install other packages using conda
subprocess.call(f'{conda_path}/conda install -n {conda_env} {" ".join(other_dependencies)} --no-update-deps -y', shell=True)

# Install packages using pip
for package in pip_dependencies:
    subprocess.call(f'{conda_path}/conda run -n {conda_env} pip install {package}', shell=True)

# Install av using conda
subprocess.call(f'{conda_path}/conda install -n {conda_env} av -c conda-forge --no-update-deps -y', shell=True)
