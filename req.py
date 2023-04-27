import subprocess

conda_env = 'CardioVt'
dependencies = ['h5py', 'yacs', 'scipy', 'pandas', 'scikit-image', 'numpy', 'matplotlib', 'opencv-python', 'PyYAML', 'scikit-learn', 'tensorboardX', 'tqdm']
pip_dependencies = ['mat73', 'torchvision', 'fvcore', 'simplejson', 'einops', 'timm', 'psutil', 'tensorboard']

# Create the conda environment
subprocess.call(f'conda create -n {conda_env} python=3.9', shell=True)
subprocess.call(f'conda activate {conda_env}', shell=True)

# Install packages using conda
subprocess.call(f'conda install -n {conda_env} {" ".join(dependencies)} -y', shell=True)

# Install packages using pip
for package in pip_dependencies:
    subprocess.call(f'pip install {package}', shell=True)

# Install av using conda
subprocess.call(f'conda install -n {conda_env} av -c conda-forge -y', shell=True)
