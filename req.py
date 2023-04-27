import subprocess

dependencies = ['torchvision', 'fvcore', 'simplejson', 'einops', 'timm', 'psutil', 'scikit-learn', 'opencv-python', 'tensorboard']

for package in dependencies:
    subprocess.call(f'pip install {package}', shell=True)
    
subprocess.call(f'conda install av -c conda-forge', shell=True)