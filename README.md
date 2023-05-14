# CardioVt:Contactless Heart Rate Estimation Based on Video Transformers

Welcome to the project repository for CardioVt!

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Issues](#issues)

## Description

This repository contains the code and experiments for the final year project in BSc. Computer Science (University of Surrey), focusing on Heart Rate Estimation using Video Transformers. The project explores the application of video transformers, including a hybrid transformer architecture, in estimating heart rates from video data. It addresses the challenges associated with illumination variance, motion artifacts, and the limitations of traditional Convolutional Neural Networks (CNNs) for heart rate estimation.

Through an iterative research process, a unique hybrid transformer architecture has been developed and evaluated against various network variations and existing state-of-the-art models. The hybrid transformer combines the strengths of both 3D-CNNs and video transformers to enhance the accuracy and generalizability of heart rate estimation from videos.

The code and experiments in this repository showcase the development and evaluation of the hybrid transformer model specifically designed for heart rate estimation. The project culminates in the implementation of a robust hybrid transformer model that takes video input and accurately estimates the corresponding heart rate.

This project represents a significant contribution to the field of medical computer vision by providing a novel approach to heart rate estimation using a hybrid transformer architecture. By leveraging the capabilities of both 3D-CNNs and video transformers, the hybrid model overcomes the limitations of traditional CNNs and demonstrates improved performance in heart rate estimation tasks

## Installation

To get started with CardioVt, please follow these steps:

1. Clone the repository:
  ``` 
  git clone git@github.com:Taimoor-R/HR-VViT-Contactless-Heart-Rate-Estimation-Based-on-VViTs.git 
  ```
2. Install Miniconda (if not already installed):
  ```mkdir -p ~/miniconda3
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
   bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
   rm -rf ~/miniconda3/miniconda.sh
   ~/miniconda3/bin/conda init bash
   ~/miniconda3/bin/conda init zsh
  ```
3. Install project dependencies and creat virtual env
  ``` 
  python req.py 
  ```
4. Activate the Conda Env 'CardioVt' we created:
  ``` 
  conda activate CardioVt 
  ```


## Usage
To use Train CardioVt, follow these steps:

[Provide instructions on how to use your project]

[Add more steps if needed]

## Contributing
We welcome contributions from the community. If you want to contribute to CardioVt, please follow these guidelines:

Fork the repository.

Create a new branch.

Make your changes and commit them with descriptive messages.

Push your changes to your forked repository.

Submit a pull request, explaining your changes in detail.

## License
CardioVt is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license. This means that you are free to use, modify, and share the code and experiments in this repository under the following conditions:

- Attribution: You must give appropriate credit to the original author(s) and provide a link to the license. You must indicate if any changes were made to the original work.

- NonCommercial: The licensed material can only be used for non-commercial purposes. You cannot use it for commercial gain.

- ShareAlike: If you adapt or modify the licensed material, you must distribute your work under the same license terms as the original.

Please refer to the [License](LICENSE) file in this repository for more detailed information about the license.

It's important to read and understand the full license text to ensure compliance with the licensing terms.

## Acknowledgements
We would like to express our gratitude to the following repositories and authors for their valuable contributions and open-source code, which greatly assisted in the development of this project:

1. rPPG-Toolbox by Xin Liu, Xiaoyu Zhang, Girish Narayanswamy, Yuzhe Zhang, Yuntao Wang, Shwetak Patel, and Daniel McDuff. "Deep physiological sensing toolbox." arXiv preprint arXiv:2210.00716 (2022). 
  - https://github.com/ubicomplab/rPPG-Toolbox
2. TimeSformer-pytorch by Gedas Bertasius, Heng Wang, and Lorenzo Torresani. "Is Space-Time Attention All You Need for Video Understanding?" arXiv preprint arXiv:2102.05095 (2021).
  - https://github.com/lucidrains/TimeSformer-pytorch
  - https://github.com/facebookresearch/TimeSformer
3. PhysNet by Zitong Yu, Xiaobai Li, and Guoying Zhao. "Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks." In Proc. BMVC, 2019.
  - https://github.com/ZitongYu/PhysNet
5. ViViT-pytorch by Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lučić, and Cordelia Schmid. "ViViT: A Video Vision Transformer." arXiv preprint arXiv:2103.15691 (2021).
  - https://github.com/rishikksh20/ViViT-pytorch

We would like to extend our appreciation to these repositories and authors for their valuable resources, which significantly contributed to the success of this project.
## Issues
### Git and GPU Notebook intergration
I faced the issue of Paperspace not properly interacting with git speciffically private github projects, therefore all the commits have been added at a later date while uploading the project to github. Furthermore, due to the nature of the project and experimentation with mulitple models and creation of new model it was not possible to keep a updated git as the objective was not to create a frontend or backend ddevelopment project rather a research. Therefore, due to these reasons the git commits are limited, if required access to Paperspace notebook can be given to see history/images of notebook throughout the project.
