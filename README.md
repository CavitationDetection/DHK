# Deep Hierarchical Knowledge Loss for Fault Intensity Diagnosis

![framework](https://github.com/CavitationDetection/DHK/blob/main/figs/losses.jpg)

## Requirements

- Python 3.8.11
- torch 1.9.1
- torchvision 0.10.1

Note: our model is trained on a SLURM-managed server node with 2011G RAM, 128-core CPUs and eight NVIDIA A100.

## Code execution

- train.py is the entry point to the code.
- main.py is the main function of our model.
- models/xxx.py is the network structure of our method (e.g. resnet.py, mobilenet_v2.py, vit.py and so on).
- opts.py is all the necessary parameters for our method (e.g. factors, learning rate and data loading path and so on).
- focal_/hiera_tree_loss.py is the hierarchical tree loss and focal hierarchical loss.
- group_triplet_loss.py is the group tre triplet loss.
- train/test_data_loader.py represents the loading of training and test datasets.
- Execute train.py


## Test dataset
Download datasets from [here](https://drive.google.com/drive/folders/1eejPrqM2hWPxSfb0gUhu-F4FD0rhO7sp?usp=sharing) and place test signals in the subdirectories of ./Data/Test/







