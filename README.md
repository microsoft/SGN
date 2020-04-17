# Semantics-Guided Neural Networks for Efficient Skeleton-Based Human Action Recognition (SGN)

## Introduction

Skeleton-based human action recognition has attracted great interest thanks to the easy accessibility of the human skeleton data. Recently, there is a trend of using very deep feedforward neural networks to model the 3D coordinates of joints without considering the computational efficiency. In this work, we propose a simple yet effective semantics-guided neural network (SGN). We explicitly introduce the high level semantics of joints (joint type and frame index) into the network to enhance the feature representation capability. Intuitively, semantic information, i.e., the joint type and the frame index, together with dynamics (i.e., 3D coordinates) reveal the spatial and temporal configuration/structure of human body joints and are very important for action recognition.
In addition, we exploit the relationship of joints hierarchically through two modules, i.e., a joint-level module for modeling the correlations of joints in the same frame and a frame-level module for modeling the dependencies of frames by taking the joints in the same frame as a whole. A strong baseline is proposed to facilitate the study of this field. With an order of magnitude smaller model size than most previous works, SGN achieves the state-of-the-art performance.
 

<div align=center>
<img src="https://github.com/microsoft/SGN/blob/master/images/para.PNG" width = 50% height = 50% div align=center>
</div>

Figure 1: Comparisons of different methods on NTU60 (CS setting) in terms of accuracy and the number of parameters. Among these methods, the proposed SGN model achieves the best performance with an order of magnitude smaller model size.



## Framework
![image](https://github.com/microsoft/SGN/blob/master/images/framework.PNG)

Figure 2: Framework of the proposed end-to-end Semantics-Guided Neural Network (SGN). It consists of a joint-level module and a frame-level module. In DR, we learn the dynamics representation of a joint by fusing the position and velocity information of a joint. Two types of semantics, i.e., joint type and frame index, are incorporated into the joint-level module and the frame-level module, respectively. To model the dependencies of joints in the joint-level module, we use three GCN layers. To model the dependencies of frames, we use two CNN layers.

## Prerequisites
The code is built with the following libraries:
- Python 3.6
- [Anaconda](https://www.anaconda.com/)
- [PyTorch](https://pytorch.org/) 1.3

## Data Preparation

We use the dataset of NTU60 RGB+D as an example for description. We need to first dowload the [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D) dataset.

- Extract the dataset to ./data/ntu/nturgb+d_skeletons/
- Process the data
```bash
 cd ./data/ntu
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```


## Training

```bash
# For the CS setting
python  main.py --network SGN --train 1 --case 0
# For the CV setting
python  main.py --network SGN --train 1 --case 1
```

## Testing

- Test the pre-trained models (./results/NTU/SGN/)
```bash
# For the CS setting
python  main.py --network SGN --train 0 --case 0
# For the CV setting
python  main.py --network SGN --train 0 --case 1
```

## Reference

This repository holds the code for the following paper:

[Semantics-Guided Neural Networks for Efficient Skeleton-Based Human Action Recognition](https://arxiv.org/abs/1904.01189). CVPR, 2020.

If you find our paper and repo useful, please cite our paper. Thanks!

```
@inproceedings{zhang2020semantics,
  title={Semantics-Guided Neural Networks for Efficient Skeleton-Based Human Action Recognition},
  author={Zhang, Pengfei and Lan, Cuiling and Zeng, Wenjun and Xing, Junliang and Xue, Jianru and Zheng, Nanning},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020},
}

```
## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

