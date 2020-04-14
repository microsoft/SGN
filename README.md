
# Semantics-Guided Neural Networks for Efficient Skeleton-Based Human Action Recognition (SGN)

## Introduction
We propose a simple yet effective semantics-guided neural network (SGN) for skeleton-based action recognition. We explicitly introduce the high level semantics of joints (joint type and frame index) into the network to enhance the feature representation capability. In addition, we exploit the relationship of joints hierarchically through two modules, i.e., a joint-level module for modeling the correlations of joints in the same frame and a framelevel module for modeling the dependencies of frames by taking the joints in the same frame as a whole. A strong baseline is proposed to facilitate the study of this field. With an order of magnitude smaller model size than most previous works, SGN achieves the state-of-the-art performance on the NTU60 dataset.
 

This repository holds the codes and methods for the following paper:

**Semantics-Guided Neural Networks for Efficient Skeleton-Based Human Action Recognition**. CVPR, 2020, [paper](https://arxiv.org/abs/1904.01189)


## Framework
![image](https://github.com/microsoft/SGN/blob/master/images/framework.pdf)
Figure 2:   Framework of the proposed end-to-end Semantics-Guided Neural Network (SGN). 

## Prerequisites
The code is built with following libraries:
- Python 3.6
- [Anaconda](https://www.anaconda.com/)
- [PyTorch](https://pytorch.org/) 1.3

## Data Preparation

We need to first dowload the [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D) dataset

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
python  main.py --model SGN --train 1
```

## Testing

```bash
python  main.py --model SGN --train 0
```

## Reference
If you find our papers and repo useful, please cite our papers. Thanks!

```
@article{zhang2019semantics,
  title={Semantics-Guided Neural Networks for Efficient Skeleton-Based Human Action Recognition},
  author={Zhang, Pengfei and Lan, Cuiling and Zeng, Wenjun and Xing, Junliang and Xue, Jianru and Zheng, Nanning},
  journal={CVPR},
  year={2020}
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

