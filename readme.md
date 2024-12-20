# PointNCBW: Towards Dataset Ownership Verification for Point Clouds via Negative Clean-label Backdoor Watermark

This is the official implementation of our paper [PointNCBW: Towards Dataset Ownership Verification for Point Clouds via Negative Clean-label Backdoor Watermark](https://ieeexplore.ieee.org/abstract/document/10745757/), accepted by IEEE Transactions on Information Forensics and Security (TIFS), 2024. 



## Reference

If our work or this repo is useful for your research, please cite our paper as follows:

```
@article{wei2024pointncbw,
  title={PointNCBW: Towards Dataset Ownership Verification for Point Clouds via Negative Clean-label Backdoor Watermark},
  author={Wei, Cheng and Wang, Yang and Gao, Kuofeng and Shao Shuo and Li, Yiming and Wang, Zhibo and Qin, Zhan},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2024},
  publisher={IEEE}
}
```

## Pipeline

![](https://github.com/weic0810/PointNCBW/blob/main/asserts/pipe.png)

## Requirements

To install requirements

```
pip install -r requirements.txt
```
## Data Preparation

Please download ModelNet dataset from its [official website](http://modelnet.cs.princeton.edu/ModelNet40.zip) to .dataset/.

## NCBW Watermark

you can run as default

```
python ncbw.py --dataset modelnet40 --model pointnet
```

 

