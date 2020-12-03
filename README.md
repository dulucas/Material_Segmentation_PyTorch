# Material Estimation
This is an unofficial implementation of this [paper](http://labelmaterial.s3.amazonaws.com/release/cvpr2015-minc.pdf) for material estimation. The authors has provided their model weights in Caffe while not the code for inference(which requires denseCRF for post processing). This repo converts their original Caffe model into PyTorch, then re-implement the denseCRF, shift-pooling and LRN(local response normalization). Note that the denseCRF used here is RGB based and the hyper-parameters are copied from this [repo](https://github.com/kazuto1011/deeplab-pytorch). Please check [here](https://www.philkr.net/code/) if you want to use the denseCRF mentioned in the paper

## Requirement
- [pydensecrf](https://github.com/lucasb-eyer/pydensecrf)
- [pytorch](https://pytorch.org/)
- opencv

## Citation
```bash
@inproceedings{bell2015material,
  title={Material recognition in the wild with the materials in context database},
  author={Bell, Sean and Upchurch, Paul and Snavely, Noah and Bala, Kavita},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={3479--3487},
  year={2015}
}
```

## Examples
