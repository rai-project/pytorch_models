# SRGAN in PyTorch

A PyTorch implementation of SRGAN based on CVPR 2017 paper Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.

# Requirements
- [Anaconda](https://www.anaconda.com/download/)
- PyTorch
```
conda install pytorch torchvision -c pytorch
```
- opencv
```
conda install opencv
```

## Convert to python-free model

```bash
python convert_python_free.py --image 'penguin.png'
```