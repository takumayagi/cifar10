# CIFAR-10
## Models
* WideResNet-16-4 top-1 93.0%
* C5 (5 convolutional layer, 1 FC layer with BN) top-1 81.6%
* C5 (80% train 20% validation 200epoch) top-1 87.7%

## Dependencies
* Python2.7+
* numpy
* chainer
* cupy
* scikit-learn
* scikit-image
* OpenCV
* joblib
* mllogger (http://gitlab.ks.cs.titech.ac.jp/yagi/mllogger)

## Functionalities
* train.py: training code
* nn.py: nearest neighbor on raw pixel distance
* nn_feature.py: nearest neighbor on both raw pixel/fc distance
* error_analysis.py: nearest neighbor plot of failure cases
* smooth_gradient.py: Smooth gradient (https://arxiv.org/abs/1706.03825) visualization

## Nearest Neighbor
Query/last layer from c3f2/image  
![top-page](https://raw.githubusercontent.com/takumayagi/cifar10/images/c3f2_nn_deep.jpg)
