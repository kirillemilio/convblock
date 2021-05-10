# convblock
Artificial neural networks architectures implemented via pytorch. This library's main purpose is to simplify creation, design and experiments with convolutional neural networks. Library is build on top of pytorch framework.

# Installation
```git clone https://github.com/kirillemilio/convblock.git && cd convblock && pip install .```

### ConvBlock module
This module provides simple, easy to understand and unified way to create typical convolutional layout. For example, simple sequential module can be described in a following way:

```python 
from convblock import ConvBlock

ConvBlock(input_shape=(32, 32, 32), layout='cna cna p',
          c=dict(kernel_size=3, filters=16), 
          p=dict(kernel_size=2, stride=2, mode='max'))

```

Various **ResNet** blocks are really simple to create:


```python
from convblock import ConvBlock

ConvBlock(input_shape=(32, 32, 32),
          layout='+ cna cna cn + a',
          c=dict(kernel_size=[1, 3, 1], filters=[8, 8, 32], stride=[1, 2, 1]),
          shortcut=dict(layout='c', kernel_size=3))
```

Simple DenseNet block can be created in a few lines of code:

```python

ConvBlock(input_shape=(32, 32, 32),
          layout='. cna cna .' * 4,
          c=dict(kernel_size=[1, 3] * 4,
                 filters=[8, 32] * 4))

```

If you want to reuse the same block structure but in 3D space(using 3D convolutions and batch normalization) you need just to change **input_shape** all other modules inside **ConvBlock** will be adapted to spatial dimension of input. So, the following code will generate the same **DenseBlock** that can be used on 3D image(for example, CT scans)

```python
from convblock import ConvBlock

ConvBlock(input_shape=(32, 32, 32, 32),
          layout='. cna cna .' * 4,
          c=dict(kernel_size=[1, 3] * 4,
                 filters=[8, 32] * 4))

```


More complex structures with bunch of branches can also be created and designed using just a few lines of code. For instance, here is implementation of **InceptionB** module using **ConvBranches** module that allows to describe branches of **ConvBlock** on par with mixing operation(., *, +):
```python
from convblock import ConvBranches

ConvBranches(
    input_shape=(32, 32, 32), mode='.',
    branch1x1={'layout': 'cna', 'c': {'filters': 192, 'kernel_size': 1}},
    branch_pool={'layout': 'p cna', 'c': {'filters': 192, 'kernel_size': 1},
                 'p': {'mode': 'avg', 'kernel_size': 3, 'stride': 1}},
    branch7x7={'layout': 'cna cna cna', 'c': {'kernel_size': [(1, 1), (1, 7), (7, 1)],
                                              'filters': [128, 128, 192]}},
    branch7x7dbl={'layout': 'cna cna cna cna cna',
                  'c': {'kernel_size': [(1, 1), (1, 7), (7, 1), (1, 7), (7, 1)],
                        'filters': (128, 128, 128, 128, 192)}}
)
```

Library also contains flexibly parametrized implementation of several popular CNN architectures for image classificaton/segmentation and detection. Here is non-exhaustive list of architectures:
- VGG
- ResNet
- DenseNet
- MobileNetV1, MobileNetV2, MobileNetV3
- ShuffleNetV1, ShuffleNetV2
- SqueezeNet
- InceptionV3, InceptionV4
- InceptionResNet
- Xception
- UNet
- VNet
- SSD