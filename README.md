# tfwn

[Weight Normalization](https://arxiv.org/abs/1602.07868) layer wrapper for TensorFlow-Keras API.

Inspired by [Sean Morgan](https://github.com/tensorflow/tensorflow/pull/21276) implementation, but:
- No data initialization (only eager mode was implemented in original pull request).
- Code refactoring
- More tests
- CIFAR10 example from original paper reimplemented

## Examples
Unfortunately I couldn't reproduce parer results on CIFAR10 with batch size 100.
As you can see there is no much difference in accuracy.

<img src="https://github.com/shkarupa-alex/tfwn/raw/master/examples/cifar10_accuracy_100.png">
<img src="https://github.com/shkarupa-alex/tfwn/raw/master/examples/cifar10_loss_100.png">


But with much smaller batch size model with weight normalization is much better then regular one.

<img src="https://github.com/shkarupa-alex/tfwn/raw/master/examples/cifar10_accuracy_16.png">
<img src="https://github.com/shkarupa-alex/tfwn/raw/master/examples/cifar10_loss_16.png">


## How to use
```python
import tensorflow as tf
from tfwn import WeightNorm


dense_wn = WeightNorm(tf.keras.layers.Dense(3))
out = dense_wn(input)
```


## References
### Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks
Tim Salimans, and Diederik P. Kingma.

```
@inproceedings{Salimans2016WeightNorm,
  title={Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks},
  author={Tim Salimans and Diederik P. Kingma},
  booktitle={Neural Information Processing Systems 2016},
  year={2016}
}
```
