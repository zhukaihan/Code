## Tensorflow Route

Adapted from [yirueilu-b's BlazePalm](https://github.com/yirueilu-b/blaze-palm-tf2).

Why not use Tensorflow to retrieve all the weights from TFLite model and finetune?
They can't.
The process of converting to TFLite model is irreversible, with all the optimization and etc.
However, with some careful engineering and matching, it is reversible.

As long as the original (or identical) model implementation is found, weights can simply be copied over.

yirueilu-b's BlazePalm implementation is indeed in Tensorflow.
However, they did not transfer the weights over from TFLite.

The following is the timeline:

1. - [X] Transfer all the weights from TFLite to Tensorflow.
2. - [ ] Modify the classification layers of the model.
3. - [ ] Modify loss to train only classifications.
4. - [ ] Collect data and train.

### Transferring the Weights
Details are in `finetune/tf/weight extraction.ipynb`.

There are two ways of getting the weights, through TFLite's inferencer, or through Python package TFLite (a TFLite model parser).

The TFLite's inferencer from Tensorflow does not work because there are custom operations in the model. Therefore, TFLite model parser needs to be used. 

The names of layers from TFLite parser matches surprisingly well with Tensorflow model. Also, the shapes of weights and bias matches as well. In the whole SSD network, there are only convolutions. Thus, the weights that needs to be copied are narrowed down (only those with "conv2d" in the name). Then, the layers that did not match by names are discovered and matched manually.

Finally, the weights are copied over. One thing to notice that the TFLite parser gives the weights in buffer. Thus, it is byte data. To reconstruct the actual weight numbers, use `np.frombuffer`(arrays can be used as buffer). Another thing to notice is that the weights' axis needs to be transposed or swapped as TFLite uses different weight layout than Tensorflow.
