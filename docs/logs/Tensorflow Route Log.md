# Tensorflow Route

Adapted from [yirueilu-b's BlazePalm](https://github.com/yirueilu-b/blaze-palm-tf2).

Why not use Tensorflow to retrieve all the weights from TFLite model and finetune?
They can't.
The process of converting to TFLite model is irreversible, with all the optimization and etc.
However, with some careful engineering and matching, it is reversible.

As long as the original (or identical) model implementation is found, weights can simply be copied over.

yirueilu-b's BlazePalm implementation is indeed in Tensorflow.
However, they did not transfer the weights over from TFLite.

The following is the timeline:

1. [X] Transfer all the weights from TFLite to Tensorflow.
2. [X] Modify the classification layers of the model.
3. [ ] Modify loss to train only classifications.
4. [ ] Collect data and train.

## Transferring the Weights
Details are in `finetune/tf/weight extraction.ipynb`.

There are two ways of getting the weights, through TFLite's inferencer, or through Python package TFLite (a TFLite model parser).

The TFLite's inferencer from Tensorflow does not work because there are custom operations in the model. Therefore, TFLite model parser needs to be used. 

The names of layers from TFLite parser matches surprisingly well with Tensorflow model. Also, the shapes of weights and bias matches as well. In the whole SSD network, there are only convolutions. Thus, the weights that needs to be copied are narrowed down (only those with "conv2d" in the name). Then, the layers that did not match by names are discovered and matched manually.

Finally, the weights are copied over. One thing to notice that the TFLite parser gives the weights in buffer. Thus, it is byte data. To reconstruct the actual weight numbers, use `np.frombuffer`(arrays can be used as buffer). Another thing to notice is that the weights' axis needs to be transposed or swapped as TFLite uses different weight layout than Tensorflow.

Then, it does not work (worked eventually). The model does not perform well at some point. The model does recognize the hand at certain distance, but not at distances. After inspecting the two models, the mediapipe's model and custom model, despite the weights are copied correctly, there is an error in max-pool layer. The original model is 2x2 window size, while the custom is 5x5. After changing it to 2x2, it worked flawlessly. 

## Modify the Classification Layers
Details are in `finetune/tf/development.ipynb` (TODO: will be cleaned up later).

The classification layers can be modified during initial model building. The weights for classification layers are not yet important (will be trained with custom data), so the weights for these layers do not need to be transferred. 

There is an issue with the new model with custom classification layers with 7 classes. It is substantially slower than the original (one class). The slowdown gets more significant as the number of classes increases. With 1 class, it takes 0.07-0.09 seconds for inference. With 7 classes, it takes 0.23-0.32 seconds for inference, which is almost unusable on an average computer. Float-16 quantization does not help. Thus, the feasibility of this approach will be re-evaluated. 
