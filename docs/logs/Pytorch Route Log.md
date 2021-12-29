## Pytorch Route
The Pytorch route is to convert the TFLite into Pytorch, finetune in Pytorch, and convert back into TFLite. 
I found this repository [vidursatija's BlazePalm](https://github.com/vidursatija/BlazePalm) where the author was trying to convert into CoreML. 
It already have the Pytorch weights available for use. 
However, it failed. 

### Change the Last Layer
Changing the last layer is simple. Set require_grads=False for all layers, create new layers for the output layers, and override the existing self.class_x with the new layers. 

### Convert Back to TFLite
This is where everything is broken. 
Traditionally, converting would first convert Pytorch to ONNX (using Pytorch) and then to Tensorflow Saved Graph (using onnx-tf) and then to TFLite (using Tensorflow). 
This method works, but it is not good. 

Converting back to TFLite with a performant model is possible, but with too much hassle. 
It lies in the difference in weights layout for Pytorch and TFLite. Pytorch has layout as NCWH and TFLite has layout as NWHC. 

Converting from Pytorch to ONNX keeps the NCWH layout, as they both only supports NCWH. 
Converting from ONNX to Tensorflow keeps the NCWH layout as well, as Tensorflow supports NCWH. 
Converting from Tensorflow to TFLite causes weights are converted into NWHC, but two transpose layers to be inserted before and after each convolution to make sure the intermediate outputs' shapes of each layer remains original (to make sure the model accepts the same input and output shapes). 

There is no way of transposing the weights in Pytorch, ONNX, or Tensorflow because it is not supported. 

There is no existing working tool that allows the transpose. 
Many efforts has been made, including [onnx2keras](https://github.com/waltYeh/onnx2keras), or [TinyNeuralNetwork](https://github.com/alibaba/TinyNeuralNetwork). They both do not work. onnx2keras returns errors, due to some unsupported operations and unimplemented marginal cases. TinyNeuralNetwork does not switch the axis of weights. It only performs optimization and quantization, which does have some speed-up, but not significant enough. 

