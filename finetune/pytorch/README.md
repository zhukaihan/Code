Adapted from [vidursatija's BlazePalm](https://github.com/vidursatija/BlazePalm). 

This is the Pytorch route of converting finetuning BlazePalm. However, due to weight layout issue, NCWH and NWHC incompatibility between Pytorch and TFLite, Pytorch model cannot convert into performant TFLite model. Thus, the Pytorch route is not working. 