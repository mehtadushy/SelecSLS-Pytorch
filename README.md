# SelecSLS Convolutional Net Pytorch Implementation
Reference ImageNet implementation of SelecSLS Convolutional Neural Network architecture proposed in [XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera](https://arxiv.org/abs/1907.00837).

The network architecture is 1.2-1.5x faster than ResNet-50, particularly for larger image sizes, with the same level of accuracy on different tasks! Further, it takes substantially less memory while training, so it can be trained with larger batch sizes!

## ImageNet results
    
<table>
  <tr>
    <th></th>
    <th colspan="6">Forward Pass Time (ms)<br>for different image resolutions</th>
    <th colspan="2">ImageNet<br>Accuracy</th>
  </tr>
  <tr>
    <td></td>
    <td colspan="2">512x512</td>
    <td colspan="2">400x400</td>
    <td colspan="2">224x224</td>
    <td>Top-1</td>
    <td>Top-5</td>
  </tr>
  <tr>
    <td>Batch Size</td>
    <td>1</td>
    <td>16</td>
    <td>1</td>
    <td>16</td>
    <td>1</td>
    <td>16</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>ResNet50</td>
    <td>15.0</td>
    <td>175.0</td>
    <td>11.0</td>
    <td>114.0</td>
    <td>7.2</td>
    <td>39.0</td>
    <td>23.85</td>
    <td>7.13</td>
  </tr>
  <tr>
    <td>VoVNet39</td>
    <td>13.0</td>
    <td>197.0</td>
    <td>10.8</td>
    <td>130.0</td>
    <td>6</td>
    <td>41.0</td>
    <td>23.23</td>
    <td>6.57</td>
  </tr>
  <tr>
    <td>SelecSLS60</td>
    <td>11.0</td>
    <td>115.0</td>
    <td>9.5</td>
    <td>85.0</td>
    <td>7.3</td>
    <td>29.0</td>
    <td>23.78</td>
    <td>7.04</td>
  </tr>
</table>

The inference time is measured on a TITAN X GPU

# SelecSLS (Selective Short and Long Range Skip Connections)
The key feature of the proposed architecture is that unlike the full dense connectivity in DenseNets, SelecSLS uses a much sparser skip connectivity pattern that uses both long and short-range concatenative-skip connections. Additionally, the network architecture is more amenable to filter/channel pruning than ResNets.
You can find more details in the [paper](https://arxiv.org/abs/1907.00837).


## Usage
This repo provides the model definition in Pytorch, trained weights for ImageNet, and code for evaluating the forward pass time
and the accuracy of the trained model on ImageNet validation set. 
In the paper, the model has been used for the task of human pose estimation, and can also be applied to a myriad of other problems as a drop in replacement for ResNet-50.

```
wget http://gvv.mpi-inf.mpg.de/projects/XNectDemoV2/content/SelecSLS60_statedict.pth -o ./weights/SelecSLS60_statedict.pth
python evaluate_timing.py --num_iter 100 --model_class selecsls --model_config SelecSLS60 --input_size 512 --gpu_id <id>
python evaluate_imagenet.py --model_class selecsls --model_config SelecSLS60 --model_weights ./weights/SelecSLS60_statedict.pth --gpu_id <id> --imagenet_base_path <path_to_imagenet_dataset>
```

## Pretrained Models
- [SelecSLS-60](http://gvv.mpi-inf.mpg.de/projects/XNectDemoV2/content/SelecSLS60_statedict.pth)






