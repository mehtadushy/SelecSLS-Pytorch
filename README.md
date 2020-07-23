# SelecSLS Convolutional Net Pytorch Implementation
Reference ImageNet implementation of SelecSLS Convolutional Neural Network architecture proposed in [XNect: Real-time Multi-Person 3D Motion Capture
with a Single RGB Camera](http://gvv.mpi-inf.mpg.de/projects/XNect/) (SIGGRAPH 2020).

The network architecture is 1.3-1.5x faster than ResNet-50, particularly for larger image sizes, with the same level of accuracy on different tasks! 
Further, it takes substantially less memory while training, so it can be trained with larger batch sizes!

### Update (28 Dec 2019)
Better and more accurate models / snapshots are now available. See the additional ImageNet table below.

### Update (14 Oct 2019) 
Code for pruning the model based on [Implicit Filter Level Sparsity](http://openaccess.thecvf.com/content_CVPR_2019/html/Mehta_On_Implicit_Filter_Level_Sparsity_in_Convolutional_Neural_Networks_CVPR_2019_paper.html) is also a part of the [SelecSLS model](https://github.com/mehtadushy/SelecSLS-Pytorch/blob/master/models/selecsls.py#L280) now. The sparsity is a natural consequence of training with adaptive gradient descent approaches and L2 regularization. It gives a further speedup of **10-30%** on the pretrained models with no loss in accuracy. See usage and results below.

## ImageNet results

The inference time for the models in the table below is measured on a TITAN X GPU using the accompanying scripts. The accuracy results for ResNet-50 are from torchvision, and the accuracy results for VoVNet-39 are from [VoVNet](https://github.com/stigma0617/VoVNet.pytorch).    
<table>
  <tr>
    <th></th>
    <th colspan="6">Forward Pass Time (ms)<br>for different image resolutions</th>
    <th colspan="2">ImageNet<br>Error</th>
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
    <td>ResNet-50</td>
    <td>15.0</td>
    <td>175.0</td>
    <td>11.0</td>
    <td>114.0</td>
    <td>7.2</td>
    <td>39.0</td>
    <td>23.9</td>
    <td>7.1</td>
  </tr>
  <tr>
    <td>VoVNet-39</td>
    <td>13.0</td>
    <td>197.0</td>
    <td>10.8</td>
    <td>130.0</td>
    <td>6</td>
    <td>41.0</td>
    <td>23.2</td>
    <td>6.6</td>
  </tr>
  <tr>
    <td>SelecSLS-60</td>
    <td>11.0</td>
    <td>115.0</td>
    <td>9.5</td>
    <td>85.0</td>
    <td>7.3</td>
    <td>29.0</td>
    <td>23.8</td>
    <td>7.0</td>
  </tr>
  <tr>
    <td>SelecSLS-60 (P)</td>
    <td>10.2</td>
    <td>102.0</td>
    <td>8.2</td>
    <td>71.0</td>
    <td>6.1</td>
    <td>25.0</td>
    <td>23.8</td>
    <td>7.0</td>
  </tr>
  <tr>  
   <td>SelecSLS-84</td>
    <td>16.1</td>
    <td>175.0</td>
    <td>13.7</td>
    <td>124.0</td>
    <td>9.9</td>
    <td>42.3</td>
    <td>23.3</td>
    <td>6.9</td>
  </tr>  
    <td>SelecSLS-84 (P)</td>
    <td>11.9</td>
    <td>119.0</td>
    <td>10.1</td>
    <td>82.0</td>
    <td>7.6</td>
    <td>28.6</td>
    <td>23.3</td>
    <td>6.9</td>
  </tr>     
  * (P) indicates that the model has batch norm fusion and pruning applied
</table>


The following models are trained using Cosine LR, Random Erasing, EMA, *Bicubic* Interpolation, and Color Jitter using [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models). The inference time for models here is measured on a TITAN Xp GPU using the accompanying scripts. The script for evaluating ImageNet performance uses *Bilinear* interpolation, hence the results reported here are marginally worse than they would be with Bicubic interpolation at inference. 

<table>
  
  <tr>
    <th></th>
    <th colspan="6">Forward Pass Time (ms)<br>for different image resolutions</th>
    <th colspan="2">ImageNet<br>Error</th>
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
    <td>SelecSLS-42_B</td>
    <td>6.4</td>
    <td>60.8</td>
    <td>5.8</td>
    <td>42.1</td>
    <td>5.7</td>
    <td>14.7</td>
    <td>22.9</td>
    <td>6.6</td>
  </tr>
  <tr>
    <td>SelecSLS-60</td>
    <td>7.4</td>
    <td>69.4</td>
    <td>7.3</td>
    <td>47.6</td>
    <td>7.1</td>
    <td>16.8</td>
    <td>22.1</td>
    <td>6.1</td>
  </tr>
  <tr>
    <td>SelecSLS-60_B</td>
    <td>7.5</td>
    <td>70.5</td>
    <td>7.3</td>
    <td>49.3</td>
    <td>7.2</td>
    <td>17.0</td>
    <td>21.6</td>
    <td>5.8</td>
  </tr>
  
</table>



# SelecSLS (Selective Short and Long Range Skip Connections)
The key feature of the proposed architecture is that unlike the full dense connectivity in DenseNets, SelecSLS uses a much sparser skip connectivity pattern that uses both long and short-range concatenative-skip connections. Additionally, the network architecture is more amenable to [filter/channel pruning](http://openaccess.thecvf.com/content_CVPR_2019/html/Mehta_On_Implicit_Filter_Level_Sparsity_in_Convolutional_Neural_Networks_CVPR_2019_paper.html) than ResNets.
You can find more details about the architecture in the following [paper](https://arxiv.org/abs/1907.00837), and details about implicit pruning in the [CVPR 2019 paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Mehta_On_Implicit_Filter_Level_Sparsity_in_Convolutional_Neural_Networks_CVPR_2019_paper.html).

Another recent paper proposed the VoVNet architecture, which shares some design similarities with our architecture. However, as shown in the above table, our architecture is significantly faster than both VoVNet-39 and ResNet-50 for larger batch sizes as well as larger image sizes.

## Usage
This repo provides the model definition in Pytorch, trained weights for ImageNet, and code for evaluating the forward pass time
and the accuracy of the trained model on ImageNet validation set. 
In the paper, the model has been used for the task of human pose estimation, and can also be applied to a myriad of other problems as a drop in replacement for ResNet-50.

```
wget http://gvv.mpi-inf.mpg.de/projects/XNectDemoV2/content/SelecSLS60_statedict.pth -O ./weights/SelecSLS60_statedict.pth
python evaluate_timing.py --num_iter 100 --model_class selecsls --model_config SelecSLS60 --model_weights ./weights/SelecSLS60_statedict.pth --input_size 512 --gpu_id <id>
python evaluate_imagenet.py --model_class selecsls --model_config SelecSLS60 --model_weights ./weights/SelecSLS60_statedict.pth --gpu_id <id> --imagenet_base_path <path_to_imagenet_dataset>

#For pruning the model, and evaluating the pruned model (Using SelecSLS60 or other pretrained models)
python evaluate_timing.py --num_iter 100 --model_class selecsls --model_config SelecSLS84 --model_weights ./weights/SelecSLS84_statedict.pth --input_size 512 --pruned_and_fused True --gamma_thresh 0.001 --gpu_id <id>
python evaluate_imagenet.py --model_class selecsls --model_config SelecSLS84 --model_weights ./weights/SelecSLS84_statedict.pth --pruned_and_fused True --gamma_thresh 0.001 --gpu_id <id> --imagenet_base_path <path_to_imagenet_dataset>
```

## Older Pretrained Models
- [SelecSLS-60](http://gvv.mpi-inf.mpg.de/projects/XNect/assets/models/SelecSLS60_statedict.pth)
- [SelecSLS-84](http://gvv.mpi-inf.mpg.de/projects/XNect/assets/models/SelecSLS84_statedict.pth)

## Newer Pretrained Models (More Accurate)
- [SelecSLS-42_B](http://gvv.mpi-inf.mpg.de/projects/XNect/assets/models/SelecSLS42_B_statedict.pth)
- [SelecSLS-60](http://gvv.mpi-inf.mpg.de/projects/XNect/assets/models/SelecSLS60_statedict_better.pth)
- [SelecSLS-60_B](http://gvv.mpi-inf.mpg.de/projects/XNect/assets/models/SelecSLS60_B_statedict.pth)

## Requirements
 - Python 3.5
 - Pytorch >= 1.1

## License 
The contents of this repository, and the pretrained models are made available under CC BY 4.0. Please read the [license terms](https://creativecommons.org/licenses/by/4.0/legalcode).

### Citing
If you use the model or the implicit sparisty based pruning in your work, please cite:

```
@inproceedings{XNect_SIGGRAPH2020,
 author = {Mehta, Dushyant and Sotnychenko, Oleksandr and Mueller, Franziska and Xu, Weipeng and Elgharib, Mohamed and Fua, Pascal and Seidel, Hans-Peter and Rhodin, Helge and Pons-Moll, Gerard and Theobalt, Christian},
 title = {{XNect}: Real-time Multi-Person {3D} Motion Capture with a Single {RGB} Camera},
 journal = {ACM Transactions on Graphics},
 url = {http://gvv.mpi-inf.mpg.de/projects/XNect/},
 numpages = {17},
 volume={39},
 number={4},
 month = July,
 year = {2020},
 doi={10.1145/3386569.3392410}
} 

@InProceedings{Mehta_2019_CVPR,
author = {Mehta, Dushyant and Kim, Kwang In and Theobalt, Christian},
title = {On Implicit Filter Level Sparsity in Convolutional Neural Networks},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
} 
```



