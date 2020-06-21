## AlexNet Implementation
This repository is about AlexNet in Tensorflow 2 , I used tf.keras.Model and tf.layers.Layer instead of tf.keras.models.Sequential.  This allows us to customize and have full control of the model, I also used custom training instead of relying on the fit() function.  
In case we have very huge dataset, I applied online loading (by batch) instead of loading the data completely at the beginning. This will eventually not consume the memory.  
However, in case you prefer tf.keras.models.Sequential based models and loading the data completely in the beginning, then please refer to my repository: [AlexNet](https://github.com/Bao-Jiarong/AlexNet)

#### AlexNet Architecrure      
<p></p>
<center>
<img src="img/1.png" align="center" width="700" height="300"/>
</center>

Figure 1: image is taken from [source](https://link.springer.com/chapter/10.1007/978-3-030-04212-7_32)   

<center>   
<img src="img/2.png" width="700" height="300"/>   
</center>

Figure 2: image is taken from [source](https://www.mdpi.com/2072-4292/9/8/848/htm)   

### Training on MNIST
<p></p>
<center>
<img src="img/mnist.png" width="400" height="350"/>
</center>

### Requirement
```
python==3.7.0
numpy==1.18.1
```
### How to use
Training & Prediction can be run as follows:    
`python train.py train`  
`python train.py predict img.png`  


### More information
* Please refer to the original paper of AlexNet [here](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) for more information.

### Implementation Notes
* **Note 1**:   
Since datasets are somehow huge and painfully slow in training ,I decided to make number of filters variable. If you want to run it in your PC, you can reduce the number of filters into 32,16,8,4 or 2. (64 is by default). For example:  
`model = alexnet.AlexNet((113, 113, 3), classes = 10, filters = 8)`

* **Note 2** :   
You can also make the size of images smaller, so that it can be ran faster and doesn't take too much memories.

### Result for MNIST:   
* Learning rate = 0.0001  
* Batch size = 32  
* Optimizer = Adam   
* Filters = 8
* epochs = 2

Name |  Training Accuracy |  Validation Accuracy  |
:---: | :---: | :---:
Alexnet | 93.75% | 95.57%
