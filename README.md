## Denoise Autoencoder Implementation
This repository is about Denoise AutoEncoder in Tensorflow 2 , I used tf.keras.Model and tf.layers.Layer instead of tf.keras.models.Sequential.  This allows us to customize and have full control of the model, I also used custom training instead of relying on the fit() function.  
In case we have very huge dataset, I applied online loading (by batch) instead of loading the data completely at the beginning. This will eventually not consume the memory.  

#### The Architecrure of Denoise Autoencoder     
<p></p>
<center>
<img src="img/d1.png" align="center" width="700" height="300"/>
</center>   

Figure 1: image is taken from [source](https://github.com/ALPHAYA-Japan/autoencoder/blob/master/README.md)   


### Training on Flowers & MNIST
<p></p>   
<center>
<img src="img/flowers.png" width="400" height="350"/>
<p></p>
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
* Please refer to the original paper of Denoise AutoEncoder [here](https://www.pyimagesearch.com/2020/02/24/denoising-autoencoders-with-keras-tensorflow-and-deep-learning/) for more information.

### Implementation Notes
* **Note 1**:   
Since datasets are somehow huge and painfully slow in training ,I decided to make number of units variable. If you want to run it in your PC, you can reduce or increase the number of units into any number you like. (512 is by default). For example:  
`model = conv_ae.Conv_AE((None,height, width, channel), latent = 200, units=16)`

* **Note 2** :   
You can also make the size of images smaller, so that it can be ran faster and doesn't take too much memories.


### Result for Flowers:
* width = 24 << 1
* height= 24 << 1  
* Learning rate = 0.0001
* Batch size = 32  
* Optimizer = Adam   
* units = 32
* latent = 200

Epoch | Training Loss |  Validation Loss  |
:---: | :---: | :---:
1  | 0.0752 | 0.0677
10 | 0.0397 | 0.0376
20 | 0.0281 | 0.0278
80 | 0.0187 | 0.0184

Epoch | True image and predicted image
:---: | :---:
1 | <img src="img/f_1.png" />
10 | <img src="img/f_10.png" />
20 |<img src="img/f_20.png" />
80 |<img src="img/f_80.png" />

##### Flowers dataset training procedure     
<p></p>  
<center>
<img src="img/chart.png" align="center" width="700" height="300"/>
</center>   
