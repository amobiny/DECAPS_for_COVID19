# Detailed-oriented Capsule Network
Official PyTorch implementation of the Detailed-oriented Capsule Network (DECAPS) proposed in 
the paper [Radiologist-Level COVID-19 Detection Using CT Scans with Detail-Oriented Capsule Networks](https://arxiv.org/pdf/2004.07407.pdf).

![FastCapsNet](imgs/decaps.png)
*Fig1. Detailed-oriented Capsule Network architecture for COVID-19 Detection from CT scans*


## Dependencies
- Python (3.5 preferably; should also works fine with python 2.7)
- NumPy
- [PyTorch](https://pytorch.org/)>=1.1
- [torchvision](https://pytorch.org/)>=0.3
- [Tensorflow](https://github.com/tensorflow/tensorflow)>=1.10
- Matplotlib (for saving images)


## How to run the code

### 1. Prepare your data
To run the code, you first need to store your data in a folder named 'data' inside the project folder. 
Given the current DataLoader code, it must be an HDF5 file containing train, validation and test sets. 


### 2. Train
Most of the network parameters can be found in ```config.py``` file. You may modify them or run with
the default values which runs the 3D Fast Capsule Network proposed in the paper.


Training the model displays the training results and saves the trained model after each epoch
if an improvement observed in the accuracy value.
- For training in the default setting: ```python train.py ```
- Loading the model and continue training: ```python train.py --reload_epoch=epoch_num``` 
where ```epoch_num``` determines th model number to be reload (e.g. is epoch_num=3, 
it will load the model trained and stored after 3 epochs).
- For training AlexNet network: 
```python train.py --model=alexnet --loss_type=cross_entropy --add_recon_loss=False```

### 3. Test:
- For running the test: ```python inference.py --reload_epoch=epoch_num```
where ```epoch_num``` determines th model number to be reload (e.g. is epoch_num=3, 
it will load the model trained and stored after 3 epochs).



