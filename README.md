# Food Classifier 

To Do: 

Improve readme 

Explain motivation 

Explain how it works 

Explain what I want to get out 

Explain choices that were made and why 

Explain how to deploy 

## Overview 

This android app classifies food into one of 101 categories using a cloud compute architecture.

The purpose is to classify different foods in real world settings. 

### Machine Learning Component 

Priority: Good Accuracy, Low FLOPs, Quick Loading

The model used for inference is EfficientNetB3 (EFNB3). 
As shown in https://arxiv.org/abs/1905.11946, EFN architectures provide the 
highest accuracy per flop compared to SOTA at the of publication of 
the paper. B4 was chosen because of its good balance between accuracy and 
flops. Compared to EFNB2, EFNB3 provides a significant boost in accuracy. 
Sizes larger than EFNB4 provide minor increases in accuracy, but significant
increase in flops. EFNB4 was not chosen due to increase in parameters, which
would increase the time to initialize the network. 

#### Dataset Choice 

Food101 https://www.tensorflow.org/datasets/catalog/food101 was chosen 
because of its accessibility and quantity. Note that the training images
have noisy labels. 

#### Model Fine Tuning 

Priority: Train Quickly, Handle Noisy Data, Focus on Hard Examples 

Imagenet pretrained weights were used to initialize the network. Food images
have many common features with imagenet images. To handle noisy data, label 
smoothing https://arxiv.org/pdf/1701.06548.pdf was used. Focal loss was 
chosen over cross entropy in order to increase the impact of hard to classify
examples, even though its primary use is for class imbalance https://arxiv.org/abs/1708.02002.

### Server Component 

### App Component 


## How to Deploy 




