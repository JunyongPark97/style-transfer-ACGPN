# style-transfer

InstaGAN 의 code 를 참고하여 작성되었으며 A 색상의 옷을 입고 있는 사람 사진에서 A 색상의 옷을 따고 input clothes B의 색상을 적용시켜 B 색상의 옷을 입고 있는 사람을 만들어내는 style-transfer 를 진행

## How to install 
```
$ pip install -r requirements.txt
```

## How to train
1. visdom server 를 통해 training 과정을 tracking 할 수 있다. 먼저 visdom server 를 켜서 visualizer 를 활성화시킨다. 
```
$ python -m visdom.server
```
2. 그 다음 training 을 진행한다. 
```
$ python train.py
```

## Experiment List
**content loss relu4_2 by using VGG19 + style loss relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 에 0.2 x gramMatrix MSE**

1. GAN + content 200 epoch
2. GAN + content + style 75 epoch -> GAN + content 125 epoch
3. GAN + content + cycle -> D initialize, GAN + content
4. base content + style 200 epoch
5. input 의 color 반영이 영향을 끼치는 것 같다 -> gray scale input 
6. GAN 으로 색을 먼저 학습하도록 설정 -> 그 다음 실험을 멈추고 content 집어넣어서 학습 
7. AdalN Generator + content + GAN + style 

## How to test
```
$ python test.py
```

## Data Structure 
```
|
|--checkpoints
|
|--data 
|    |
|    |--dataset (test dataset 으로 나누어야함)
|    |    |--clothes
|    |    |    |--base
|    |    |    |--mask
|    |    |
|    |    |--images
|    |    |    |--base
|    |    |    |--mask
|    |    |    |--segmentation
|    |
|    |--sgunit_train_dataset.py
|    |
|    |
|    
|--models 
|    |
|    |--base_model.py
|    |--networks.py
|    |--SGUNIT_gan_model.py
|
|--options
|    |
|    |--base_options.py
|    |--train_options.py
|    |--test_options.py
|    
|--util
|    |
|    |--gramMatrix.py
|    |--get_data.py
|    |-- ...
|
|--train.py
|--test.py
```
