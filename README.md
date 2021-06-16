# Building Segmentation Using Deep Learning

I have tried various methods to generate the segmentation map from Remote Sensing Images. The results are stored in `Project_results` folder. For more details and experimentaion done please read this further. All the images can be viewed in QGIS or normal image viewer application.  (Updated code will be uploaded later)

### Different Approaches to Generate Segmentation Map

* ## Cycle-GAN
    * The aim was to use image to image translation using Cycle-GAN to get map from RS Images and vice-versa.
    * I used U-NET as generator and PatchGan as discriminator.
    * The model was able to generate some images resembling to satellite images using the mask images with some features resembling to ground and roof of buildings.
    * But the model was not performing well for the original task of generating buildings even after experimenting on Batch Sizes, Learning rates, Loss Functions and finally trying different Discriminators.




* ## DCGAN
    * I also tried using simple DCGAN approach to produce mask.
    * I used UNET and RESUNET as generator.
    * The UNET generated visually good looking mask compared to RESUNET.
    * I tried 256x256, 512x512, 128x128 and 384x384 as tile size. 256x256 performed better than all of the rest.
    * Among the batch sizes of 1, 8, 16 and 32, batch size of 1 performed better.
    * Among Loss functions like BCELoss, MSE, and L1, BCELoss performed better. Among optimisers, Adam was better.
    * There was a problem with RESUNET model. After some epochs the model suffered MODE COLLAPSE. I tried various learning rates and batch sizes. But the problem persisted.
    * For remedy of Mode Collapse, I tried different discriminator but it did not work.
    * So I used __Wasserstein Loss__ to check the difference. It worked better than other loss functions but it also failed after some epochs.
    * So as advised in __Wasserstein Loss__ paper, I trained the PatchGan discriminator 10-15 times more than the generator. I also reduced the learning rate of generator from 0.0002 to 0.0001.
    * After changing all this and training for 120 epochs, __the model did not suffer mode collapse__. However, the quality of output has suffered.

* ## UNET and RESUNET Segmentation
    * As expected, the supervised learning alogrithms generated much more better output than Generative Adversarial Network models.
    * UNET was trained for 100 epochs.
    * RESUNET was trained for 70 epochs.



### Project_results folder

It has 6 folder each specifiying image name. Each folder contains 6 images.
* `Original_Austin{}.tif`  --  Original Remote Sensing Image.
* `Original_Austin{}_MASK.tif` -- Orignal ground truth mask.
* `ResUnet_GAN_Austin{}.tif` -- Mask generated using DCGAN with RESUNET generator.
* `Unet_GAN_Austin{}.tif` -- Mask generated using DCGAN with UNET generator.
* `ResUnet_Segmentation_Austin{}.tif` -- Mask generated using supervised RESUNET.
* `Unet_Segmentation_Austin{}.tif` -- Mask generated using supervised UNET.

### Code Files Details

* `CycleGan_pytorch.py` : Cycle Gan model training using Pytorch
* `cyclegan_predict.py` : Cycle Gan model prediction using Pytorch
* `U_ResUnet_pytorch.py` : Unet and Resunet model using Pytorch
* `Predict_U_resunet_pytorch.py` : Unet and Resunet dcgan model prediction using Pytorch
* `ResUnetGan_Wasserstein_loss.py` : Resunet dcgan model using wasserstein loss using Pytorch
* `unet_resunet_train_keras.py` : Unet and Resunet segmentation using Keras
* `unet_resunet_predict_keras.py` : Unet and Resunet segmentation using Keras
