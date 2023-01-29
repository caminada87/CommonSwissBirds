# Computer Vision Project "Common Swiss Birds"
*FHNW - DAS Data Science - HS2022*

*Author: Stefan Caminada*

## Task
It should be possible to classify images of the 10 most common bird species living in Switzerland.

## GIT info
There are not all pictures and models I saved in this git repo. But the complete dataset is and most of the best models except the EffcientNetV2L-model because and the customVGG16-model because they are too big.

## Data Collection / Preprocessing
1. the image data was extracted (crawled with selenium) from Google Image Search.
> Notebook:
> - 01_data_prep/notebook/crawl_chrome_selenium.ipynb
2. it was planned to use Object Detecion from OpenCV to get the most accurate (bird) image details. Unfortunately, it turned out that 2 different models could not reliably draw bounding boxes around the birds in the images.
3. I continued to use CV2 anyway to create image sections with dimension (224, 224), flip them and change brightness and contrast. Intermediate results were always saved.
> Python code:
> - 01_data_prep/python/image_functions.py
> - 01_data_prep/python/prepare_dataset.py
4. to avoid data leakage, training/validation was separated from test/showcase images from the beginning. --> More precisely, the training set does not contain mirrored or otherwise modified versions of the images that are in the test set.
5. finally this is how the datasets were created:
    - 02_data\99_dataset_preparation\train_images (5000 files. is divided into training and validation).
    - 02_data\99_dataset_preparation\test_images (1222 Files)
    - 02_data\99_dataset_preparation\showcase_images (30 Files, 3 per class)

## "03_base_model/base_model.ipynb"
A minimal CNN was created to see what could be classified without much effort.

## More sophisticated models...
The structure of the notebooks is now pretty much the same everywhere.
- Imports
- Setting up the connection to Google Drive
- Import of my own "utils" functions
- Defining the 10 classes 
- Loading the Tensorflow batch datasets (training, valid, test and showcase)
- Autotune for optimal execution times
- Loading the pre-trained model without head
- Recompile with new output layers
- Make only new layers trainable
- Set callbacks to save the best weights during training and earlystopping if validation Accurracy gets worse two times in a row.
- Train over maximum 50 epochs
- Training and validation history plotting
- Make all layers of the model trainable, recompile with changed optimizer and with small learning rate.
- Train over a maximum of 50 epochs (did not work with EfficientNetV2L, because it was too big, but in the first round it is already better than anything else...)
- trainings and validations - plot history

- Reinitialize model
- Load stored (best) weights
- Evaluate model
- Own evaluation (For test set and showcase):
    - Predict dataset labels
    - Create Classification report on the dataset
    - Confusion Matrix
- Advanced showcase evaluation:
    - Plot all 30 images
    - Create Single Prediction of each image and add Grad-Cam Overlay to see which image parts were important for prediction.

## Summary
**It's amazing how well a small model can perform and how much more it takes to get even better**

| Model | Size(weights) | training accurracy | Validation accurracy | Test accurracy | Path(training/evaluation Notebook) |
|---|---|---|---|---|---|
| BaseModel | 10'923 KB | 0.9860 | 0.8670 | 0.7390 | 03_base_model/base_model.ipynb |
| ResNet50 | 100'806 KB | 0.9868 | 0.9540 | 0.9182 | 04_resnet_50/resnet_50_model.ipynb |
| EffcientNetV2L | 467'350 KB | 0.9760 | 0.9700 | 0.9501 | 05_efficientNetV2L/efficient_net_v2l_model.ipynb |
| MobileNetV3_Minimalistic | 4'435 KB | 0.9933  | 0.9680 | 0.9092 | 06_mobileNetV3_mini/mobilenet_v3_mini_model.ipynb |
| VGG16 | 59'628 KB | 0.9985 | 0.9640 | 0.9206 | 07_VGG16/vgg_16_model.ipynb |
| custom VGG16 | 524'674 KB | 0.9983 | 0.9050 | 0.7905 | 08_custom_CNN/custom_CNN.ipynb |

**Observations**

- Why is the custom VGG16 almost 10 times bigger than VGG 16? Because in the custom one i strictly applied the head from the riginal architecture flatten/dense(4096)/dense(4096/dense(10)) instead of using my own head (GlobalAveragePooling2D/dense(1024)/dense(10))

- The custom VGG16 takes many epochs to learn because it has to learn everything from the beginning. It is also quite overfitting as we see in the difference between the training/val accurracy and the test accurracy in the evaluation.

- The only model which was not getting better during finetuning was the MobileNetV3_Minimalistic.

- efficient_net_v2l_model was the most resistant to overfitting. The training and val and even the test score behave very similar.

- The GradCam overlay shows that sometimes the model learns things maybe a bit wrongly... I guess here it could be helpful to have a bigger Dataset to train it properly. The most sense of this overlay I get out of the resnet example.

**Conclusion / learnings**

- From the beginning, before the data augmentation, I have consistently separated the test set from the training set. It would have been better for the training to do the same with the validation set. There it can be in this work that mirrorings of pictures or pictures with changed brightness/contrast values are contained as in the training set, which could lead to slight data leakage at training time. I would do this differently next time.

- All in all I am very satisfied with the result, if I wanted to increase the values I would continue with the Keras tuner and start tuning the hyperparameters. But this requires more capacities. With my Colab account I probably don't have this capacity for this project.
