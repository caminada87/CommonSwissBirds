# Computer Vision Project "Common Swiss Birds"
*FHNW - DAS Data Science - HS2022*

*Author: Stefan Caminada*

## Task
It should be possible to classify images of the 10 most common bird species living in Switzerland.

## Data Collection / Preprocessing
1. the image data was extracted (crawled with selenium) from Google Image Search.
2. it was planned to use Object Detecion from OpenCV to get the most accurate image details. Unfortunately, it turned out that 2 different models could not reliably draw bounding boxes around the birds in the images.
3. I continued to use CV2 anyway to create image sections with dimension (224, 224), flip them and change brightness and contrast. Intermediate results were always saved.
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

| Model | Size(weights) | Test accurracy | Pfad(training/evaluation Notebook) |
|-------|---------------|----------------|------------------------------------|
| BaseModel | 10'923 KB | 0.7193126082420349 | 03_base_model/base_model.ipynb |
| ResNet50 | 100'806 KB | 0.9132569432258606 | 04_resnet_50/resnet_50_model.ipynb |
| EffcientNetV2L | 467'350 KB | 0.9533551335334778 | 05_efficientNetV2L/efficient_net_v2l_model.ipynb |
| MobileNetV3_Minimalistic | 4'435 KB | 0.8862520456314087 | 06_mobileNetV3_mini/mobilenet_v3_mini_model.ipynb |
| VGG16 | 59'628 KB | 0.929623544216156 | 07_VGG16/vgg_16_model.ipynb |
| custom VGG16 | - | - | Lernt nichts ⛷ gotta go now |
