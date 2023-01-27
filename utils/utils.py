def recompile_custom_output(base_model):
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer with the 10 bird classes
    predictions = Dense(10, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional ResNet50 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def plot_history(history, filename=None):
    """Plots or stores the history of an optimization run
    Parameters: 
        history: history
            The history to plot
        filename: str
            The path and name of the file to save the confusion matrix (will not be plotted to the screen if set)
    """
    
    import matplotlib.pyplot as plt
        
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()

    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    
    if filename is not None:
        plt.savefig(filename, dpi=300)
    else: 
        plt.show()

def get_labels_from_categorical_batch_dataset(ds):
    import numpy as np

    test_ls = list(ds.as_numpy_iterator())
    true_labels = []
    for batch in test_ls:
        for batchitem in batch[1].tolist():
            true_labels.append(np.argmax(batchitem))
    return true_labels

def get_labels_from_int_batch_dataset(ds):
    import numpy as np

    test_ls = list(ds.as_numpy_iterator())
    true_labels = []
    for batch in test_ls:
        for batchitem in batch[1].tolist():
            true_labels.append(batchitem)
    return true_labels

def plot_sns_confusion_matrix(confusion_matrix, class_names):
    import pandas as pd
    import seaborn as sn
    import matplotlib.pyplot as plt
    
    df_cm = pd.DataFrame(confusion_matrix, columns=class_names, index = class_names)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 12}, fmt='g')

def plot_showcase(showcase_ds, class_names, predictions):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(25,16))
    for i in range(1):
        for images,labels in showcase_ds.take(1):
            j=0
            for image in images:
                image = image/255.
                plt.subplot(6,5,j+1)
                plt.imshow(image)
                class_name= class_names[labels[j].numpy()]
                prediction_name = class_names[predictions[j].argmax()]
                title_obj = plt.title("C: " + class_names[labels[j].numpy()] + " || P: " + class_names[predictions[j].argmax()])
                if class_name != prediction_name:
                    plt.setp(title_obj, color='r') 
                plt.axis(False)
                j=j+1
    plt.show()

def plot_showcase_gradcam_overlay(showcase_path, class_names, model, last_convolution_layer_name, dim=(7,7)):
    from os import walk, path
    import numpy as np
    import matplotlib.pyplot as plt

    gradcam_images = []

    for (dirpath, dirnames, filenames) in walk(showcase_path):
        label = path.basename(dirpath)
        #print(f'Label: {label}')
        for filename in filenames:
            filepath = path.join(showcase_path, label, filename)
            #print(f'Filename: {filepath}')
            img, pred = get_gradcam_overlay_and_pred_from_path(filepath, model, last_convolution_layer_name, dim)
            #print(class_names[np.argmax(pred)])
            gradcam_images.append((img, label, class_names[np.argmax(pred)]))

    plt.figure(figsize=(25,16))
    j=0
    for img, label, pred in gradcam_images:
        plt.subplot(6,5,j+1)
        plt.imshow(np.clip((img/255.), 0, 1).astype(np.float32))
        title_obj = plt.title("C: " + label + " || P: " + pred)
        if label != pred:
            plt.setp(title_obj, color='r') 
        plt.axis(False)
        j=j+1
    plt.show()

def get_gradcam_overlay_from_path(image_path, model, last_convolution_name):
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    import cv2
    from google.colab.patches import cv2_imshow
    import tensorflow.keras.backend as K

    DIM = 224
    single_image_path = image_path
    img = image.load_img(single_image_path, target_size=(DIM, DIM))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    #preds = model.predict(x)

    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer(last_convolution_name)
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(x)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
    
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape((7, 7))
    
    #used earlier to show the 7 by 7 heatmap:
    #plt.matshow(heatmap)
    #plt.show  

    img = cv2.imread(single_image_path)
    INTENSITY = 0.5
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    img = heatmap * INTENSITY + img

    return img

def get_gradcam_overlay_and_pred_from_path(image_path, model, last_convolution_name, dim):
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    import cv2
    from google.colab.patches import cv2_imshow
    import tensorflow.keras.backend as K

    DIM = 224
    single_image_path = image_path
    img = image.load_img(single_image_path, target_size=(DIM, DIM))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x, verbose=0)

    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer(last_convolution_name)
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(x)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
    
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape(dim)
    
    #used earlier to show the 7 by 7 heatmap:
    #plt.matshow(heatmap)
    #plt.show  

    img = cv2.imread(single_image_path)
    INTENSITY = 0.5
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    img = heatmap * INTENSITY + img

    return img, preds[0]