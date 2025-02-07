<h1 align="center"> Wildfire Classification from Satellite Images </h1>
<p align="center">
    Implemented a CNN-based classification system for over 40,000 satellite images. 
    <br>
    Built and trained the DenseNet201 model for wildfire prediction. The model can achieve 97% accuracy in the testing set.
</p>    

<div align="center">
  <img alt="Tensorflow" src="https://img.shields.io/badge/-Tensorflow-orange?style=flat&logo=Tensorflow&logoColor=white">
</div>

## Exploratory
+ Number of no wildfire images : 14500 images
+ Number of wildfire images : 15750 images
``` python
def exploratory(directory):

    labels = os.listdir(directory)

    label_counts = Counter()
    for label in labels:
        label_path = os.path.join(directory, label)
        label_counts[label] = len(os.listdir(label_path))

    print("Number of no wildfire images : " + str(label_counts["nowildfire"]))
    print("Number of wildfire images : " + str(label_counts["wildfire"]))

    # Plot the distribution
    plt.bar(label_counts.keys(), label_counts.values())
    plt.xlabel('Labels')
    plt.ylabel('Number of Images')
    plt.title('Label Distribution of Training Images')
    plt.show()

```
```
exploratory(train_data_dir)
```

![img_2.png](figure/distribution.png)

## Data Preprocessing
Preprocessed images  using multiple image preprocessing techniques including colour adjustment, increasing sharpness, median filtering, and contrast enhancement.
``` python
def color_adjustment(image, brightness_range=(-50, 50)):
    brightness = random.randint(brightness_range[0], brightness_range[1])
    adjusted_image = np.clip(image.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
    return adjusted_image

def increase_sharpness(image):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32)  # Sharpening kernel
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

def median_filtering(image):
    # Separate color channels
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    # Apply median filtering to each channel
    filtered_red_channel = medfilt2d(red_channel, kernel_size=3)
    filtered_green_channel = medfilt2d(green_channel, kernel_size=3)
    filtered_blue_channel = medfilt2d(blue_channel, kernel_size=3)

    # Combine filtered channels back into an image
    filtered_image_array = np.stack((filtered_red_channel, filtered_green_channel, filtered_blue_channel), axis=-1)

    return filtered_image_array

#contrast enhancement
def contrast_enhancement(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2HSV)  # Cast for OpenCV

    # Split channels (Hue, Saturation, Value)
    h, s, v = cv2.split(hsv)

    # Equalize the value channel
    v_eq = cv2.equalizeHist(v)

    # Combine channels back to HSV
    hsv_equalized = cv2.merge((h, s, v_eq))

    # Convert back to RGB
    image_equalized = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2BGR)

    return image_equalized
``` 
``` 
image_array = np.array(image)
preprocessing_img = contrast_enhancement(increase_sharpness(median_filtering(color_adjustment(image_array))))

plt.imshow(preprocessing_img)
``` 
Before Preprocessing

![img_3.png](figure/original_image.png)

After Preprocessing

![img.png](figure/preprocessed_image.png)

## Training
Load images from training, validation, and testing set
``` python
train_data_dir = "preprocessed/train"
valid_data_dir =  "preprocessed/valid"
test_data_dir = "preprocessed/test"

# Set image size for resizing
img_width, img_height = 224, 224

batch_size = 64

# Create data generators for training and validation sets
train_datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                             target_size=(img_width, img_height),
                                             batch_size=batch_size,
                                             class_mode='binary') 
                                             
valid_generator = validation_datagen.flow_from_directory(valid_data_dir,
                                                 target_size=(img_width, img_height),
                                                 batch_size=batch_size,
                                                 class_mode='binary')

test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                 target_size=(img_width, img_height),
                                                 batch_size=batch_size,
                                                 class_mode='binary')
```

Train the model using DenseNet201 as the base model and add the last layer with one neuron unit and activation Sigmoid for binary classification (Wildfire/ No wildfire).

``` python
base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  # number of classes

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the pre-trained base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

``` python
# Defining an Early Stopping and Model Checkpoints
early_stopping = EarlyStopping(monitor = 'val_accuracy',
                              patience = 3, mode = 'max',
                              restore_best_weights = True)

checkpoint = ModelCheckpoint('best_model.h5',
                            monitor = 'val_accuracy',
                            save_best_only = True)
```

``` python
# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),  # Adjust based on data size
    epochs=3,  # Adjust number of epochs
    validation_data=valid_generator,
    validation_steps=len(valid_generator),  # Adjust based on data size
    callbacks=[early_stopping, checkpoint]  # Add callbacks as needed
)
```
``` python
# Extract loss values from the history
loss = history.history['loss']
val_loss = history.history['val_loss']  # Extract validation loss values

# Plot loss
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')  # Plot validation loss
plt.title('Training and Validation Loss Over Epochs (DenseNet201)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

![img_1.png](figure/loss.png)

## Evaluation
``` python
preds = model.predict(test_generator)  
test_loss, test_acc = model.evaluate(test_generator) 
print('\nTest Loss: ', test_loss)
print('\nTest Accuracy: ', np.round(test_acc * 100), '%')
```
Test Loss:  0.087

Test Accuracy:  97.0 %

## Prediction
``` python
def get_prediction(model_name, image):
    if model_name == "DenseNet201":
        base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        # Add custom classification layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)  # number of classes
        # Create the model
        model = Model(inputs=base_model.input, outputs=predictions)
        model.load_weights('best_model_densenet201.h5')
        
    pred_prob = model.predict(image) 

    if pred_prob >= 0.5:
        prediction_result = "WildFire"
    else:
        prediction_result = "No WildFire"

    return pred_prob[0][0], prediction_result
```
```
prob, result = get_prediction("DenseNet201",img_data_batch)

print("The probability is ", prob)
print("The classification result is ", result)
```

The probability is  6.792417e-05

The classification result is  No WildFire
