import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from tensorflow.keras.applications import InceptionV3  # or other pre-trained models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# Define the parameters
input_shape = (128, 128, 3)  # Adjust for the specific pre-trained model
num_classes = 165
batch_size = 16
epochs = 15

# Create a data generator for training and validation data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

validation_generator = validation_datagen.flow_from_directory(
    'dataset/val',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

# Load the pre-trained model (e.g., InceptionV3) with weights pre-trained on ImageNet
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

# Add custom layers for fine-tuning
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
#model_checkpoint = ModelCheckpoint('165class.h5', save_best_only=True, monitor='val_accuracy')
#early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)

# Train the model
history = model.fit(train_generator, validation_data=validation_generator, epochs=epochs)

# Save the final model
model.save('165class.h5')

# Evaluate the model on the validation data
validation_loss, validation_accuracy = model.evaluate(validation_generator)
print(f'Validation Loss: {validation_loss:.4f}')
print(f'Validation Accuracy: {validation_accuracy:.4f}')

# Print classification report and confusion matrix
