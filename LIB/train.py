import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

def train_species_classifier(data_dir="data"):
    # Data Augmentation & 20% Validation Split
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=15,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        data_dir, target_size=(224, 224), batch_size=32,
        class_mode='categorical', subset='training'
    )
    
    val_gen = datagen.flow_from_directory(
        data_dir, target_size=(224, 224), batch_size=32,
        class_mode='categorical', subset='validation'
    )

    num_classes = len(train_gen.class_indices)

    # Base Model (Pre-trained on 1,000 objects)
    base = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
    base.trainable = False 

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation='softmax') # Softmax for multi-species
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=10)
    model.save("species_model.h5")