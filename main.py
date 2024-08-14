import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Cargar el modelo base
base_model = MobileNetV2(weights='imagenet', include_top=False)


# Añadir nuestras propias capas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x) # Capa completamente conectada
predictions = Dense(2, activation='softmax')(x) # Capa de salida para dos clases de colibríes

# Definir el modelo completo
model = Model(inputs=base_model.input, outputs=predictions)

# Congelar las capas del modelo base
for layer in base_model.layers:
    layer.trainable = False

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



# Directorio de datos de entrenamiento
train_data_dir = './dataset'

# Crear generadores de datos
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Imprimir índices de clases para ver la asignación
print(train_generator.class_indices)

# Entrenar el modelo
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=5
)

# Guardar el modelo entrenado
model.save('colibri_classifier_model.h5')

# Mapa de etiquetas de clase

class_labels = {0: 'Colibrí Inca ventrivioleta (Coeligena helianthea)', 1: 'Colibrí picoespada (Ensifera ensifera)'}

# Cargar y preprocesar la imagen para la predicción
img_path = 'picoespada80.jpg'  # Cambia esto al camino de la imagen que quieras predecir
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Predecir la clase
preds = model.predict(x)

# Obtener el índice de la clase con la mayor probabilidad
predicted_class_index = np.argmax(preds, axis=1)[0]

# Obtener la etiqueta de la clase predicha
predicted_class_label = class_labels[predicted_class_index]

print(f'Predicted class: {predicted_class_label}')