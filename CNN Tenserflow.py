import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# -----------------------------
# 1) Device
# -----------------------------
print("TensorFlow version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# -----------------------------
# 2) Paths (garde les mêmes que ton PyTorch)
# -----------------------------
train_dir = "/home/kevin/Downloads/shose/train"
test_dir  = "/home/kevin/Downloads/shose/test"

# -----------------------------
# 3) Params
# -----------------------------
batch_size = 64
img_size = (224, 224)     # modèle reçoit 224x224
resize_256 = (256, 256)   # pour reproduire Resize(256,256) avant crop
epochs = 20
lr = 1e-3
seed = 123

# -----------------------------
# 4) Load dataset (équivalent ImageFolder)
#    label_mode="int" -> labels entiers, comme CrossEntropyLoss
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="int",
    image_size=img_size,     # <-- 224,224
    batch_size=batch_size,
    shuffle=True,
    seed=seed
)


val_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="int",
    image_size=img_size,     # test: Resize(224,224)
    batch_size=batch_size,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Number of classes:", num_classes)
print("Classes:", class_names)

# Perf
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)

# -----------------------------
# 5) Data augmentation (équivalent transforms.Compose train)
# PyTorch:
# - Resize(256,256)
# - RandomResizedCrop(224, scale=(0.85,1.0))
# - RandomHorizontalFlip(0.5)
# - ColorJitter(...)
#
# Keras:
# - random crop via RandomZoom + RandomCrop (approx)
# - flip + contrast/brightness
# -----------------------------
data_augmentation = keras.Sequential(
    [
        # images arrivent en 256x256, on simule "RandomResizedCrop(224, scale 0.85-1.0)"
        # Approche simple: zoom aléatoire puis crop à 224
        layers.RandomZoom(height_factor=(-0.15, 0.0), width_factor=(-0.15, 0.0)),
        layers.RandomCrop(img_size[0], img_size[1]),
        layers.RandomFlip("horizontal"),
        layers.RandomBrightness(factor=0.15),
        layers.RandomContrast(factor=0.15),
        # saturation jitter n'est pas direct; alternative possible via TF image ops personnalisées
    ],
    name="data_augmentation"
)

# Normalisation [0,255] -> [0,1]
normalizer = layers.Rescaling(1./255)

# -----------------------------
# 6) Model (équivalent SimpleCNN)
# PyTorch features:
# Conv(3->16)->ReLU->MaxPool
# Conv(16->32)->ReLU->MaxPool
# Conv(32->64)->ReLU->AdaptiveAvgPool2d(1)
# + FC: LazyLinear(128)->ReLU->Linear(num_classes)
# -----------------------------
inputs = keras.Input(shape=(img_size[0], img_size[1], 3))

x = data_augmentation(inputs)
x = normalizer(x)

x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
# équivalent AdaptiveAvgPool2d(1)
x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(num_classes)(x)  # logits

model = keras.Model(inputs, outputs, name="SimpleCNN")
model.summary()

# -----------------------------
# 7) Loss/Optimizer (CrossEntropyLoss + Adam)
# PyTorch CrossEntropyLoss = softmax + CE sur logits
# => SparseCategoricalCrossentropy(from_logits=True)
# -----------------------------
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
)

# -----------------------------
# 8) Train + Eval
# -----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# -----------------------------
# 9) Save / Load
# PyTorch: state_dict -> .pth
# TF: .keras ou SavedModel
# -----------------------------
save_path = "/home/kevin/Downloads/shos.keras"
model.save(save_path)

loaded_model = keras.models.load_model(save_path)

# -----------------------------
# 10) Prediction sur une image (équivalent PIL + transform + argmax)
# -----------------------------
img_path = "/home/kevin/Downloads/45e62.jpg"
img = keras.utils.load_img(img_path, target_size=img_size)
img_arr = keras.utils.img_to_array(img)          # (224,224,3) en float32 [0..255]
img_arr = np.expand_dims(img_arr, axis=0)        # (1,224,224,3)

logits = loaded_model.predict(img_arr, verbose=0)
pred = int(np.argmax(logits, axis=1)[0])

print("Prediction index:", pred)
print("Prediction class:", class_names[pred])
