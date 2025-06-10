import os
import gzip
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

def load_mnist(path, kind='t10k'):
    """
    Load MNIST/Fashion-MNIST from `path` using only the two t10k-*.gz files.
    """
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)
        images = images.reshape(len(labels), 784)

    return images, labels

if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # 1. 載入全部資料 (只有 t10k-*.gz)
    # -----------------------------------------------------------------------------
    base_dir = os.path.join(os.path.dirname(__file__), 'data', 'fashion')
    x_all, y_all = load_mnist(base_dir, kind='t10k')  # (10000, 784), uint8

    # reshape & normalize
    x_all = x_all.reshape(-1, 28, 28).astype('float32') / 255.0

    # -----------------------------------------------------------------------------
    # 2. 隨機打散並做 80/20 切分
    # -----------------------------------------------------------------------------
    np.random.seed(42)
    idx = np.arange(x_all.shape[0])
    np.random.shuffle(idx)

    split = int(0.8 * idx.shape[0])
    train_idx, test_idx = idx[:split], idx[split:]

    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_test,  y_test  = x_all[test_idx],  y_all[test_idx]

    print(f"Train samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")

    # -----------------------------------------------------------------------------
    # 3. 建立並訓練模型
    # -----------------------------------------------------------------------------
    model = Sequential([
        Flatten(input_shape=(28,28)),
        Dense(128, activation='relu'),
        Dense(64,  activation='relu'),
        Dense(10,  activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        verbose=2
    )

    # -----------------------------------------------------------------------------
    # 4. 最終評估
    # -----------------------------------------------------------------------------
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final test accuracy: {acc:.4f}")

    # -----------------------------------------------------------------------------
    # 5. 存檔： .h5 / .json / .npz
    # -----------------------------------------------------------------------------
    model.save("fashion_model.h5")
    with open("fashion_mnist.json", "w") as f:
        f.write(model.to_json())

    weights = {}
    for i, layer in enumerate(model.layers):
        w = layer.get_weights()
        if len(w) == 2:
            weights[f"dense_{i}_w"] = w[0]
            weights[f"dense_{i}_b"] = w[1]
    np.savez("fashion_mnist.npz", **weights)

    print("Saved: fashion_model.h5, fashion_mnist.json, fashion_mnist.npz")
