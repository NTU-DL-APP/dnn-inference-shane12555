# train.py
import os
import gzip
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization

def load_mnist(path, kind='t10k'):
    """
    Load MNIST/Fashion-MNIST from `path` using gzip files.
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
    # 1. 載入資料（只有 t10k-*.gz）
    base_dir = os.path.join(os.path.dirname(__file__), 'data', 'fashion')
    x_all, y_all = load_mnist(base_dir, kind='t10k')
    x_all = x_all.reshape(-1,28,28).astype('float32') / 255.0

    # 2. 隨機打散並做 80/20 切分
    np.random.seed(42)
    idx = np.arange(len(x_all))
    np.random.shuffle(idx)
    split = int(0.8 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]
    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_test,  y_test  = x_all[test_idx],  y_all[test_idx]
    print(f"Train: {len(x_train)} samples, Test: {len(x_test)} samples")

    # 3. 定義模型（含 BatchNormalization）
    model = Sequential([
        Flatten(input_shape=(28,28)),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 4. 訓練
    model.fit(
        x_train, y_train,
        epochs=40,
        batch_size=32,
        validation_split=0.1,
        verbose=2
    )

    # 5. 評估
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc*100:.2f}%")

    # 6. 匯出 .h5 / .json / .npz
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)

    # 6.1 存 h5 （備份用）
    model.save(model_dir / "fashion_mnist.h5")

    # 6.2 存 JSON 架構
    with open(model_dir / "fashion_mnist.json", "w") as f:
        f.write(model.to_json())

    # 6.3 存所有層的權重（含 BN 的 gamma/beta/mean/var）
    weights = {}
    for layer in model.layers:
        name = layer.name  # e.g. "dense", "batch_normalization"
        for i, w in enumerate(layer.get_weights()):
            # get_weights() 回傳 list：Dense 2 個、BN 4 個
            weights[f"{name}_{i}"] = w
    np.savez(model_dir / "fashion_mnist.npz", **weights)

    print("Saved to model/fashion_mnist.{h5,json,npz}")
