import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from numpy.random import seed
from ..utilities.config import DP

seed(DP['SEED'])
tf.set_random_seed(DP['SEED'])

def train_ffnn(X_tr, y_tr, model_name):
    [n, d] = X_tr.shape

    # Build a model
    model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(d,)),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2),
    tf.keras.layers.Softmax()
    ])

    # compute loss
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

    model.fit(X_tr, y_tr, epochs=300, class_weight={0:1.,1:3.})
    model.save(model_name)
    return None

def test_ffnn(X_t, y_t, model_name):
    model = tf.keras.models.load_model(model_name)
    model.evaluate(X_t,  y_t, verbose=2)
    y_p = model.predict_classes(X_t)
    print(y_p.shape)
    print(f'NN Accuracy: {accuracy_score(y_p, y_t)}')
    print(f'NN Precision: {precision_score(y_p, y_t)}')
    print(f'NN Recall: {recall_score(y_p, y_t)}')
    print(f'NN F1: {f1_score(y_p, y_t)}')
    return None
