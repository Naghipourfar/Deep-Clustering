import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

"""
    Created by Mohsen Naghipourfar on 9/9/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


def stacked_auto_encoder(n_features, n_latent):
    DROPOUT = 0.25
    input_layer = Input(shape=(n_features,))
    latent = Dense(64, activation='relu')(input_layer)

    latent = BatchNormalization()(latent)
    latent = Dropout(DROPOUT)(latent)
    latent = Dense(32, activation='relu')(latent)

    latent = BatchNormalization()(latent)
    latent = Dropout(DROPOUT)(latent)
    latent = Dense(16, activation='relu')(latent)

    latent = BatchNormalization()(latent)
    latent = Dropout(DROPOUT)(latent)
    latent = Dense(8, activation='relu')(latent)

    latent = BatchNormalization()(latent)
    latent = Dropout(DROPOUT)(latent)
    latent = Dense(n_latent, activation='relu', name="encoded")(latent)

    output_layer = Dense(n_features, activation='sigmoid')(latent)

    model = Model(input_layer, output_layer)
    model.compile(optimizer="nadam", loss="mape", metrics=["mse", "mae"])
    model.summary()
    return model


def load_data(data_path="../Data/3mermotif_na.csv"):
    data = pd.read_csv(data_path, index_col="icgc_sample_id")
    data = normalize(data.values, norm='max', axis=0)
    return np.array(data)


def plot_results(path="../Results/encoded.csv"):
    encoded_output = pd.read_csv(path, header=None)
    print(encoded_output.describe())
    x, y, z = encoded_output[0], encoded_output[1], encoded_output[2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, y, z, c='Red')
    ax.set_xlim(0, x.max())
    ax.set_ylim(0, y.max())
    ax.set_zlim(0, z.max())
    plt.show()


def plot_results_2d(path="../Results/encoded2D.csv"):
    encoded_output = pd.read_csv(path, header=None)
    print(encoded_output.describe())
    x, y = encoded_output[0], encoded_output[1]

    plt.plot(x, y, 'o')
    plt.show()


def main():
    data = load_data()
    print(data.shape)
    model = stacked_auto_encoder(data.shape[1], 2)

    x_train, x_test = train_test_split(data, test_size=0.25, shuffle=True)
    print(x_train.shape)
    print(x_test.shape)

    model.fit(
        x=x_train,
        y=x_train,
        epochs=500,
        batch_size=64,
        validation_data=(x_test, x_test),
        verbose=2
    )

    layer_name = "encoded"
    encoded_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    encoded_output = encoded_layer_model.predict(data)
    model.save("../Results/SAE2d.h5")
    np.savetxt(X=encoded_output, fname="../Results/" + "encoded2D.csv", delimiter=",")


if __name__ == '__main__':
    plot_results()
    plot_results_2d()
