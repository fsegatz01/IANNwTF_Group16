{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMvijSgntYPSWluDo5sMP0y",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/fsegatz01/IANNwTF_Group16/blob/main/project/library/my_models.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "id": "czK-Yenomr6J"
      },
      "outputs": [],
      "source": [
        "import tensorflow as tf\n",
        "from sklearn.linear_model import LinearRegression\n",
        "from keras import backend as K\n",
        "from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "############## CNN MODEL  #############\n",
        "def output_to_input(self, layer_number, data):\n",
        "      \n",
        "        self.get_layer_output = K.function([model.layers[0].input],\n",
        "                                          [model.layers[layer_number].output])\n",
        "        self.layer_output = self.get_layer_output([data])[0]\n",
        "        self.hidden_layer_outputs_flat = tf.keras.layers.Flatten()(self.layer_output)\n",
        "\n",
        "        return self.hidden_layer_outputs_flat"
      ],
      "metadata": {
        "id": "y8xtQ8suoTfb"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "############## ResNet50 MODEL  #############\n",
        "\n",
        "class ResNet50Model:\n",
        "    def __init__(self):\n",
        "        self.model = tf.keras.applications.resnet.ResNet50(\n",
        "            input_shape=(32,32,3), #.output_shapes()[0],\n",
        "            include_top=True,\n",
        "            weights=None,\n",
        "            classes=100\n",
        "        )\n",
        "        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)\n",
        "        self.loss = 'categorical_crossentropy'\n",
        "        self.metrics = ['categorical_accuracy']\n",
        "\n",
        "\n",
        "    def output_to_input(self, layer_number, data):\n",
        "      \n",
        "        self.get_layer_output = K.function([model.layers[0].input],\n",
        "                                          [model.layers[layer_number].output])\n",
        "        self.layer_output = self.get_layer_output([data])[0]\n",
        "        self.hidden_layer_outputs_flat = tf.keras.layers.Flatten()(self.layer_output)\n",
        "\n",
        "        return self.hidden_layer_outputs_flat"
      ],
      "metadata": {
        "id": "ROHXPcdxnGFv"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "############## Linear MODEL  #############\n",
        "class LinearModel:\n",
        "    def __init__(self):\n",
        "        self.model = LinearRegression()\n",
        "\n",
        "    def train(self, x_train, y_train):\n",
        "        self.model.fit(x_train, y_train)\n",
        "\n",
        "    def predict(self, x_test):\n",
        "        return self.model.predict(x_test)\n",
        "\n",
        "    "
      ],
      "metadata": {
        "id": "9ttdYtSxoUDk"
      },
      "execution_count": 6,
      "outputs": []
    }
  ]
}