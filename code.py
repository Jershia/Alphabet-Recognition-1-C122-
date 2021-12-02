{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Alphabet Recognition-1(C122 Project)",
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
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
        "<a href=\"https://colab.research.google.com/github/Jershia/Alphabet-Recognition-1-C122-/blob/main/Alphabet_Recognition_1(C122_Project).ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "pimUvK_SBntM"
      },
      "source": [
        "# Logistic Regression for Image Classification\n",
        "\n",
        "We have already seen how powerful openCV can be, and we also know how to use it now. ***CV stands for computer vision***.\n",
        "\n",
        "We also know how Logistic Regression could be used to separate different outcomes. We have learnt about binary logistic regression where the outcome is always in True or False, but what if it's not?\n",
        "\n",
        "Don't worry! We got you covered. Today, we are going to see how we can make our computer detect a hand-written number.\n",
        "\n",
        "\\\n",
        "To get started, let's first import all the libraries that we will require to do this."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "kKIbgm5j2eSv"
      },
      "source": [
        "import cv2\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.datasets import fetch_openml\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import accuracy_score"
      ],
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "FZAlR40H2htn",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "de6feed8-5e31-401f-8189-46787380c080"
      },
      "source": [
        "X = np.load('image.npz')['arr_0']\n",
        "y = pd.read_csv(\"labels.csv\")[\"labels\"]\n",
        "print(pd.Series(y).value_counts())\n",
        "classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', \"K\", \"L\", \"M\", \"N\", \"O\", \"P\", \"Q\", \"R\", \"S\", \"T\", \"U\", \"V\", \"W\", \"X\", \"Y\", \"Z\"]\n",
        "nclasses = len(classes)"
      ],
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "T    550\n",
            "F    550\n",
            "N    550\n",
            "K    550\n",
            "A    550\n",
            "C    550\n",
            "O    550\n",
            "R    550\n",
            "I    550\n",
            "V    550\n",
            "U    550\n",
            "P    550\n",
            "M    550\n",
            "Y    550\n",
            "S    550\n",
            "W    550\n",
            "E    550\n",
            "D    550\n",
            "L    550\n",
            "J    550\n",
            "X    550\n",
            "G    550\n",
            "H    550\n",
            "B    550\n",
            "Z    550\n",
            "Q    550\n",
            "Name: labels, dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "unYMDQdQ973W",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 260
        },
        "outputId": "ae23e6c2-6af7-4798-8ff0-0e01ebb77f26"
      },
      "source": [
        "samples_per_class = 5\n",
        "figure = plt.figure(figsize=(nclasses*2,(1+samples_per_class*2)))\n",
        "\n",
        "idx_cls = 0\n",
        "for cls in classes:\n",
        "  idxs = np.flatnonzero(y == cls)\n",
        "  idxs = np.random.choice(idxs, samples_per_class, replace=False)\n",
        "  i = 0\n",
        "  for idx in idxs:\n",
        "    plt_idx = i * nclasses + idx_cls + 1\n",
        "    p = plt.subplot(samples_per_class, nclasses, plt_idx);\n",
        "    p = sns.heatmap(np.reshape(X[idx], (22,30)), cmap=plt.cm.gray, \n",
        "             xticklabels=False, yticklabels=False, cbar=False);\n",
        "    p = plt.axis('off');\n",
        "    i += 1\n",
        "  idx_cls += 1"
      ],
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 3744x792 with 130 Axes>"
            ]
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "mBPSeO827tO7"
      },
      "source": [
        "Here, we are representing 5 samples of each of the labels. Let's break this down line by line.\n",
        "\n",
        "\\\n",
        "With the first line of code - We are saying that we want to display 5 samples per labels.\n",
        "\n",
        "\\\n",
        "Next, with `figure = plt.figure(figsize=(nclasses*2,(1+samples_per_class*2)))`, we are setting up the total size of the figure that we plotted above.\n",
        "\n",
        "\\\n",
        "Now we get into a loop where we are iterating over the classes that we created earlier, which was a list of all the labels (0, 1, 2, ..., 9)\n",
        "\n",
        "\\\n",
        "We are first filtering out all the indexes of the elements with value equal to our label with `idxs = np.flatnonzero(y == cls)` and then we are selecting any 5 random indexes with `np.random.choice(idxs, samples_per_class, replace=False)`\n",
        "\n",
        "\\\n",
        "We are then iterating over these random indexes for the given label. First, we are doing `plt_idx = i * nclasses + idx_cls + 1` to define the position of the given label. Here, the `idx_cls` acts as the column while `i` acts as the rows. For all the samples of label 0 that we are plotting, the column `idx_cls` would remain to be the same while the row `i` will change. This helps us form a grid of samples while plotting.\n",
        "\n",
        "\\\n",
        "Now, we have the index of the random sample of the label, but how are we plotting it? We are creating a heatmap to plot the image with \n",
        "\n",
        "`p = sns.heatmap(np.reshape(X[idx], (28,28)), cmap=plt.cm.gray, xticklabels=False, yticklabels=False, cbar=False);`\n",
        "\n",
        "Here, we are taking the index `idx` and fetching it's element from `X`, and we are reshaping it in a 28*28 grid. Remember, images here are represented as binary. Let's cross verify the same for `X`."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "sMs6DMfF0QTy",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "47f5d1bc-b706-4d33-a957-e499cd1c105d"
      },
      "source": [
        "print(len(X))\n",
        "print(len(X[0]))"
      ],
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "14300\n",
            "660\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6NCi2rBQyhhe"
      },
      "source": [
        "Here, we can see that `X` has 70,000 image data and each image has 784 pixels of data, which is equivalent to `28*28`.\n",
        "\n",
        "\\\n",
        "Now let's prepare the data by splitting it and train a Logistic Regression Model but before that, let's check the array of a particular image."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vWKhwPOdqznb",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "6c938300-4d09-49ac-bdca-99ed6ecaf804"
      },
      "source": [
        "print(X[0])\n",
        "print(y[0])"
      ],
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0. 255. 255. 255. 255. 255. 255. 255. 255.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0. 255. 255. 255. 255. 255. 255. 255. 255.\n",
            " 255.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0. 255. 255.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0. 255.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0. 255.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            " 255. 255. 255. 255.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0. 255. 255. 255. 255. 255.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0. 255. 255.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0. 255. 255.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0. 255. 255.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0. 255. 255.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0. 255. 255.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.\n",
            "   0.   0.]\n",
            "F\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "aMWZJy1w0r6a"
      },
      "source": [
        "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)\n",
        "\n",
        "#scaling the features\n",
        "X_train_scaled = X_train/255.0\n",
        "X_test_scaled = X_test/255.0"
      ],
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "MMN8Bt-E2kh_"
      },
      "source": [
        "Here, we are going to use a total of 10000 samples since computing 70,000 samples might take a lot of time. For this, we have split our data into 7500 for training and 2500 for testing.\n",
        "\n",
        "\\\n",
        "We then simply divide our training and testing data by 255 to scale it and have values between 0 and 1. Let's train a Logistic Regression model with this data."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "am6I9TAt2jez",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "72d3e0a4-4a08-44f3-b563-709b79803dcd"
      },
      "source": [
        "clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)"
      ],
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_sag.py:354: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge\n",
            "  ConvergenceWarning,\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Y1WTw8iH7Mee"
      },
      "source": [
        "Until now, we were dealing with binary logistic regressions, but here, we have 10 labels, 0 to 9. For this, we write `multi_class='multinomial'` to specify that this is a `multinomial` logistic regression.\n",
        "\n",
        "Generally, there is a solver involved in all the logistic regressions, and the default solver is `liblinear`, which is highly efficient for linear logistic regression. This is also efficient with binary logistic regressions that we learnt earlier. For `multinomial` logistic regression, `solver='saga'` is highly efficient. It works well with large number of samples and supports `multinomial` logistic regressions, like this one."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ozJFsxj96pam",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "08cec48d-ef85-4926-f867-fcfa05d93e38"
      },
      "source": [
        "y_pred = clf.predict(X_test_scaled)\n",
        "\n",
        "accuracy = accuracy_score(y_test, y_pred)\n",
        "print(accuracy)"
      ],
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0.9952\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "4vK7NZqP9qS8"
      },
      "source": [
        "We have an accuracy score of 99.52%. Looks like an extremely efficient model. Let's check how it's confusion matrix looks like -"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "gp13OIYP9oPz",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 606
        },
        "outputId": "46cb84bc-1dc9-41b4-86db-b05baac51aec"
      },
      "source": [
        "cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])\n",
        "\n",
        "p = plt.figure(figsize=(10,10));\n",
        "p = sns.heatmap(cm, annot=True, fmt=\"d\", cbar=False)"
      ],
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 720x720 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Xb2l-xkW-EG6"
      },
      "source": [
        "Here, we are creating an array with `cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])` and setting it's size to be 10*10.\n",
        "\n",
        "\\\n",
        "We are finally plotting the graph with sns, which is used to prettify the charts that we draw with matplotlib.\n",
        "\n",
        "\\\n",
        "Finally, it's time to integrate our code with our camera, so that our model can detect the digit we write on a piece of paper using the model we just trained.\n",
        "\n",
        "\\\n",
        "For that, we'll have to switch back to our devices and do the same."
      ]
    }
  ]
}
