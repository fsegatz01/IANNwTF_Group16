import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# task 2.1 - Loading the MNIST data set
train_ds, test_ds = tfds.load('mnist', split=['train', 'test'], as_supervised=True)

# task 2.2 - Setting up the data pipeline
def prepare_mnist_data(mnist):
    
    mnist = mnist.map(lambda img, target: (tf.reshape(img, (-1,)), target))  # flatten data into vector
    mnist = mnist.map(lambda img, target: (tf.cast(img, tf.float32), target))  # convert data from unit8 to float32
    mnist = mnist.map(lambda img, target: ((img / 128.) - 1., target))  # adjust values
    mnist = mnist.map(lambda img, target: (img, tf.one_hot(target, depth=10)))  # create one_hot targets

    mnist = mnist.shuffle(1000)
    mnist = mnist.batch(32)  # give 32 elements at once
    mnist = mnist.prefetch(20)  # prepare 20 datapoints

    return mnist


train_dataset = train_ds.apply(prepare_mnist_data)
test_dataset = test_ds.apply(prepare_mnist_data)
train_accuracy_aggregator = []

train_dataset = train_dataset.take(1000)
test_dataset = test_dataset.take(100)


# task 2.3 - Building a deep neural network with TensorFlow
class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(10, activation=tf.nn.softmax) # 10 units in output layer because of the numbers 0 to 9

    @tf.function
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)

        x = self.out(x)
        return x

### Hyperparameters
num_epochs = 10 # 10 epochs
learning_rate = 0.1 # learning rate 0.1

# Initialize lists
train_losses = []
test_losses = []
test_accuracies = []
train_accuracies = []

# Initialize the model.
model = MyModel()
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy() # categorical cross entropy loss
optimizer = tf.keras.optimizers.SGD(learning_rate) # optimizer SGD


# task 2.4 - Training the network
def train_step(model, input, target, loss_function, optimizer):
    
    with tf.GradientTape() as tape:
        prediction = model(input)
        loss = loss_function(target, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # including train accuracy
    sample_train_accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
    sample_train_accuracy = np.mean(sample_train_accuracy)

    train_accuracy_aggregator.append(np.mean(sample_train_accuracy))

    return loss, train_accuracy_aggregator


def test(model, test_data, loss_function):

    test_accuracy_aggregator = []
    test_loss_aggregator = []

    for (input, target) in test_data:
        prediction = model(input)
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

    return test_loss, test_accuracy


# testing before we begin
test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

# check how model performs on train data
train_loss, train_accuracy = test(model, train_dataset, cross_entropy_loss)
train_losses.append(train_loss)
train_accuracies.append(train_accuracy)

# training loop
def training_loop(num_epochs, model, train_dataset, test_dataset, cross_entropy_loss, optimizer, train_losses,
                  test_losses, test_accuracies, train_accuracies):
    
    for epoch in range(num_epochs):
        print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

        # training (and checking in with training)
        epoch_loss_agg = []
        for input, target in train_dataset:
            train_loss, train_accuracy = train_step(model, input, target, cross_entropy_loss, optimizer)
            epoch_loss_agg.append(train_loss)

        train_loss = tf.reduce_mean(train_loss)
        train_accuracy = tf.reduce_mean(train_accuracy)
        # track training loss and accuracy
        train_losses.append(tf.reduce_mean(epoch_loss_agg))
        train_accuracies.append(train_accuracy)
        # testing, so we can track accuracy and test loss
        test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)



    return train_losses, test_losses, test_accuracies, train_accuracies

# execute training loop
train_losses0, test_losses0, test_accuracies0, train_accuracies0 = training_loop(num_epochs, model, train_dataset, test_dataset, cross_entropy_loss, optimizer, train_losses, test_losses, test_accuracies, train_accuracies)

# task 2.5 - vizualization
def vizualize(train_losses, test_losses, test_accuracies, train_accuracies, title):
    plt.figure()
    line1, = plt.plot(train_losses)
    line2, = plt.plot(test_losses)
    line3, = plt.plot(test_accuracies)
    line4, = plt.plot(train_accuracies)
    plt.title(title)
    plt.xlabel("Training steps")
    plt.ylabel("Loss/Accuracy")
    plt.legend((line1, line2, line3, line4), ("training loss", "test loss", "test accuracy", "train accuracy"))
    plt.show()

# vizualize the losses and accuracies
vizualize(train_losses0, test_losses0, test_accuracies0, train_accuracies0, "Original")