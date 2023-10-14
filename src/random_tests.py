import tensorflow as tf
import numpy as np

strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    model.compile(loss='mse', optimizer='sgd')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=np.float32)

batch_size_per_replica = 1
global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

train_dataset = tf.data.Dataset.from_tensor_slices((xs, ys)).shuffle(len(xs)).batch(global_batch_size)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

@tf.function
def train_step(inputs):
    features, labels = inputs

    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(labels, predictions))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

# Define the distributed training loop
@tf.function
def distributed_train_step(inputs):
    per_replica_losses = strategy.run(train_step, args=(inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


def train(epochs=2):
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for x in train_dataset:
            total_loss += distributed_train_step(x)
            num_batches += 1
        average_loss = total_loss / num_batches
        print("Epoch {}/{}: loss={}".format(epoch+1, epochs, average_loss))

def print_tf_info():
    print(tf.__version__)
    print(tf.test.is_built_with_cuda())
    gpus = tf.config.experimental.list_logical_devices('GPU')
    cpus = tf.config.experimental.list_logical_devices('CPU')
    print("GPUs:", gpus)
    print("CPUs:", cpus)

def main():
    print_tf_info()
    train(epochs=10)
    test_input = np.array([10.0], dtype=np.float32)
    print(model.predict(test_input))

    print("Done")

if __name__ == "__main__":
    main()
