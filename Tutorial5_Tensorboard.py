# We learn to:
# 1 - Push an amount of data equal to the batch size to the model
# 2 - Feed the data to the Tensors
# 3 - Train the model
# 4 - Display the number of batches during the training.
# 5 - Save the model on the disk.
import tensorflow as tf
import numpy as np

# Create Data
# Random Array of 10000 rows and 5 columns

X_train = (np.random.sample((10000,5)))
y_train = (np.random.sample((10000,1)))

feature_columns = [ tf.feature_column.numeric_column('x', shape=X_train.shape[1:]) ]

DNN_reg = tf.estimator.DNNRegressor(feature_columns=feature_columns,
                                    model_dir='train/linreg', # Indicate where to store the log file
                                    hidden_units=[500, 300],
                                    optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1, l1_regularization_strength=0.001)
                                    )

# Train the estimator
train_input = tf.estimator.inputs.numpy_input_fn( x={"x": X_train}, y=y_train, shuffle=False, num_epochs=None)

DNN_reg.train(input_fn=train_input,steps=3000)


# After finishing open up terminal in the folder and run:
# tensorboard --logdir=.\train\linreg

# on the browser type http://localhost:6006

