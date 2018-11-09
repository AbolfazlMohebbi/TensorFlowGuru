# Use the Boston dataset to predict the price of a house using TensorFlow estimator.

import pandas as pd
# from sklearn import datasets
import sklearn.datasets
import tensorflow as tf
import itertools

# ============ Import data using Pandas =================
COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio", "medv"]
training_set = pd.read_csv("boston_train/boston_train.csv", skipinitialspace=True,skiprows=1, names=COLUMNS)
test_set = pd.read_csv("boston_train/boston_test.csv", skipinitialspace=True,skiprows=1, names=COLUMNS)
prediction_set = pd.read_csv("boston_train/boston_predict.csv", skipinitialspace=True,skiprows=1, names=COLUMNS)

FEATURES = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio"]
LABEL = "medv"

feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]	# convert the numeric variables from strings

# ============ Create Linear Regressor =====================
estimator = tf.estimator.LinearRegressor(feature_columns=feature_cols, model_dir="boston_model/train")

# You need to feed the model many times so you define a function to repeat this process.
def get_input_fn(data_set, num_epochs=None, n_batch=128, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
        y=pd.Series(data_set[LABEL].values),
        batch_size=n_batch,
        num_epochs=num_epochs,
        shuffle=shuffle)


# ============== Train the model ==========================
estimator.train(input_fn=get_input_fn(training_set, num_epochs=None, n_batch = 128, shuffle=False), steps=1000)

# ============== Evaluate the model ==========================
ev = estimator.evaluate(input_fn=get_input_fn(test_set, num_epochs=1, n_batch = 128, shuffle=False))
loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))
training_set['medv'].describe()

# We are getting a loss of arround 3000.
# From the describe() we see that the average price for a house is 22 thousand,
# with a minimum price of 9 thousands and maximum of 50 thousand.
# The model makes a typical error of 3k dollars.


# ======================= Make a Prediction ============================
y = estimator.predict( input_fn=get_input_fn(prediction_set, num_epochs=1, n_batch = 128, shuffle=False))
predictions = list(p["predictions"] for p in itertools.islice(y, 6))
print("Predictions: {}".format(str(predictions)))

# ===================== Import data using Tensorflow itself ==========================
# df_train = "boston_train/boston_train.csv"
# df_eval = "boston_train/boston_test.csv"
#
# # Construct data structures
# COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio", "medv"]
# RECORDS_ALL = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
#
# # Import the data
# def input_fn(data_file, batch_size, num_epoch=None):
#     # Step 1
#     def parse_csv(value):
#         columns = tf.decode_csv(value, record_defaults=RECORDS_ALL)
#         features = dict(zip(COLUMNS, columns))
#         # labels = features.pop('median_house_value')
#         labels = features.pop('medv')
#         return features, labels
#
#     # Extract lines from input files using the dataset API.
#     dataset = (tf.data.TextLineDataset(data_file)  # Read text file
#                       .skip(1)  # Skip header row
#                       .map(parse_csv))
#
#     dataset = dataset.repeat(num_epoch)
#     dataset = dataset.batch(batch_size)
#     # Step 3
#     iterator = dataset.make_one_shot_iterator()
#     features, labels = iterator.get_next()
#     return features, labels
# ==========================================================================
