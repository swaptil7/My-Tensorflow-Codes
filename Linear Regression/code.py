import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys


#-------------------logging verbosity--------------------
tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#--------------------------------------------------------

#-------------------load datasets------------------------
training_dataset = tf.contrib.learn.datasets.base.load_csv_without_header(filename="abalone_train.csv",
					target_dtype=np.int,features_dtype=np.float32)
test_dataset = tf.contrib.learn.datasets.base.load_csv_without_header(filename="abalone_test.csv",
					target_dtype=np.int,features_dtype=np.float32)
predict_dataset = tf.contrib.learn.datasets.base.load_csv_without_header(filename="abalone_predict.csv",
					target_dtype=np.int,features_dtype=np.float32)
#---------------------------------------------------------

#-------------------feature columns-----------------------

no_of_feature_cols = training_dataset.data.shape[1]
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=no_of_feature_cols)]
#---------------------------------------------

#------------------create Estimator-----------------------
model_dir = "/tmp/abalone3/"+str(sys.argv[2])
regressor = tf.contrib.learn.LinearRegressor(feature_columns=feature_columns,optimizer=tf.train.GradientDescentOptimizer(learning_rate=float(sys.argv[1])),model_dir=model_dir)
#---------------------------------------------------------

#------------------input functions-----------------------
def input_func(dataset):
	x = tf.constant(dataset.data) #feature set list to tensor
	y = tf.constant(dataset.target) #target labels list to tensor
	return x, y
#--------------------------------------------------------

#-----------------fit or train the model-----------------
regressor.fit(input_fn = lambda: input_func(training_dataset), steps=2000)
#use lambda functions for flexibility
#--------------------------------------------------------

#-----------------Evaluate accuracy----------------------
accuracy = regressor.evaluate(input_fn = lambda: input_func(test_dataset), steps=1)
print("\nTest Accuracy: {}\n".format(accuracy))
#--------------------------------------------------------

#-----------------predict--------------------------------
predictions = list(regressor.predict(input_fn = lambda: input_func(predict_dataset)))
print("New Samples, Class Predictions:{}\n".format(predictions))
#--------------------------------------------------------
