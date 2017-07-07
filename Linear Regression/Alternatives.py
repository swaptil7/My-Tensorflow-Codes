import os
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#(1)-------------------logging verbosity--------------------
tf.logging.set_verbosity(tf.logging.INFO)
#--------------------------------------------------------
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#(2.1)-------------------load datasets------------------------
training_dataset = tf.contrib.learn.datasets.base.load_csv_without_header(filename="abalone_train.csv",
					target_dtype=np.int,features_dtype=np.float32)
test_dataset = tf.contrib.learn.datasets.base.load_csv_without_header(filename="abalone_test.csv",
					target_dtype=np.int,features_dtype=np.float32)
predict_dataset = tf.contrib.learn.datasets.base.load_csv_without_header(filename="abalone_predict.csv",
					target_dtype=np.int,features_dtype=np.float32)

#(2.2)-------------------read dataset by using pandas----------------------------------------

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)

#(2.3)-------------------How to create a label column---------------------------------------

LABEL_COLUMN = "label"
df_train[LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
df_test[LABEL_COLUMN] = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
#---------------------------------------------------------
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

#(3)xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#(3.1)-------------------feature columns-----------------------

no_of_feature_cols = training_dataset.data.shape[1]
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=no_of_feature_cols)]

#Two types of columns 
#How to make feature columns

CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]

#(3.2)--------------------------sparse columns with or without keys--------------------------
gender = tf.contrib.layers.sparse_column_with_keys(
  column_name="gender", keys=["Female", "Male"])

race = tf.contrib.layers.sparse_column_with_hash_bucket("race", hash_bucket_size=100)
marital_status = tf.contrib.layers.sparse_column_with_hash_bucket("marital_status", hash_bucket_size=100)
relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size=100)
workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size=100)
occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000)
native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=1000)
#---------------------------------------------------------------------------------------

#(3.3)-------------------------real valued columns------------------------------------------
age = tf.contrib.layers.real_valued_column("age")
education_num = tf.contrib.layers.real_valued_column("education_num")
capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")
#--------------------------------------------------------------------------------------

#(3.4)------------------bucketized column--------------------------------------------
if we want to learn the fine-grained correlation between income and each age group 
separately, we can leverage bucketization. Bucketization is a process of dividing 
the entire range of a continuous feature into a set of consecutive bins/buckets, 
and then converting the original numerical feature into a bucket ID 
(as a categorical feature) depending on which bucket that value falls into. So, we can 
define a bucketized_column over age as:

age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

where the boundaries is a list of bucket boundaries. 
In this case, there are 10 boundaries, resulting in 11 age group buckets 
(from age 17 and below, 18-24, 25-29, ..., to 65 and over).
#----------------------------------------------------------------------------------

#(3.5)----------------------Intersecting Multiple Columns with CrossedColumn--------------
Using each base feature column separately may not be enough to explain the data. For example, the correlation between education and the label (earning > 50,000 dollars) may be different for different occupations. Therefore, if we only learn a single model weight for education="Bachelors" and education="Masters", we won't be able to capture every single education-occupation combination (e.g. distinguishing between education="Bachelors" AND occupation="Exec-managerial" and education="Bachelors" AND occupation="Craft-repair"). To learn the differences between different feature combinations, we can add crossed feature columns to the model.

education_x_occupation = tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4))
We can also create a CrossedColumn over more than two columns. Each constituent column can be either a base feature column that is categorical (SparseColumn), a bucketized real-valued feature column (BucketizedColumn), or even another CrossColumn. Here's an example:

age_buckets_x_education_x_occupation = tf.contrib.layers.crossed_column(
  [age_buckets, education, occupation], hash_bucket_size=int(1e6))

#-------------------------------------------------------------------------------------

Note that feature columns are name and type of columns in feature set that we have
to pass to the estimator. So that it will know what types of columns it has and 
what type of tensors they contain e.g conti_column = real valued tensors such as 
constant tensor , categorical column = sparse tensors, crossed tensors etc.
so making of feature column is declaration of feature set type and data.

Also when we apply fit method to regressor or estimator, Now it knows the feature columns
types such as [gender is categorical,price is real valued] so feature column becomes 
feature_column = [gender(categorical),price(real valued),occedu(crossed),...]
Now to the fit method we have to pass values for respective column in feature column 
therefore an input function is used to pass these values consider following code for
input function for different columns 


Input function for both categorical and continuous columns

def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      dense_shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols.items() + categorical_cols.items())
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label 
#---------------------------------------------------------
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

#(4)xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#(4.1)------------------create Estimator-----------------------
regressor = tf.contrib.learn.LinearRegressor(feature_columns=feature_columns,model_dir="/tmp/abalone")

#pass created feature columns variables to regressor
model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.LinearClassifier(feature_columns=[
  gender, native_country, education, occupation, workclass, marital_status, race,
  age_buckets, education_x_occupation, age_buckets_x_education_x_occupation],
  optimizer=tf.train.FtrlOptimizer(
    learning_rate=0.1,
    l1_regularization_strength=1.0,
    l2_regularization_strength=1.0),
  model_dir=model_dir
#---------------------------------------------------------

#(4.2)------------------input functions-----------------------
def input_func(dataset):
	x = tf.constant(dataset.data) #feature set list to tensor
	y = tf.constant(dataset.target) #target labels list to tensor
	return x, y

#Input function for both categorical and continuous columns

'''def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      dense_shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols.items() + categorical_cols.items())
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label '''
#--------------------------------------------------------
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

#(5)xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#(5.1)-----------------fit or train the model-----------------
regressor.fit(input_fn = lambda: input_func(training_dataset), steps=2000) 
#use lambda functions for flexibility
#--------------------------------------------------------

#(5.2)-----------------Evaluate accuracy----------------------
accuracy = regressor.evaluate(input_fn = lambda: input_func(test_dataset), steps=1)
print("\nTest Accuracy: {}\n".format(accuracy))
#--------------------------------------------------------

#(5.3)-----------------predict--------------------------------
predictions = list(regressor.predict(input_fn = lambda: input_func(predict_dataset)))[0]
print("New Samples, Class Predictions:{}\n".format(predictions))
#--------------------------------------------------------
########################################################################################
