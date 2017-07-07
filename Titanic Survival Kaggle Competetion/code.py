import tensorflow as tf 
import os
import numpy as np
import pandas as pd

#-------------------logging verbosity--------------------
tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#--------------------------------------------------------

#-------------------load datasets------------------------
COLUMNS = ["Sex","Pclass","Embarked","SibSp","Parch","Age", "Fare","Survived"]
test_COLUMNS = ["Sex","Pclass","Embarked","SibSp","Parch","Age", "Fare"]
to_del = ["PassengerId","Name","Ticket","Cabin"]
training_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

original_test = pd.read_csv('test.csv')
#delete columns
for i in to_del:
	del training_dataset[i]
	del test_dataset[i]

# clean dataset
for i in COLUMNS:
    training_dataset = training_dataset[training_dataset[i].notnull()]

for i in test_COLUMNS:
    test_dataset = test_dataset[test_dataset[i].notnull()]
#---------------------------------------------------------

#-------------------feature columns-----------------------

#(3.2)--------------------------sparse columns with or without keys--------------------------
sex = tf.contrib.layers.sparse_column_with_keys(
  column_name="Sex", keys=["female", "male"])

embarked_port = tf.contrib.layers.sparse_column_with_keys(
  column_name="Embarked", keys=['C', 'Q' , 'S'])

# we have to convert sparse tensors into embedding column in dnn

sex = tf.contrib.layers.embedding_column(sex, dimension=8)
embarked_port = tf.contrib.layers.embedding_column(embarked_port, dimension=8)

# siblings_spouses = tf.contrib.layers.sparse_column_with_hash_bucket("SibSp", hash_bucket_size=100)
# parents_children = tf.contrib.layers.sparse_column_with_hash_bucket("Parch", hash_bucket_size=100)
# pclass = tf.contrib.layers.sparse_column_with_keys(column_name="Pclass")
#---------------------------------------------------------------------------------------

#(3.3)-------------------------real valued columns------------------------------------------
age = tf.contrib.layers.real_valued_column("Age")
fare = tf.contrib.layers.real_valued_column("Fare")
pclass = tf.contrib.layers.real_valued_column("Pclass")
siblings_spouses = tf.contrib.layers.real_valued_column("SibSp")
parents_children = tf.contrib.layers.real_valued_column("Parch")
#---------------------------------------------

CATEGORICAL_COLUMNS = ["Sex","Embarked"]
CONTINUOUS_COLUMNS = ["Age", "Fare","Pclass","SibSp","Parch"]

def input_func(df):
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}

  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      dense_shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}

  feature_cols = continuous_cols.copy()
  feature_cols.update(categorical_cols)
  print(feature_cols.keys())
  label = tf.constant(df["Survived"].values,dtype=tf.int8)
  return feature_cols, label 

def test_input_func(df):
  t_continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}

  t_categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      dense_shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}

  t_feature_cols = t_continuous_cols.copy()
  t_feature_cols.update(t_categorical_cols)
  return t_feature_cols

classifier = tf.contrib.learn.DNNClassifier(
    feature_columns=[sex,pclass,embarked_port,siblings_spouses,parents_children,age,fare],
    hidden_units=[64, 32],model_dir='/tmp/titanic/second')
    # optimizer=tf.train.ProximalAdagradOptimizer(
    #   learning_rate=0.1,
    #   l1_regularization_strength=0.001
    # ))

#(5.1)-----------------fit or train the model-----------------
# classifier.fit(input_fn = lambda: input_func(training_dataset), steps=2000) 
#use lambda functions for flexibility
#--------------------------------------------------------

#(5.2)-----------------Evaluate accuracy----------------------
# accuracy = classifier.evaluate(input_fn = lambda: input_func(test_dataset), steps=1)
# print("\nTest Accuracy: {}\n".format(accuracy))
#--------------------------------------------------------

# #(5.3)-----------------predict--------------------------------
predictions = list(classifier.predict(input_fn = lambda: test_input_func(original_test)))
print("New Samples, Class Predictions:{}\n".format(predictions))
original_test['Survived'] = predictions
original_test.to_csv('results_C.csv')