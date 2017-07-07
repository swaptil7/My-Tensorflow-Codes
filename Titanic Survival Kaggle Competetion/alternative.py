import tensorflow as tf 
import os
import numpy as np
import pandas as pd

tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


df = pd.read_csv('train.csv')

COLUMNS = ["Pclass","SibSp","Parch","Age"]
# for i in COLUMNS:
df.Age[np.isnan(df["Age"])] = df["Age"].median()
df.Embarked[df["Embarked"].isnull()] = 'S'


training_dataset = df[0:601]
test_dataset = df[601:]
predict_dataset = pd.read_csv('test.csv')

sex = tf.contrib.layers.sparse_column_with_keys(
  column_name="Sex", keys=["female", "male"])

embarked_port = tf.contrib.layers.sparse_column_with_keys(
  column_name="Embarked", keys=['C', 'Q' , 'S'])

pclass = tf.contrib.layers.sparse_column_with_keys(
column_name="Pclass", keys=[1,2,3],dtype=tf.int64)

siblings_spouses = tf.contrib.layers.sparse_column_with_hash_bucket("SibSp", hash_bucket_size=100,dtype=tf.int64)
parents_children = tf.contrib.layers.sparse_column_with_hash_bucket("Parch", hash_bucket_size=100,dtype=tf.int64)

# sex = tf.contrib.layers.embedding_column(sex, dimension=1)
# embarked_port = tf.contrib.layers.embedding_column(embarked_port, dimension=1)
# pclass = tf.contrib.layers.embedding_column(pclass, dimension=1)
# siblings_spouses = tf.contrib.layers.embedding_column(siblings_spouses, dimension=8)
# parents_children = tf.contrib.layers.embedding_column(parents_children, dimension=8)


age = tf.contrib.layers.real_valued_column("Age")
# siblings_spouses = tf.contrib.layers.real_valued_column("SibSp")
# parents_children = tf.contrib.layers.real_valued_column("Parch")
fare = tf.contrib.layers.real_valued_column("Fare")


CATEGORICAL_COLUMNS = ["Sex","Embarked","Pclass","SibSp","Parch"]
CONTINUOUS_COLUMNS = ["Age"]
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
  label = tf.constant(df["Survived"].values)
  return feature_cols, label


def predict_input_func(df):
  t_continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}

  t_categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      dense_shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}

  t_feature_cols = t_continuous_cols.copy()
  t_feature_cols.update(t_categorical_cols)
  return t_feature_cols


# classifier = tf.contrib.learn.DNNClassifier(
#     feature_columns=[sex,pclass,embarked_port,siblings_spouses,parents_children,age],
#     hidden_units=[32],model_dir='/tmp/titanic5/fifth_m_16',optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1))
#     # optimizer=tf.train.ProximalAdagradOptimizer(
#     #   learning_rate=0.1,
#     #   l1_regularization_strength=0.001
#     # ))

classifier = tf.contrib.learn.SVM(example_id_column='Pclass',
    feature_columns=[sex,pclass,embarked_port,siblings_spouses,parents_children,age],model_dir='/tmp/titanic6/first',
    l2_regularization=10.0)

classifier.fit(input_fn = lambda: input_func(training_dataset), steps=5000) 

accuracy = classifier.evaluate(input_fn = lambda: input_func(test_dataset), steps=1)
print("\nTest Accuracy: {}\n".format(accuracy))

predictions = list(classifier.predict(input_fn = lambda: predict_input_func(predict_dataset)))
print("New Samples, Class Predictions:{}\n".format(predictions))

# predict_dataset['Survived'] = predictions
# predict_dataset.to_csv('resultsCfive.csv')