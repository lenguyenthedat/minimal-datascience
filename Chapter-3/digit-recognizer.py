import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sknn.mlp import Classifier, Layer, Convolution

goal = 'Label'
myid = 'ImageId'

df = pd.read_csv('./data/train-1000.csv')
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
train, test = df[df['is_train']==True], df[df['is_train']==False]

features = test.columns.tolist()
features.remove('is_train')
features.remove('Label')

# Neural Network, Stochastic Gradient Descent is sensitive to feature scaling, so it is highly recommended to scale your data.
scaler = StandardScaler()
for col in features:
    scaler.fit(list(train[col])+list(test[col]))
    train[col] = scaler.transform(train[col])
    test[col] = scaler.transform(test[col])

# Define Classifier
myNNClassifier = Classifier(
                    layers=[
                        Convolution("Rectifier", channels=16, kernel_shape=(5,5)),
                        Convolution("Rectifier", channels=16, kernel_shape=(5,5)),
                        Convolution("Rectifier", channels=16, kernel_shape=(5,5)),
                        Layer('Rectifier', units=100),
                        Layer('Softmax')],
                    learning_rate=0.01,
                    learning_rule='momentum',
                    learning_momentum=0.9,
                    batch_size=100,
                    valid_size=0.01,
                    n_stable=20,
                    n_iter=200,
                    verbose=True)

# Train
myNNClassifier.fit(np.array(train[list(features)]), train[goal])

# Evaluation
print 'Accuracy Score:'
print accuracy_score(test[goal].values,
    myNNClassifier.predict(np.array(test[features])))
