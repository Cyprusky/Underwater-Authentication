import pandas as pd
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt
import utils

'''
Neural network for binary classification of alice/eve transmissions using CH_GAIN
Alice is the positive class, Eve is the negative class
'''

#read in authentic data 
alice = pd.read_csv("dataset/bed2bed.csv", sep = ';')

#read in non-authentic data 
eve = pd.read_csv("dataset/surf2bed.csv", sep = ';')

#print info
#print(alice.info())
#print(eve.info())

#'alice' label set to 1
alice['type'] = 1

#'eve' label set to 0
eve['type'] = 0

#build complete dataset
dataset = alice.append(eve, ignore_index = True)

#specify the data (samples)
#CH_GAIN
X = dataset.iloc[:, 0]
X = np.asarray(X)
X = X.reshape(-1, 1)

#specify the target labels and flatten the array 
y=np.ravel(dataset.type)

#split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

#scale train and test set, normalizing with mean = 0 and variance = 1
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#neural network
model = keras.models.Sequential()
model.add(keras.layers.Dense(1, activation = 'relu', input_shape = (1,)))
model.add(keras.layers.Dense(1, activation = 'relu'))
model.add(keras.layers.Dense(1, activation = 'sigmoid'))

# Model config
model.get_config()

# List all weight tensors 
model.get_weights()

model.summary()

#set model losses and metrics
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
  
#train the model  
model.fit(X_train, y_train, epochs = 1, batch_size = 32, validation_split = 0.2, verbose = 1)

#predict on the test set
y_pred = model.predict(X_test)

score = model.evaluate(X_test,y_test,verbose=1)
print("\n")
print("Score: ",score,"\n")

#uncomment the following to compute false positive and false negative rates for each threshold
#and then plot the resulting DET curve
'''
#use predictions as threshold values
thresholds = np.copy(np.unique(y_pred))

fpr = []
fnr = []
print("Number of thresholds:", len(thresholds), "\n")
print("Computing false positive and false negative rates...")

#compute false positive and false negative rates
for t in thresholds:
    fp_r, fn_r = utils.fp_fn_rates(y_pred, y_test, t)
    fpr.append(fp_r)
    fnr.append(fn_r)

#plot the DET curve
utils.det(fpr, fnr)
plt.show()
'''