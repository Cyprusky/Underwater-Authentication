'''
Autoencoder for detecting fake underwater transmissions 
authentic messages -> label 0 (Alice)
fake messages -> label 1 (Eve)
'''
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, accuracy_score, precision_score

from numpy.random import seed
seed(1567)
tf.random.set_seed(2349)

#authentic transmissions 
alice_dataset = pd.read_csv("dataset/stat_channel_data_bed2bed_FA.csv", sep = ',')

#fake transmissions
eve_dataset = pd.read_csv("dataset/stat_channel_data_surf2bed_FA.csv", sep = ',')

#ordinary event -> negative label
alice_dataset['Class'] = 0

#anomaly -> positive label
eve_dataset['Class'] = 1

#build complete dataset
dataset = alice_dataset.append(eve_dataset, ignore_index = True)
dataset = dataset.drop(['SRC_INDEX', 'RX_INDEX'], axis=1)

print(dataset.info)
dataset = dataset.values
X = dataset[:, 0:4]
y = np.ravel(dataset[:,-1])

#split the data in train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 42)

#scale train and test set, mean = 0 variance = 1
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#divide training set and test set depending on label
y_train = y_train.astype(bool)
y_test = y_test.astype(bool)

X_train_alice = X_train[~y_train]
X_test_alice = X_test[~y_test]
X_train_eve = X_train[y_train]
X_test_eve = X_test[y_test]

print(X_train[y_train])
print("Samples in Alice train ",len(X_train_alice))
print("Samples in Eve train",len(X_train_eve))
print("Samples in Alice test",len(X_test_alice))
print("Samples in Eve test",len(X_test_eve))

#building the autoencoder 
input_dim = 4

hidden_dim = 3
#hidden_dim = 2

#input Layer
input_layer = tf.keras.layers.Input(shape=(input_dim, ))

#encoder
encoder = tf.keras.layers.Dense(hidden_dim, activation='relu')(input_layer)

#decoder
decoder = tf.keras.layers.Dense(input_dim, activation='linear')(encoder)

'''
#latent space dim = 2 and "intermidiate" layer of size 3
hidden_dim = 2

#input Layer
input_layer = tf.keras.layers.Input(shape=(input_dim, ))

#encoder
encoder = tf.keras.layers.Dense(3, activation='relu')(input_layer)
encoder = tf.keras.layers.Dense(hidden_dim, activation='relu')(encoder)

#decoder
decoder = tf.keras.layers.Dense(3, activation='relu')(encoder)
decoder = tf.keras.layers.Dense(input_dim, activation='linear')(decoder)
'''

autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)

autoencoder.summary()

autoencoder.compile(metrics=['accuracy'],
                    loss='mse',
                    optimizer='adam')

#autoencoder is trained only on authentic transmissions
history = autoencoder.fit(X_train_alice, X_train_alice,
                    epochs=10,
                    batch_size=1,
                    validation_split=0.10,
                    shuffle=True,
                    verbose = 1).history


#plot loss on training set and validation set to check overfitting
plt.plot(history['loss'], linewidth=2, label='Train loss')
plt.plot(history['val_loss'], linewidth=2, label='Validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

#making predictions on transmissions
predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse,
                        'True_class': y_test})

#write reconstruction losses on files
pred_alice = open("pred_alice.csv", "w")
pred_eve = open("pred_eve.csv", "w")

for i in range(len(error_df)):
    if error_df.iloc[i,1] == 1:
        pred_eve.write(str(error_df.iloc[i,0]))
        pred_eve.write("\n")
    else:
        pred_alice.write(str(error_df.iloc[i,0]))
        pred_alice.write("\n")
        
pred_alice.close()
pred_eve.close()


#plotting results using a fixed threshold to "have an idea of the performance"
threshold_fixed = 0.15 #set "intuitively" looking at the plot
groups = error_df.groupby('True_class')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Eve" if name == 1 else "Alice")

ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for Alice and Eve transmissions")
plt.ylabel("Reconstruction error")
plt.xlabel("Transmissions")
plt.show()

threshold_fixed = 0.15
pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
error_df['pred'] =pred_y

#print accuracy, precision and recall
print("\n")
print("Using fixed threshold ", threshold_fixed)
print("Accuracy: ",accuracy_score(error_df['True_class'], error_df['pred']))
print("Recall: ",recall_score(error_df['True_class'], error_df['pred']))
print("Precision: ",precision_score(error_df['True_class'], error_df['pred']))
                       
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         