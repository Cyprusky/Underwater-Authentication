# Underwater-Authentication
Neural networks and autoenconders applied to the problem of authentication in underwater acoustic transmissions.
Code used to produce my Bachelor's thesis in Computer Engineering at University of Padua.

## Problem
The problem considered involved discriminating a legitimate transmitter from an impersonating attacker in an underwater acoustic network.
The scenario includes two transmitters: Alice (legitimate transmitter) and Bob (malicious transmitter). Both send messages to a receiver, which has to distinguish if the communication is coming from a trusted node or a transmitter that is pretending to be one.

## Solution
Acoustic transmissions are described using features extracted from the acoustic channel that semsibly vary with the position of the transmitter from the receiver.
The features considered are the channel gain (g), the channel delay (d) and the delay spread (s).
To achieve classification of transmissions, machine learning techniques are employed. 
Using neural networks we are able to efficiently classify transmissions, depending on the combination of features. In this case we assume that the receiver can be trained with both "good" and "bad" examples.
Using an autoencoder we model the scenario in which the receiver has only knowledge about what legitimate transmissions look like and thus, malicious transmissions will be everything that looks "unordinary".

To evaluate results we compute the accuracy of the models and we trace the DET curve, to see how the classification threshold influences false positives and false negatives.

