This is the first cnn model achieving 95% test accuracy
after trained on 7000 training set and tested on the rest.

The structure of the cnn model is as follows:
Preprocessing: Take log2 over all pixels.
Layer 1: conv by 2x2 filter with 10 channels.
Layer 2: conv by 2x2 filter with 20 channels.
Layer 3: full connected to 100 nodes.
Layer 4: full connected to 100 nodes with dropout rate 0.7
Layer 5: full connected to 4 nodes with dropout rate 0.7
Output Layer: softmax to probability
Loss function: cross entropy
Activation function: ReLU
