# Perceptron Implementation

# Basic Theory

Perceptron is an algorithm that is inspired by the way neuron works. A perceptron receives inputs by a weight and checks either this weighted sum and input is greater or lower than a threshold. If the weighted sum is greater, a decision can be made.

Perceptron uses a term called bias, to adjust a fixed shift if there is a result that happens to be far from the origin. Because of this bias, the main formula for the activation score is :

The main process of the algorithm is to do weight update during each iteration. The update is done using the difference between the initial score and the weight vector and bias.

The diagram of Perceptron looks like this, where x1 until x5 means input and w1 until w5 means weight. The neuron will count the weighted sum (a), and outputs a decision based on it.

How Perceptron works will be shown in this example. Consider dataset below :

The default w0, w1, and w2 for this example is 1. The dataset will have this process shown in the table below, on the first epoch (the process rotates in the whole dataset) :

The process can be defined as below :
1. Determine the ya. This can be defined by doing a multiplication between the weights (w0, w1, w2) and the input vectors (x0, x1, x2)
2. Apply the sign function to ya.
3. Determine the Δw0, Δw1, and Δw2. This is done by doing a multiplication between the difference between the desired result and the real result and the vector.
4. Add the result of Δw0, Δw1, and Δw2. If there is no change, there is no value updated.
5. Continue this process until the Δw0, Δw1, and Δw2 are 0.

The result of this process will be shown below, where there are 3 epochs to ensure there is no more error on this process.

Based on the explanation above, the pseudocode of a binary classification will be shown below.

# Results

To make sure that the binary perceptron applies the same if one of the class are labelled “positive” or “negative”, the label of the classes will be swapped, and there will be a set of weights given at the end of the iterations.

## Class 1 and Class 2

Perceptron can successfully differ both class 1 and 2, since for both labels, the weight did not change until the end of the iterations. This fact also concludes that class 1 and 2 are linearly separable. The models successfully predict both class 1 and 2 with perfect accuracy.

The pictures below show how the misclassification of classes changes over time. The final change happens on the third epoch, and there are no more misclassifications after that.

## Class 2 and Class 3

Perceptron fails to differ both class 2 and 3, since the weight of all the elements keeps changing until the end of the iteration. This also concludes that class 2 and class 3 are not linearly separable. The models fail to predict the accuracy, giving a 0% accuracy for both class 2 and 3.

The pictures below show how the misclassification of classes changes over time. The final change happens on the third epoch, and there are no more misclassifications after that.

## Class 1 and Class 3

Perceptron can successfully differ both class 1 and 3, since for both labels, the weight did not change until the end of the iterations. This fact also concludes that class 1 and 3 are linearly separable. The models successfully predict both class 1 and 3 with perfect accuracy.

The pictures below show how the misclassification of classes changes over time. The final change happens on the third epoch, and there are no more misclassifications after that.

As a conclusion, class 2 and 3 are the most difficult to separate, since the data on both classes are not linearly separable.
