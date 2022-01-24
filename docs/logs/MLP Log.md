# MLP for Gesture Classification

Using Multilayer Perceptron (a simple feed-forward fully connected) for gesture classification is straightforward. 
I used 1 hidden layer size of 100. 
Just record the data outputted from the hand landmark model and label it. Then, feed the labeled data into MLP to train. I used the one from sklearn, since it is a small model with little data. Then, just save it as pickle and use it during hand tracking. 

Due to the fact some cases of hand may not be covered at the first time the data is collected, we can test the model and record the data that the model is struggling with (active learning). However, make sure the number of samples for each class are the same. 