### Principles of Artificial Neural Nets
Basically, artificial neural networks (ANN) simulate biological brain work. The digital learning process is known as `Deep learning`, which requires various digital layers. Those contain (at least) one or (a lot) more artificial neurons / units that communicate with each other from one layer to the nearest.

In general, artificial neural nets consist of
- input layer
- hidden layer(s)
- output layer


#  
#### Artificial Neural Net (ANN)
Information is initially provided to the input layer with small random weights and biases (initialized with small positive number or 0). From there, it moves further through the hidden layer(s) straight to the output layer. This process is called `feed-forward` and requires dot product calculations and application of `activation functions`.

To find the best working feed-forward process you could play around with the following hyperparameters:
-  change number of neurons / units
- change number of layers
- user another activation function
 - for hidden features:  
    - ELU
    - Leaky ReLU
    - ReLU
    - tanh (hyperbolic tangent)
 - for output features:
    - linear (regression)
    - sigmoid (binary classification)
    - softmax (multiclass classification)
- use another loss function
    - log-loss (binary classification)
    - cross-entropy (multiclass classification)
    - MSE (regression)
    - hinge loss (penalize wrong classifications)
    - triplett loss (image classification)
- use an optimization algorithm
- use batch normalization (add 2 hyperparameters per layer to normalize the inputs to the activation function to a given scale/average &rightarrow; reduces training time, improves accuracy)

See also [info_on_NN](info_on_NN.md)


For deep learning, it is necessary to involve `backpropagation` subsequently, so that feed-forward can again be performed, alternating with backpropagation until the best evaluated suitable weights and biases for task of interest. Backpropagation is comparable to the gradient descent and is needed to calculate the degree of weight variation between two layers.


#  
#### Convolutional Neural Net (CNN)
CNNs process only a portion of input data at a time. Those 'sliding windows' represent spatial relationships between pixels. For data processing, not only one set of neurons is required so that `'feature maps' / 'filters'` are used. One convolutional layer contains several of those filters, that each of these finally specializes in a particulare feature of input (e.g. lines, curves, dots).
Retreived information of each layer is simplified by condensing the information via `Max Pooling` before entering the next one.

Convolutional layer hyperparameters to be varied are:
- convolutional kernel (size of sliding window)
- stride (step in which the sliding window moves)
- number of feature maps / filters
- kind of padding (applied on the borders of the input)

CNNs are used for images (2D), sometimes also for audio (1D) processing.

#  
#### Recurrent Neural Net (RNN)
RNNs learn to predict the next input from previous inputs. At each time step, a learning rule is applied and the input is fed forward. The most common RNN is *fully recurrent* meaning that the outputs of all neurons are connected to the inputs of all neurons. If simulating the lack of connections between neurons, weights could set to zero.

RNNs are advantages as they make it possible to process input of any length because the model size does not increase with increasing amount of input. Furthermore, weights are shared across time and historical information is taken into account for computational tasks. 
However, computations might be slow and no future input can be considered for the current state.

RNNs are mainly used in:
- sentiment classification
- language translation
- video classification

RNNs could be also used for image recognition.
