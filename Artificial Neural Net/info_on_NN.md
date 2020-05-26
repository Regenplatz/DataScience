#### Binary Classification (K=2):
- activation function of last layer: `sigmoid`
- number of neurons / units of last layer: 1
- choice of loss function: log_loss / binary_crossentropy
#   

#### Multiclass Classification (K>2):
- activation function of last layer: `softmax`
- number of neurons / units of last layer: K
- choice of loss function: categorical_crossentropy
- categorical (like 'one-hot encoded') target y
#   

#### Multilabel Classification (K>=2), e.g. a cat and a dog on one picture:
- activation function of last layer: `sigmoid`
- number of neurons / units of last layer: K
- choice of loss function: binary_crossentropy (e.g. dog or not, cat or not)
- categorical (like 'one-hot encoded') target y
#    

#### Regression:
- activation function of last layer: `linear`
- number of neurons / units of last layer: 1
- choice of loss function: mse
