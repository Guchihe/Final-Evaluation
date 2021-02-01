# Final Evaluation
## Problem 2.

#### Loading the packages.


```python
from scipy import stats
import numpy as np
import pandas as pd 
import math
import matplotlib.pyplot as plt
from random import seed
```


```python
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
```

#### Making the entries of the logic table.


```python
x_train = np.array([0,1,0,1])                 
y_train = np.array([0,0,1,1])
amp_train = np.array([1,1,1,0])
doll_train = np.array([0,0,0,1])
```


```python
xy_train = np.append(x_train, y_train).reshape(len(x_train),2, order='F')
xy_train
```




    array([[0, 0],
           [1, 0],
           [0, 1],
           [1, 1]])



#### Defining the perceptron for the operator &.


```python
def amp_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['binary_accuracy'])
    return model
```


```python
ampmodel = amp_model()
```


```python
resultados_amp = ampmodel.fit(xy_train,  amp_train, epochs=1000,  verbose = 0)
```

#### Defining the percpectron with for opertator $.


```python
def doll_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['binary_accuracy'])
    return model
dollmodel = doll_model()
```


```python
resultados_doll = dollmodel.fit(xy_train,  doll_train, epochs=1000,  verbose = 0)
```

#### Now, we test the perceptrons with the following samples.


```python
A = np.array([1.001,0,0.001,1])                 
B = np.array([0,1,0,1])
C = np.array([0,1,1,0])
AampB =  np.append(A, B).reshape(len(A),2, order='F')
AampB
```




    array([[1.001e+00, 0.000e+00],
           [0.000e+00, 1.000e+00],
           [1.000e-03, 0.000e+00],
           [1.000e+00, 1.000e+00]])



#### First, calculate A&B. We can check that the result is correct.


```python
pred_AB = ampmodel.predict(AampB)
print(pred_AB)
```

    [[0.99189234]
     [0.99090075]
     [0.9891622 ]
     [0.01112247]]
    

#### Then, calculate (A&B)$C. Again, the result is correct.


```python
ABdollC = np.append(pred_AB, C).reshape(len(pred_AB),2, order='F')
pred_ABC = dollmodel.predict(ABdollC)
print(pred_ABC)
```

    [[0.01060218]
     [0.9840308 ]
     [0.9837887 ]
     [0.00830448]]
    


```python

```
