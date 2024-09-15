# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

First we can take the dataset based on one input value and some mathematical calculus output value.Next define the neural network model in three layers.First layer has six neurons and second layer has four neurons,third layer has one neuron.The neural network model takes the input and produces the actual output using regression.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:SNEHA HV
### Register Number:212222040157
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # Space was missing between "models" and "import"
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default

auth.authenticate_user()

creds,_ = default()
gc = gspread.authorize (creds)
worksheet = gc.open('dataset ex1').sheet1
data = worksheet.get_all_values()

dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'INPUT': 'float'}) # Changed 'X' to '4' 
dataset1 = dataset1.astype({'OUTPUT': 'float'}) # Changed 'Y' to '9'
dataset1.head()

X = dataset1[['INPUT']].values
y = dataset1[['OUTPUT']].values
#X
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size = 0.33, random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)

X_train1 = Scaler.transform(X_train)
ai_brain = Sequential ([
Dense (8, activation = 'relu'),
Dense (10, activation = 'relu'),
Dense (1)
])
ai_brain.compile(optimizer = 'rmsprop', loss = 'mse')
ai_brain.fit(X_train1, y_train, epochs =10)

loss_df= pd.DataFrame(ai_brain.history.history)
loss_df.plot()

X_test1=Scaler.transform(X_test)
ai_brain.evaluate(X_test1, y_test)

X_n1=[[7]]
X_n1_1=Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)

```
## Dataset Information

![image](https://github.com/user-attachments/assets/80486d86-7cf6-4f8b-8448-4a30f6ada40a)


## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2024-09-15 114409](https://github.com/user-attachments/assets/3c54939e-dacf-4eb5-aab2-fb0da94436e8)


### Test Data Root Mean Squared Error

![image](https://github.com/user-attachments/assets/59d25619-b073-4496-9433-1262c91c59d9)
r

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/7a0d89bf-3871-4158-bb65-f862fdff23c9)


## RESULT

Include your result here
