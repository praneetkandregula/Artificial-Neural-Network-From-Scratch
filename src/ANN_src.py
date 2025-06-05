''' 
IMPLEMENTATION DETAILS AND HYPERPARAMETERS

Implementation:
	> The model contains 2 hidden layers, and an output layer. The hidden layers contain 14 and 15 neurons respectively, followed by an output layer of size 1 neuron.
	> forward_prop and backward_prop have been implemented as two seperate functions, later called in fit() function. 
	> Hidden layers use ReLU as the activation function and the output layer uses sigmoid.
	> The loss function used is Binary Cross Entropy as it is known to provide the best results for binary classification problems.
	> find_CE_Cost() computes the Cross Entropy loss. This function includes L2 regularisation and model is penalised for high weights
	> All activation functions and their respective derivatives are defined seperately outside the class.
	> L2 regularisation parameter (lambda, called as lam in our function) is an optional parameter in the model.fit() method, which has been set by default to 0 since for this specific model the regularisation does not enhance the performance much. 
	> All weights, biases, and delta weights and biases are stored as dictionaries for convenience.
	> feedforward calculates first, the value of S, which is  x*weights + bias. Next it applies the activation function and stores it in A.
	> backwardprop starts with the output layer, uses the derivatives, finds the values of delta weights/biases, and updates them accordingly upto the first layer.
    > Updation happens with decaying learning rate.
	> fit() runs feedforward and backwardprop 'epoch' number of times.
	> predict() is a simple feedforward() with X_test as the input.
	
Hyperparameters:
	> Learning Rate 
	> number of epochs
	> L2 regularisation  (default lam =0)
	> Number of neurons 
	> Number of layers are kept static thus not included in list of hyperparameters (Number of hidden layers set to 2)

'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing

#Uncomment the following commented lines and import attached preprocessed dataset if you want to work on google colab

# from google.colab import files
# uploaded = files.upload()

# import io
# df = pd.read_csv(io.BytesIO(uploaded['LBW_Dataset.csv']))

#importing preprocessed dataset
df = pd.read_csv('../data/preprocessed_LBW_Dataset.csv')

#defining activation functions and their derivatives
def relu(Z):
    R = np.maximum(0, Z)
    return R

def sigmoid(Z):
    S = 1 / (1 + np.exp(-1 * Z))
    return S

def relu_derivative(Z):
    Z[Z >= 0] = 1
    Z[Z < 0]  = 0
    return Z

def sigmoid_derivative(Z):
    SD = sigmoid(Z) * (1 - sigmoid(Z))
    return SD

#parsing preprocessed dataset to get X and Y 
def parseDataset(df):
    X = df.iloc[:,0:5]
    X1=df.iloc[:,6:14]
    X=pd.concat([X,X1], axis=1)
    Y = df['Result']
    Y = [[i] for i in Y]
    Y = np.array(Y)
    X = np.array(X.values)
    scaler = preprocessing.StandardScaler()

    # transform data
    X = scaler.fit_transform(X)
    return X,Y

#NN class
class model():
    
    #initialise required variables
    def __init__(self,n1,n2):
        
        self.Weights = {}
        self.n1=n1                #Number of neurons in hidden layer1
        self.n2=n2                #Number of neurons in hidden layer2
        self.bias = {}
        self.Act = {}             #Holds result of applying activatio function on recieved summation(S) at each layer
        self.S = {}               #Result of summation (x*weights + bias) at each layer
        self.m = 0
        self.lam = 0              #lambda, L2 regularisation constant
        self.learning_rate = 0.
        self.epochs = 0
        self.del_Act = {}
        self.del_S = {}
        self.del_weights = {}
        self.del_bias = {}
        self.cost = 0.

        return
    
    #forward propagation
    def feedForward(self, X):
        
        #intial value at input layer
        self.Act['0'] = X
        
        #take dot product of input layer and weights and add the bias
        #activation function applied next, Repeat for each layer with previous layer output
        self.S['1'] = np.dot(self.Weights['1'], self.Act['0']) + self.bias['1']
        self.Act['1'] = relu(self.S['1'])
                
        self.S['2'] = np.dot(self.Weights['2'], self.Act['1']) + self.bias['2']
        self.Act['2'] = relu(self.S['2'])
                
        self.S['3'] = np.dot(self.Weights['3'], self.Act['2']) + self.bias['3']
        self.Act['3'] = sigmoid(self.S['3'])
        return
    
    #Cross Entropy Loss function
    def find_CE_Cost(self, Y):
        self.cost = -1 * np.sum(np.multiply(Y, np.log(self.Act['3'])) + 
                           np.multiply(1 - Y, np.log(1 - self.Act['3']))) / self.m 
        
        #L2 regularisation, Penalise the model for having high-valued weights by increasing cost
        #lam set to 0 by default
        if self.lam != 0:
            reg = (self.lam / (2 * self.m))
            reg += np.sum(np.dot(self.Weights['1'], self.Weights['1'].T))
            reg += np.sum(np.dot(self.Weights['2'], self.Weights['2'].T))
            reg += np.sum(np.dot(self.Weights['3'], self.Weights['3'].T))
            
            self.cost += reg
        return
    
    #Backward propagation to change weights and biases
    def backwardProp(self, Y):
        
        #Go from output layer to layer 1, calculate the change required in weights and biases
        self.del_Act['3'] = -1 * (np.divide(Y, self.Act['3']) - np.divide(1 - Y, 1 - self.Act['3']))
        
        self.del_S['3'] = np.multiply(self.del_Act['3'], sigmoid_derivative(self.S['3']))
        self.del_weights['3'] = np.dot(self.del_S['3'], self.Act['2'].T) / self.m + (self.lam/self.m) * self.Weights['3']
        self.del_bias['3'] = np.sum(self.del_S['3'], axis = 1, keepdims = True) / self.m
        self.del_Act['2'] = np.dot(self.Weights['3'].T, self.del_S['3'])
            
        self.del_S['2'] = np.multiply(self.del_Act['2'], relu_derivative(self.S['2']))
        self.del_weights['2'] = np.dot(self.del_S['2'], self.Act['1'].T) / self.m + (self.lam/self.m) * self.Weights['2']
        self.del_bias['2'] = np.sum(self.del_S['2'], axis = 1, keepdims = True) / self.m
        self.del_Act['1'] = np.dot(self.Weights['2'].T, self.del_S['2'])
            
        self.del_S['1'] = np.multiply(self.del_Act['1'], relu_derivative(self.S['1']))
        self.del_weights['1'] = np.dot(self.del_S['1'], self.Act['0'].T) / self.m + (self.lam/self.m) * self.Weights['1']
        self.del_bias['1'] = np.sum(self.del_S['1'], axis = 1, keepdims = True) / self.m
        self.del_Act['0'] = np.dot(self.Weights['1'].T, self.del_S['1'])
        
        #update weights and biases with calculated delta values(del->delta)
        self.Weights['1'] = self.Weights['1'] - self.learning_rate * self.del_weights['1']
        self.bias['1'] = self.bias['1'] - self.learning_rate * self.del_bias['1']
        
        self.Weights['2'] = self.Weights['2'] - self.learning_rate * self.del_weights['2']
        self.bias['2'] = self.bias['2'] - self.learning_rate * self.del_bias['2']
        
        self.Weights['3'] = self.Weights['3'] - self.learning_rate * self.del_weights['3']
        self.bias['3'] = self.bias['3'] - self.learning_rate * self.del_bias['3']
    
        return
    
    #Training the model 
    def fit(self, X, Y, epochs, learning_rate, decay = True, lam = 0):
        
        self.m = Y.shape[1] 
        self.learning_rate = learning_rate
        self.epochs= epochs
        self.lam = lam
        
        # initialize weights and biases with random values
        np.random.seed(44)
        self.Weights['1'] = np.random.randn(self.n1, 13) * np.sqrt(2. / 13)
        self.bias['1'] = np.zeros((self.n1, 1))
        np.random.seed(44)  
        self.Weights['2'] = np.random.randn(self.n2, self.n1) * np.sqrt(2. / self.n1)
        self.bias['2'] = np.zeros((self.n2, 1))
        np.random.seed(44)
        self.Weights['3'] = np.random.randn(1, self.n2) * np.sqrt(2. / self.n2)
        self.bias['3'] = np.zeros((1, 1))
        print("\n")

        #calling feefForward and backProp in a loop 'epoch' number of times
        for i in range(epochs):
            
            self.feedForward(X)
            self.find_CE_Cost(Y)
            self.backwardProp(Y)
            
            #learning rate decay 

            #initialising decay values
            decay_iter = 5
            decay_rate = 0.9
            stop_decay_counter = 100
            
            #if decay set to true in fit, learning rate gradually reduces at the specified rate (at set iterations)
            if ((decay and stop_decay_counter > 0) and (i % decay_iter == 0)):
                self.learning_rate = decay_rate * self.learning_rate
                stop_decay_counter -= 1
                
            #print cost per epoch
            print('Epoch: {} Loss: {}'.format(i, self.cost))
        
        return

   
    def predict(self, X,):

        #simple feedforward 
        self.feedForward(X)
        preds = self.Act['3'] 
        return preds

    #Given CM function
    @staticmethod
    def CM(y_test,y_test_obs):
        for i in range(len(y_test_obs)):
            if(y_test_obs[i]>0.6):
                y_test_obs[i]=1
            else:
                y_test_obs[i]=0

        cm=[[0,0],[0,0]]
        fp=0
        fn=0
        tp=0
        tn=0

        for i in range(len(y_test)):
            if(y_test[i]==1 and y_test_obs[i]==1):
                tp=tp+1
            if(y_test[i]==0 and y_test_obs[i]==0):
                tn=tn+1
            if(y_test[i]==1 and y_test_obs[i]==0):
                fp=fp+1
            if(y_test[i]==0 and y_test_obs[i]==1):
                fn=fn+1
        cm[0][0]=tn
        cm[0][1]=fp
        cm[1][0]=fn
        cm[1][1]=tp

        p= tp/(tp+fp)
        r=tp/(tp+fn)
        f1=(2*p*r)/(p+r)
        a = (tn+tp)/(tn+tp+fn+fp)
        print("Confusion Matrix : ")
        print(cm)
        print("\n")
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")
        print(f"Accuracy: {a}")

#main function
if __name__ == "__main__":

    #Retrieve X,Y from preprocessed dataset
    X,Y=parseDataset(df)
    m = model(14,15)

    #Splitting X and Y into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1)
    y_train = y_train.reshape(1, y_train.shape[0])

    #calling fit to train the model on train split
    m.fit(X_train.T, y_train, epochs= 300, learning_rate = 0.15, lam=0)

    #printing Train accuracy 
    print("Accuracy for train")
    yhat2=m.predict(X_train.T) 
    y_train = y_train.reshape(y_train.shape[1],1) 
    yhat2=yhat2.reshape(yhat2.shape[1],1)
    m.CM(y_train,yhat2) 
    print("\n")

    #printing test accuracy
    print("Accuracy for test") 
    yhat=m.predict(X_test.T) 
    yhat=yhat.reshape(yhat.shape[1],1)
    m.CM(y_test,yhat)





