## 1. Implementation:
	> The model contains 2 hidden layers, and an output layer. The hidden layers contain 14 and 15 neurons respectively, followed by an output layer of size 1 neuron.
	> forward_prop and backward_prop have been implemented as two seperate functions, later called in fit() function. 
	> Hidden layers use ReLU as the activation function and the output layer uses sigmoid.
	> The loss function used is Binary Cross Entropy as it is known to provide the best results for binary classification problems.
	> find_CE_Cost() computes the Cross Entropy loss. This function includes L2 regularisation and model is penalised for high weights
	> All activation functions and their respective derivatives are defined seperately outside the class.
	> L2 regularisation parameter (lambda, called as lam in our function) is an optional parameter in the model.fit() method, which has been set by default to 0 since for this specific model the 		regularisation does not enhance the performance much. 
	> All weights, biases, and delta weights and biases are stored as dictionaries for convenience.
	> feedforward calculates first, the value of S, which is  x*weights + bias. Next it applies the activation function and stores it in A.
	> backwardprop starts with the output layer, uses the derivatives, finds the values of delta weights/biases, and updates them accordingly upto the first layer.
    > Updation happens with decaying learning rate.
	> fit() runs feedforward and backwardprop 'epoch' number of times.
	> predict() is a simple feedforward() with X_test as the input.
	
	
## 2. List of hyperparameters:
	> Learning Rate
	> number of epochs
	> L2 regularisation  
	> Number of neurons
	> Number of layers are kept static thus not included in list of hyperparameters 
	
## 3. Key Features:
	> Decaying learning rate 
	> Usage of L2 regularisation.
	> Allows dynamic initialisation of layer sizes (number of neurons).
	> Uses dictionaries, thereby improving readabilty of code.
	
## 4. Even more features:
	> Decaying learning rate
	> L2 regularisation 
	These are features that are beyond the basics, that we have implemented.
	
## 5. Detailed steps to run your files:
	> run the following line on terminal with numpy, sklearn,pandas, and python3 installed
		python Assignment3_0063_0342.py 
	> The preprocessed dataset is stored in the 'data' folder in this zip file. 
	> The python code Assignment3_0063_0342.py is stored in the 'src' folder and uses the relative path of the preprocessed dataset in the 'data' folder.
	> The python file used for preprocessing the dataset- preprocessing_0063_0342.py, is stored in the 'src' folder in this zip.
	> The ORIGINAL dataset (before preprocessing) has NOT been attached in this zip.
	
 
