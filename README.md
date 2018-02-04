# ConvolutionalNeuralNetwork-Keras
This routine is implemented in Keras (Tensorflow as backend) 

## Describition
* Python 3.5 or higher version
* Keras
* Tensorflow

## Usage
* put `CNNFitting.py` into your works folder
* write the command : 
             `import CNNFitting` 
  in the top of your master script file
* Referencing the function :
  *  `A=CNNTraining(X_train,Y_train,X_CrossValidation,Y_CrossValidation,X_test,Y_test,epoch=12,batch_size=4,ConvLay=3,ConvAct='relu',Filter=32,kernel_size=(3,3),pool_size=(2,2),Drop_perc=0.25,DensLay=2,DensUnit=128,DensAct='sigmoid',OutAct='softmax')`
  * [Prameter]
    *A:The score of the lost_function value and accuracy after training.
    *X_train,Y_train: The training dataset
    *X_CrossValidation,Y_CrossValidation: The cross_Validation dataset
    *X_Test,Y_Test:The test dataset

### Applendix
* Still working on it 
