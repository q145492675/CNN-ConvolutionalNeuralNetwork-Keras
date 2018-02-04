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
  *  ```A=CNNTraining(X_train,Y_train,X_CrossValidation,Y_CrossValidation,X_test,Y_test,nb_classes=3,epoch=12,batch_size=4,ConvLay=3,ConvAct='relu',Filter=32,kernel_size=(3,3),pool_size=(2,2),Drop_perc=0.25,DensLay=2,DensUnit=128,DensAct='sigmoid',OutAct='softmax') //Python
  ```
  * `A`: The score of the lost_function value and accuracy after training.
  * `X_train,Y_train`: The training dataset.
  * `X_CrossValidation,Y_CrossValidation`:  The cross validation dataset.
  * `X_test,Y_test`: The test dataset.
  * `nb_classes`: How many label in your dataset [default number is 2].
  * `epoch`: How many epoch you want to training [default number is 10].    
  * `batch_size`: How many data you want to train each batch [default number is 128].
  * `ConvLay`: The number of 2D-Convolutional layer in the framework [default number is 3].
  * `ConvAct`: The activation function of Convolutional layer [default is `ReLU`]. 
  * `Filter`: How many Filter in each Convolutional layer [default number is 32].  
  * `kernel_size`: The size of each convolutional layer's convolutional window [default size is (3,3)].
   * `pool_size`: The size of each Pooling layer's window [default size is (2,2)]. 
   * `Drop_perc`: The percentage of data need to drop out in each drop layer [default number is 0.45].   
  * `DensLay`: The number of Full_Collected layer in the framework [default number is 2].
  * `DensUnit`: How many units in each Full_Collected layer [default number is 128]
  * `DensAct`: The activation function of each Full_Collected layer [default is `sigmoid`].
  * `OutAct`: The activation function of the output layer [default is `softmax`].
  
### Applendix
* Still working on it 
