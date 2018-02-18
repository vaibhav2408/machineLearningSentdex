'''
We're going to be working first with the MNIST dataset, 
which is a dataset that contains 60,000 training samples 
and 10,000 testing samples of hand-written and labeled 
digits, 0 through 9, so ten total "classes." I will 
note that this is a very small dataset in terms of what 
you would be working with in any realistic setting, 
but it should also be small enough to work on everyone's 
computers.

The MNIST dataset has the images, which we'll be 
working with as purely black and white, thresholded, 
images, of size 28 x 28, or 784 pixels total. Our 
features will be the pixel values for each pixel, 
thresholded. Either the pixel is "blank" (nothing there, a 0), 
r there is something there (1). Those are our features. 
We're going to attempt to just use this extremely 
rudimentary data, and predict the number we're 
looking at (a 0,1,2,3,4,5,6,7,8, or 9). We're 
hoping that our neural network will somehow create 
an inner-model of the relationships between pixels, 
and be able to look at new examples of digits and 
predict them to a high degree.
'''

'''
input > weights > hidden layer1(activation function) > weights > 
hidden layer 2(activation function) > weights > output later 
'''


'''
passing the data that goes straight through at the very end
is called "feed forward"
So compare the output to the intended output and see how close
it is. The way we compare it is using cost/loss function

Example of cost function - cross entropy

optimation function(optimizer) - tries to minizise the cost func
Ex of optimizer - AdamOptimizer, SGD, AdaGrad
Optimizer - goes backwards and maniplates the weights - This process is called backpropogation

===> feed forward + backprop = epoch 

