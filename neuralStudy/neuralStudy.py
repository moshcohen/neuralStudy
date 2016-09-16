import numpy

x_sleep_study=numpy.array(([3,5],[5,1],[10,2]),dtype=float)
y_score=numpy.array(([75],[82],[93]),dtype=float)

x_sleep_study=x_sleep_study/numpy.amax(x_sleep_study,axis=0)
y_score=y_score/100



class Neural_Network(object):
    def __init__(self):
        self.inputLayerSize=2
        self.outputLayerSize=1
        self.hiddenLayerSize=3

        #weights
        self.W1=numpy.random.randn(self.inputLayerSize,
                                   self.hiddenLayerSize)
        self.W2=numpy.random.randn(self.hiddenLayerSize,
                                   self.outputLayerSize)

    def forward(self,X):
        #propagate inputs through network
        self.z2=numpy.matrix.dot(X,self.W1)
        self.a2=self.sigmoid(self.z2)
        self.z3=numpy.matrix.dot(self.a2,self.W2)
        yHat=self.sigmoid(self.z3)

        return yHat
    def sigmoid(self,z):
         #apply sigmoid activation function
         return 1/(1+numpy.exp(-z))

    def sigmoidPrime(self,z):
         #derivative
         return numpy.exp(-z)/((1+numpy.exp(-z))**2)
     
    def cost_func(self,y,yHat):
        return sum(0.5*(y-yHat)**2)

    def costFunctionPrime(self,X,y):
        self.yHat=self.forward(X)
        delta3=numpy.multiply(-(y-self.yHat),self.sigmoidPrime(self.z3))
        dJdW2=numpy.matrix.dot(self.a2.T,delta3)

        delta2=numpy.matrix.dot(delta3,self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1=numpy.matrix.dot(X.T,delta2)

        return dJdW1,dJdW2

nn=Neural_Network()
cost1=nn.cost_func(x_sleep_study,y_score)
dJdW1,dJdW2=nn.costFunctionPrime(x_sleep_study,y_score)
a=1
#arr=nn.forward(x_sleep_study)
#print(arr)
#print(y_score)
#print("cost:",cost_func(y_score,arr))

