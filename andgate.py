import numpy as np 
def step_function(x):
    return 1 if x>=0 else 0
class perception:
    def __init__(self,input_size,learning_rate=0.1):
        self.weights=np.zeros(input_size)
        self.bias=0
        self.lr=learning_rate
    def predict(self,x):
        summation=np.dot(x,self.weights)+self.bias
        return step_function(summation)
    def train(self,x,y,epochs=10):
            for epoch in range(epochs):
                print("\n Epoch:",(epoch+1))
                for i in range(len(x)):
                    y_pred=self.predict(x[i])
                    error=y[i]-y_pred
                    self.weights=self.lr*error*x[i]
                    self.bias=self.lr*error
                    print(f"input:{x[i]},predicted:{y_pred},error:{error},weights:{self.weights}")
x=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([0,0,0,1])
p=perception(input_size=2)
p.train(x,y)
print("\nfinal prediction\n")
for i in range(len(x)):
    print(f"{x[i]}->{p.predict(x[i])}")