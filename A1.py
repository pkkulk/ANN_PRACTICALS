import matplotlib.pyplot as plt 
import numpy as np

x=np.linspace(-10,10,100)

def sigmoid(x):
   return 1 / (1 + np.exp(-x))

def  softmax(i):
   x1 = np.exp(i)
   return x1/np.sum(x1)

def tanh(x):
  return np.tanh(x)

def relu(x):
   return np.maximum(0,x)


plt.figure(figsize=(20,8))
plt.subplot(2,2,1)
plt.plot(x,sigmoid(x))
plt.title("sigmoid function")
plt.grid(True)


plt.subplot(2,2,2)
i = np.array([1,2,3,4,5])
plt.plot(i,softmax(i))
plt.title("softmax function")
plt.grid(True)


plt.subplot(2,2,3)
plt.plot(x,tanh(x))
plt.title("tanh function")
plt.grid(True)

plt.subplot(2,2,4)
plt.plot(x,relu(x))
plt.title("relu")
plt.grid(True)
plt.show()