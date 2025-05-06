import numpy as np
def to_ascii(x):
  ascii = ord(str(x))
  binary_string = format(ascii,'08b')
  binary_list=[]
  for bit in binary_string:
    binary_list.append(int(bit)) 
  return binary_list

x=[]
y=[]

for i in range(10):
  binary = to_ascii(i)
  x.append(binary)
  if i % 2 == 0 :
     y.append(0)
  else :
     y.append(1)

X = np.array(x)
Y =np.array(y)

w=np.zeros(8)
b=0
learning_rate=0.1

for epochs in range(0):
   for i in range(len(X)):
       inputs=X[i]
       target=Y[i]
       weighted_sum = np.dot(inputs,w)+b

       if weighted_sum >= 0 :
          prediction = 1 
       else:
          prediction = 0 

       error= target - prediction

       for j in range(len(w)):
          w[j] += learning_rate*error*inputs[j]
       b += learning_rate*error

def predict(inputss,W,b):
   weighted_sum = 0
   for i in range(len(W)):
      weighted_sum += inputss[i]*W[i]
   weighted_sum +=b 
   if weighted_sum >= 0:
      return 1 
   else :
      return 0

   
for i in range(10):
 input_bits=to_ascii(i)
 prediction = predict(input_bits, w, b)
 label = "Odd" if prediction == 1 else "Even"
 print(f"Digit {i} is predicted as: {label}")