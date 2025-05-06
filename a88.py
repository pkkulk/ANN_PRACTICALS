import numpy as np 

def relu(x):
    return np.maximum(0,x)

def softmax(x):
    exp = np.exp(x-np.max(x,axis=1,keepdims=True))
    return exp/np.sum(exp,axis=1,keepdims=True)

def cross_entropy(predictions, labels):
    return -np.mean(np.sum(labels * np.log(predictions + 1e-9), axis=1))

def one_hots(x, y_):
    one_hot_labels = np.zeros((len(x), y_))
    one_hot_labels[np.arange(len(x)), x] = 1
    return one_hot_labels



def derivative_relu(x):
    return (x > 0).astype("float")


input =np.array( [[0.1,0.2,0.7,0.0],
         [ 0.5,0.1,0.0,0.4],                                                                                                                       
         [0.3,0.8,0.1,0.5],            
         [0.9,0.3,0.6,0.2]]    )

y=np.array([0,1,2,1])
y_encoded=one_hots(y,3)
input_neurons = input.shape[1]
hidden_neurons = 100 
output_neurons = 3
epochs = 1000
learning_rate = 0.1

np.random.seed(42)
w1=np.random.rand(input_neurons,hidden_neurons)
b1=np.random.rand(1,hidden_neurons)
w2=np.random.rand(hidden_neurons,output_neurons)
b2=np.random.rand(1,output_neurons)

for epoch in range(epochs):
    hidden_input=np.dot(input,w1)+b1
    hidden_output=relu(hidden_input)

    outputer_input=np.dot(hidden_output,w2)+b2
    outputer_output=softmax(outputer_input)

    loss = cross_entropy(outputer_output, y_encoded)

    if epoch % 100 == 0 :
        print (f"loss=> {loss:.4f}")
    error = outputer_output - y_encoded

    dw2 = np.dot(hidden_output.T, error)
    db2 = np.sum(error, axis=0, keepdims=True)

    error_hidden = np.dot(error, w2.T) * derivative_relu(hidden_input)

    dw1 = np.dot(input.T, error_hidden)
    db1 = np.sum(error_hidden, axis=0, keepdims=True)

    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2

  





