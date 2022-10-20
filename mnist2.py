import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot

def main():

    #experiment variables
    learn_rate = 0.1
    nr_correct = 0
    epochs = 50
    hidden = 128
    momentum_value = 0.0
    batch = 1000


    #load and preprocess data
    (X_train, Y_train), (X_test, Y_test ) = mnist.load_data()
    X_train, X_test = np.divide(X_train, 255), np.divide(X_test, 255)
    X_test = X_test.reshape(-1, 784)
    X_train = X_train.reshape(-1, 784)

    #label targets
    labels = np.zeros((len(Y_train),10), np.float32)
    labels[range(labels.shape[0]),Y_train] = 1

    #weight tables
    ih_weight = np.random.uniform(-0.05, 0.05, (hidden, 784))
    ho_weight = np.random.uniform(-0.05, 0.05, (10, hidden))
    bias_ih = np.random.uniform(-0.05, 0.05, (hidden, 1))
    bias_ho = np.random.uniform(-0.05, 0.05, (10, 1))

    sample = np.random.randint(0, X_train.shape[0])

    X_train_batch = X_train[sample:sample+batch]
    labels_train_batch = labels[sample:sample+batch]

    accuracies, val_accuracies  = [] , []
    for epoch in range(epochs):
        for digit, truth_vector in zip(X_train_batch, labels_train_batch):
            #convert to vector
            digit.shape += (1,)
            truth_vector.shape += (1,)

            # Forward propagation input -> hidden
            h_pre = bias_ih + ih_weight @ digit
            h = 1 / (1 + np.exp(-h_pre))

            # Forward propagation hidden -> output
            out_pre = bias_ho + ho_weight @ h
            out = 1 / (1 + np.exp(-out_pre))
            category = np.argmax(out)
            g_truth = np.argmax(truth_vector)
            

            # Cost / Error calculation
            e = 1 / len(out) * np.sum((out - truth_vector) ** 2, axis=0)
            nr_correct += int(category == g_truth)

            # #compute accuracy and append to the plot
            # accuracy = nr_correct/count
            # pyplot.plot(accuracy)

            # Backpropagation output -> hidden (cost function derivative)
            delta_out = out - truth_vector
            ho_weight += -learn_rate * delta_out @ np.transpose(h) 
            ho_weight += (momentum_value * -learn_rate * delta_out @ np.transpose(h))

            bias_ho += -learn_rate * delta_out
            bias_ho += (momentum_value *(-learn_rate * delta_out))


            # Backpropagation hidden -> input (activation function derivative)
            delta_h = np.transpose(ho_weight) @ delta_out * (h * (1 - h))
            ih_weight += -learn_rate * delta_h @ np.transpose(digit) 
            ih_weight += (momentum_value *(-learn_rate * delta_h @ np.transpose(digit)))

            bias_ih += -learn_rate * delta_h 
            bias_ih += momentum_value * (-learn_rate * delta_h )

        # Show accuracy for this epoch
        # print(f"Acc: {round((nr_correct / X_train.shape[0]) * 100, 2)}%")
        print(f"Acc: {round((nr_correct / batch) * 100, 2)}%")
        nr_correct = 0
    #     pyplot.ylim(-0.1, 1.1)
    #     pyplot.xlim(0, epochs)
        
    # pyplot.show()

# #Sigmoid funstion
# def sigmoid(x):
#     return 1/(np.exp(-x)+1)


# #Softmax
# def softmax(x):
#     exp_element=np.exp(x-x.max())
#     return exp_element/np.sum(exp_element,axis=0)


if __name__ == "__main__":
    main()