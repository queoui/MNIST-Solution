
from array import array
import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix

def main():

    #experiment variables
    learn_rate = 0.01
    epochs = 50
    hidden = 128
    momentum_value = 0.0
    batch = 1


    #load and preprocess data
    train_correct = 0
    test_correct = 0
    count = 0
    (X_train, Y_train), (X_test, Y_test ) = mnist.load_data()
    X_train, X_test = np.divide(X_train, 255), np.divide(X_test, 255)
    X_test = X_test.reshape(-1, 784)
    X_train = X_train.reshape(-1, 784)

    #label targets
    labels_train = np.zeros((len(Y_train),10), np.float32)
    labels_train[range(labels_train.shape[0]),Y_train] = 1

    labels_test = np.zeros((len(Y_test),10), np.float32)
    labels_test[range(labels_test.shape[0]),Y_test] = 1


    #weight tables
    ih_weight = np.random.uniform(-0.05, 0.05, (hidden, 784))
    ho_weight = np.random.uniform(-0.05, 0.05, (10, hidden))
    bias_ih = np.random.uniform(-0.05, 0.05, (hidden, 1))
    bias_ho = np.random.uniform(-0.05, 0.05, (10, 1))



    train_accuracies, test_accuracies  = [] , []
    for epoch in range(epochs):

        sample_train = np.random.randint(0, X_train.shape[0])
        sample_test = np.random.randint(0, X_test.shape[0])

        X_train_batch = X_train[sample_train:sample_train+batch]
        labels_train_batch = labels_train[sample_train:sample_train+batch]

        X_test_batch = X_test[sample_test:sample_test+batch]
        labels_test_batch = labels_test[sample_test:sample_test+batch]

        
        confusion_vector_category = []
        confusion_vector_truth = []

        for digit, truth_vector in zip(X_train, labels_train):
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
            train_correct += int(category == g_truth)

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

        for digit, truth_vector in zip(X_test, labels_test):
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
            test_correct += int(category == g_truth)    

            confusion_vector_category.append(category)
            confusion_vector_truth.append(g_truth)



        # Show accuracy for this epoch
        print(f"Epoch: {epoch}:   Train Acc.: {round((train_correct / 60000) * 100, 2)}%  |  Test Acc: {round((test_correct / 10000) * 100, 2)}%")
        train_accuracies.append(train_correct / 60000)
        train_correct = 0

        test_accuracies.append(test_correct / 10000)
        test_correct = 0

        cfm_t = np.array(confusion_vector_truth)
        cfm_c = np.array(confusion_vector_category)
        confusion = confusion_matrix(cfm_t, cfm_c)
    print(confusion)


    pyplot.ylim(0, 1)
    pyplot.xlim(0, epochs)
    pyplot.plot(train_accuracies)
    pyplot.plot(test_accuracies)
    pyplot.show()

# #Sigmoid funstion
# def sigmoid(x):
#     return 1/(np.exp(-x)+1)


# #Softmax
# def softmax(x):
#     exp_element=np.exp(x-x.max())
#     return exp_element/np.sum(exp_element,axis=0)


if __name__ == "__main__":
    main()