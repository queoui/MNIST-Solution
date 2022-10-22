from array import array
import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix

def main():

    #experiment variables
    learn_rate = 0.1
    epochs = 50
    hidden = 100
    momentum_value = 0.9
    batch = 60000


    #load and preprocess data
    train_correct = 0
    test_correct = 0
    delta_out = np.zeros((10,1), np.float64)
    (X_train, Y_train), (X_test, Y_test ) = mnist.load_data()
    X_train, X_test = np.divide(X_train, 255), np.divide(X_test, 255)
    X_test = X_test.reshape(-1, 784)
    X_train = X_train.reshape(-1, 784)

    #train label targets
    labels_train = np.full((len(Y_train),10), 0.1, np.float32)
    labels_train[range(labels_train.shape[0]),Y_train] = 0.9

    # test label targets
    labels_test = np.full((len(Y_test),10), 0.1,  np.float32)
    labels_test[range(labels_test.shape[0]),Y_test] = 0.9


    #weight tables for input-?hidden, hidden->output, and their biases
    ih_weight = np.random.uniform(-0.05, 0.05, (hidden, 784))
    ho_weight = np.random.uniform(-0.05, 0.05, (10, hidden))
    bias_ih = np.random.uniform(-0.05, 0.05, (hidden, 1))
    bias_ho = np.random.uniform(-0.05, 0.05, (10, 1))


    # for data plotting
    train_accuracies, test_accuracies  = [] , []

    for epoch in range(epochs):

        sample_train = np.random.randint(0, X_train.shape[0])
        sample_test = np.random.randint(0, X_test.shape[0])

        # used for smaller batches 
        # X_train_batch = X_train[sample_train:sample_train+batch-batch]
        # labels_train_batch = labels_train[sample_train:sample_train+batch]
        
        # confusion matrix params
        confusion_vector_category = []
        confusion_vector_truth = []

        # train
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
            
            # correct amount
            train_correct += int(category == g_truth)


            # Backpropagation output -> hidden (cost function derivative)
            # Add momentum
            ho_weight += (momentum_value * (-learn_rate * delta_out @ np.transpose(h)))
            bias_ho += (momentum_value *(-learn_rate * delta_out))

            delta_out = out - truth_vector
            ho_weight += -learn_rate * delta_out @ np.transpose(h) 
            bias_ho += -learn_rate * delta_out



            # Backpropagation hidden -> input (activation function derivative)
            delta_h = np.transpose(ho_weight) @ delta_out * (h * (1 - h))
            ih_weight += -learn_rate * delta_h @ np.transpose(digit) 
            bias_ih += -learn_rate * delta_h 


        # test
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

            # correct
            test_correct += int(category == g_truth)    

            # append for confusion matrix
            confusion_vector_category.append(category)
            confusion_vector_truth.append(g_truth)



        # Calculate accuracies, plot graph, plot confusion matrix and print. 
        print(f"Epoch: {epoch}:   Train Acc.: {round((train_correct / batch) * 100, 2)}%  |  Test Acc: {round((test_correct / 10000) * 100, 2)}%")
        train_accuracies.append(train_correct / batch)
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


if __name__ == "__main__":
    main()