import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import Helpers


def initializing_Weights(NumberOfHiddenLayers, NumberOfNeurons, bias, isPrintWeights=False):
    Weights = []
    NumberOfNeurons = NumberOfNeurons.split(',')
    NumberOfNeurons = np.array([int(m) for m in NumberOfNeurons])
    if len(NumberOfNeurons) == 1 and NumberOfHiddenLayers > 1:
        NumberOfNeurons = NumberOfNeurons * np.ones(NumberOfHiddenLayers, dtype=int)

    # initilizing the weights for Input layer
    W = []
    for j in range(NumberOfNeurons[0]):
        # Setting the Small random weights for each Neuron in the Layer i
        if (bias == 1):
            W.append(np.random.rand(6))
        else:
            W.append(np.random.rand(5))
    # Now W Contains the Weights of Layer i but in List DataType So we will convert it to a matrix
    W = np.array(W)
    # Then we will append this W to the Global "Weights" List
    Weights.append(W)

    # -----------------------------------------------
    # initializing Hidden layers
    for i in range(1, (NumberOfHiddenLayers)):
        # Creating Wi for layer i
        W = []
        for j in range(NumberOfNeurons[i]):
            # Setting the Small random weights for each Neuron in the Layer i
            if (bias == 1):
                W.append(np.random.rand(NumberOfNeurons[i-1]+1))
            else:
                W.append(np.random.rand(NumberOfNeurons[i-1]))
        # Now W Caontains the Weights of Layer i but in List DataType So we will convert it to a matrix
        W = np.array(W)
        # Then we will append this W to the Global "Weights" List
        Weights.append(W)

    # -----------------------------------------------
    # initializing output layer
    W = []
    for j in range(3):
        # Setting the Small random weights for each Neuron in the Layer i
        if (bias == 1):
            W.append(np.random.rand(NumberOfNeurons[NumberOfHiddenLayers-1]+1))
        else:
            W.append(np.random.rand(NumberOfNeurons[NumberOfHiddenLayers-1]))
    # Now W Caontains the Weights of Layer i but in List DataType So we will convert it to a matrix
    W = np.array(W)
    # Then we will append this W to the Global "Weights" List
    Weights.append(W)

    # Print weights
    if isPrintWeights:
        counter = 0
        for i in Weights:
            if counter == 0:
                print("Weights of Hidden Layer: ",
                      1, " (Connected to the Input)")
            elif counter == NumberOfHiddenLayers:
                print("Weights of Output Layer")
            else:
                print("Weights of Hidden Layer: ", counter+1)
            print(i)
            print('--------------------')
            counter += 1

    return Weights


def encode(Y):
    if Y == 1:
        return np.array([1, 0, 0])
    elif Y == 2:
        return np.array([0, 1, 0])
    else:
        return np.array([0, 0, 1])


def train_model(class1, class2, class3, NumberOfHiddenLayers, NumberOfNeurons, learningRate, epochs, bias, activationFn, isEarlyStop):

    # Creating Weights List wich contains all W for all Layers
    Weights = initializing_Weights(NumberOfHiddenLayers, NumberOfNeurons, bias, isPrintWeights=False)

    class1["Class"] = 1
    class2["Class"] = 2
    class3["Class"] = 3
    class1_train = class1.head(30)
    class2_train = class2.head(30)
    class3_train = class3.head(30)

    data_shffled = pd.concat([class1_train, class2_train, class3_train])
    data_shffled = data_shffled.sample(frac=1)

    X = data_shffled.iloc[:, :-1]
    Y = np.array(data_shffled["Class"])

    Training_Accuracy = 0

    for epoch in range(epochs):

        Accuracy_Counter = 0
        for i in range(len(X)):
            Z_outputOfAllLayers = []
            Y_Target_Vector = np.transpose([encode(Y[Accuracy_Counter])])

            if bias == 1:
                expanded_data = np.concatenate(list([[1, ], X.values[i]]))
                X_input = np.array([expanded_data])
            else:
                X_input = np.array([X.values[i]])

            X_input = np.transpose(X_input)
            # --------------------------------------------------------------
            # Feedforward Step
            # For the first Layer
            current_Neurons_Outputs = []
            if activationFn == 'Sigmoid':
                current_Neurons_Outputs = Helpers.sigmoid(np.dot(Weights[0], X_input))
            elif activationFn == 'Hyperbolic Tangent sigmoid':
                current_Neurons_Outputs = Helpers.tanh(np.dot(Weights[0], X_input))
            if bias == 1:
                current_Neurons_Outputs = np.concatenate((np.array([[1, ]]), current_Neurons_Outputs), axis=0)

            current_Neurons_Outputs = np.array(current_Neurons_Outputs)
            Z_outputOfAllLayers.append(current_Neurons_Outputs)

            # 1 Forward
            for j in range(1, NumberOfHiddenLayers+1):
                current_Neurons_Outputs = []
                if activationFn == 'Sigmoid':
                    current_Neurons_Outputs = Helpers.sigmoid(
                        np.dot(Weights[j], Z_outputOfAllLayers[j-1]))
                elif activationFn == 'Hyperbolic Tangent sigmoid':
                    current_Neurons_Outputs = Helpers.tanh(np.dot(Weights[j], Z_outputOfAllLayers[j-1]))
                if bias == 1 and j != NumberOfHiddenLayers:
                    current_Neurons_Outputs = np.concatenate((np.array([[1, ]]), current_Neurons_Outputs), axis=0)

                current_Neurons_Outputs = np.array(current_Neurons_Outputs)
                Z_outputOfAllLayers.append(current_Neurons_Outputs)

            # For the output Layer
            Y_output = current_Neurons_Outputs

            # ----------------------------------------------------------------------------------
            # 2 Backward step

            Segmas_for_all_Layers = []
            # Errors in output Layer (Y target is known)
            temp_segma = (Y_Target_Vector-Y_output) * Helpers.get_Derivative(Y_output, activationFn)
            Segmas_for_all_Layers.append(temp_segma)

            # Errors in Hidden Layers (Y target is known)
            counter = 0
            for k in reversed(range(0, NumberOfHiddenLayers)):
                # print(Segmas_for_all_Layers[counter].shape, Weights[k + 1].shape)

                temp_segma = (Segmas_for_all_Layers[counter] * Weights[k + 1])
                temp_segma = np.sum(temp_segma, axis=0)
                temp_segma = np.transpose(np.array(
                    [temp_segma])) * Helpers.get_Derivative(Z_outputOfAllLayers[k], activationFn)
                if bias == 1:
                    Segmas_for_all_Layers.append(temp_segma[1:])
                else:
                    Segmas_for_all_Layers.append(temp_segma)
                counter += 1

            # -----------------------------------------------------------
            # Update the weights
                # For The first layer
            reversed_Counter = NumberOfHiddenLayers
            Weights[0] = Weights[0] + learningRate * Segmas_for_all_Layers[reversed_Counter] * np.transpose(X_input)
            reversed_Counter -= 1

            # For all hidden layers
            for h in range(1, NumberOfHiddenLayers+1):
                Weights[h] = Weights[h] + learningRate * Segmas_for_all_Layers[reversed_Counter] * np.transpose(Z_outputOfAllLayers[h-1])
                reversed_Counter -= 1

            Accuracy_Counter += 1
        # ---------------------------------------- end --------------------------------------------------------------
        # Evaluating The model in the current epoch
        epoch_Training_Accuracy = 0
        if isEarlyStop == 1 or epoch == (epochs-1):
            Accuracy_Counter = 0
            for e in range(len(X)):
                Z_outputOfAllLayers = []
                Y_Target_Vector = np.transpose([encode(Y[Accuracy_Counter])])

                if bias == 1:
                    expanded_data = np.concatenate(list([[1, ], X.values[e]]))
                    X_input = np.array([expanded_data])
                else:
                    X_input = np.array([X.values[e]])

                X_input = np.transpose(X_input)
                # ------------------------------------------------
                # Feedforward Step
                # First Layer
                current_Neurons_Outputs = []
                if activationFn == 'Sigmoid':
                    current_Neurons_Outputs = Helpers.sigmoid(np.dot(Weights[0], X_input))
                elif activationFn == 'Hyperbolic Tangent sigmoid':
                    current_Neurons_Outputs = Helpers.tanh(np.dot(Weights[0], X_input))
                if bias == 1:
                    current_Neurons_Outputs = np.concatenate((np.array([[1, ]]), current_Neurons_Outputs), axis=0)

                current_Neurons_Outputs = np.array(current_Neurons_Outputs)
                Z_outputOfAllLayers.append(current_Neurons_Outputs)

                # For all hidden layer
                for eh in range(1, NumberOfHiddenLayers + 1):
                    current_Neurons_Outputs = []
                    if activationFn == 'Sigmoid':
                        current_Neurons_Outputs = Helpers.sigmoid(np.dot(Weights[eh], Z_outputOfAllLayers[eh - 1]))
                    elif activationFn == 'Hyperbolic Tangent sigmoid':
                        current_Neurons_Outputs = Helpers.tanh(np.dot(Weights[eh], Z_outputOfAllLayers[eh - 1]))
                    if bias == 1 and eh != NumberOfHiddenLayers:
                        current_Neurons_Outputs = np.concatenate((np.array([[1, ]]), current_Neurons_Outputs), axis=0)

                    current_Neurons_Outputs = np.array(current_Neurons_Outputs)
                    Z_outputOfAllLayers.append(current_Neurons_Outputs)
                Accuracy_Counter += 1
                # Output Layer
                Y_output = current_Neurons_Outputs
                if np.argmax(Y_output) == np.argmax(Y_Target_Vector):
                    epoch_Training_Accuracy += 1

            epoch_Training_Accuracy = (epoch_Training_Accuracy/len(X)) * 100
            Training_Accuracy = epoch_Training_Accuracy
            if epoch_Training_Accuracy == 100 and epoch != (epochs-1):
                print('Early Stopping at Epoch number: ', epoch)
                break

    confusionMatrix, Testing_Accuracy = test_model(
        class1[30:50], class2[30:50], class3[30:50], NumberOfHiddenLayers, Weights, activationFn, bias)

    print('----------------------------------------')
    print('Training model with (num of epochs = ', epochs, " ,Learning Rate = ",
          learningRate, " in ", NumberOfHiddenLayers, ' layers network).')
    print('Neurons on each Layer is: ', NumberOfNeurons)
    print('Used activation fn: ', activationFn)
    print('Training accuracy is: ', Training_Accuracy)
    print('Testing Accuracy: ', Testing_Accuracy, '\n')
    print('---- ---- ---- ----')
    print('Confusion Matrix: \n      C1  C2  C3 \n C1 ',
          confusionMatrix[0], '\n C2 ', confusionMatrix[1], '\n C3 ', confusionMatrix[2])
    print('---- ---- ---- ----')
    print('----------------------------------------')

    # draw confussion matrix


def test_model(class1, class2, class3, NumberOfHiddenLayers, Weights, activationFn, bias):
    # Confusion matrix variables
    ConfusionMatrix = np.zeros([3, 3])

    dataS = pd.concat([class1, class2, class3])
    X = dataS.iloc[:, :-1]
    # Target Y
    Y = np.array(dataS["Class"])

    # Evaluating The model
    Testing_Accuracy = 0
    Accuracy_Counter = 0
    for i in range(len(X)):
        Z_outputOfAllLayers = []
        Y_Target_Vector = np.transpose([encode(Y[Accuracy_Counter])])

        if bias == 1:
            expanded_data = np.concatenate(list([[1, ], X.values[i]]))
            X_input = np.array([expanded_data])
        else:
            X_input = np.array([X.values[i]])

        X_input = np.transpose(X_input)
        # ------------------------------------------------
        # Feedforward Step
        # For the first Layer
        List_of_Neurons_Output = []
        if activationFn == 'Sigmoid':
            List_of_Neurons_Output = Helpers.sigmoid(
                np.dot(Weights[0], X_input))
        elif activationFn == 'Hyperbolic Tangent sigmoid':
            List_of_Neurons_Output = Helpers.tanh(np.dot(Weights[0], X_input))
        if bias == 1:
            List_of_Neurons_Output = np.concatenate((np.array([[1, ]]), List_of_Neurons_Output), axis=0)

        List_of_Neurons_Output = np.array(List_of_Neurons_Output)
        Z_outputOfAllLayers.append(List_of_Neurons_Output)

        # For all hidden layer
        for h in range(1, NumberOfHiddenLayers + 1):
            List_of_Neurons_Output = []
            if activationFn == 'Sigmoid':
                List_of_Neurons_Output = Helpers.sigmoid(np.dot(Weights[h], Z_outputOfAllLayers[h - 1]))
            elif activationFn == 'Hyperbolic Tangent sigmoid':
                List_of_Neurons_Output = Helpers.tanh(np.dot(Weights[h], Z_outputOfAllLayers[h - 1]))
            if bias == 1 and h != NumberOfHiddenLayers:
                List_of_Neurons_Output = np.concatenate((np.array([[1, ]]), List_of_Neurons_Output), axis=0)

            List_of_Neurons_Output = np.array(List_of_Neurons_Output)
            Z_outputOfAllLayers.append(List_of_Neurons_Output)

        Accuracy_Counter += 1
        # For the output Layer
        Y_output = List_of_Neurons_Output

        if np.argmax(Y_output) == np.argmax(Y_Target_Vector):
            Testing_Accuracy += 1

        ConfusionMatrix[np.argmax(Y_output), np.argmax(Y_Target_Vector)] += 1

    Testing_Accuracy = (Testing_Accuracy / len(X)) * 100

    return ConfusionMatrix, Testing_Accuracy
