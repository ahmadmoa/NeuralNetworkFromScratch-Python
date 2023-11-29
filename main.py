import tkinter as tk
from tkinter import ttk
import pandas as pd
from tkinter import messagebox
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import model
import preprocess as datapp



drybean_Data = pd.read_csv('Dry_Bean_Dataset.csv', skiprows=1, header=None, names=['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes', 'Class'])

Class1_Data, Class2_Data, Class3_Data = datapp.preprocess_dataSet(drybean_Data)

# Creating The GUI
window = tk.Tk()
window.geometry("600x500")
window.title(" Classifier")

availableFunctionsToSelect = ['Sigmoid', 'Hyperbolic Tangent sigmoid']
activationFuntion = tk.StringVar()

# create a combobox


def createComboBox(feature, values):
    combobox = ttk.Combobox(window, textvariable=feature)
    combobox['values'] = values
    combobox['state'] = 'readonly'
    combobox.pack(fill=tk.X, padx=50, pady=10)
    return combobox


def createLabel(text):
    label = ttk.Label(text=text)
    label.pack(fill=tk.X, padx=50, pady=10)
    return label


def createTextBox():
    textbox = tk.Entry(window)
    textbox.pack(fill=tk.X, padx=50, pady=10)
    return textbox

# create a Start Button


def startTraining_btn_Clicked():

    # Calling The function in other class to start training the model
    # print(1)
    model.train_model(Class1_Data, Class2_Data, Class3_Data,int(NumberOfHiddenLayers_textBox.get())
                      ,(NumberOfNeurons_textBox.get())
                      ,float(learningRate_textBox.get())
                      ,int(NumberOfEpochs_textBox.get()),int(isBias.get())
                      ,activationFuntion.get(), int(isEarlyStop.get()))

###########################################################################

# create Show Visualizations Button


def showVisualization_btn_Clicked():
    model.showVisualization(Class1_Data, Class2_Data, Class3_Data)

# GUI from creation -----------------------------------------------------------------------------------


# Number of hidden layers
NumberOfHiddenLayers_Label = createLabel("Enter number of hidden layers")
NumberOfHiddenLayers_textBox = createTextBox()

# Number Of Neurons in each hidden layer
NumberOfNeurons_Label = createLabel(
    "Enter number of neurons in each hidden layer")
NumberOfNeurons_textBox = createTextBox()

# learning rate
learningRateLabel = createLabel("Please Enter learning Rate")
learningRate_textBox = createTextBox()

# number of epochs
NumberOfEpochsLabel = createLabel("Please Enter Number Of Epochs")
NumberOfEpochs_textBox = createTextBox()


# Sigmoid or Hyperbolic
activationFuntionLabel = createLabel(
    "Choose to use Sigmoid or Hyperbolic Tangent sigmoid as the activation function")
activationFuntion_combo = createComboBox(
    activationFuntion, availableFunctionsToSelect)
activationFuntion_combo.bind('<<ComboboxSelected>>')

# create a Bias Check box
isBias = tk.BooleanVar()
ttk.Checkbutton(window, text='Bias',  # command=agreement_changed,variable=isBias,
                onvalue=True,
                offvalue=False,
                variable=isBias).place(x=60, y=400)

# create early stopping Check box
isEarlyStop = tk.BooleanVar()
ttk.Checkbutton(window, text='Early Stopping',  # command=agreement_changed,variable=isBias,
                onvalue=True,
                offvalue=False,
                variable=isEarlyStop).place(x=130, y=400)

startTraining_btn = tk.Button(
    window, text="Start The System", width=30, command=startTraining_btn_Clicked)
startTraining_btn.place(x=50, y=450)

showVisualization_btn = tk.Button(
    window, text="Show Visualizations", width=30, command=showVisualization_btn_Clicked)
showVisualization_btn.place(x=325, y=450)

window.mainloop()
# ----------------------------------------------------------------------------------------------------
