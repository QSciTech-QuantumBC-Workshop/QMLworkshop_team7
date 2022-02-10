
#General Imports
import numpy as np 
import matplotlib.pyplot as plt
import pylab as pl


#ScikitLearn Imports
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 

#Qiskit imports
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.kernels import QuantumKernel

import pandas as pd


def importDataset(filename):
    '''
    Parameters
    ----------
    filename : String
        Path of filename on your hardware.

    Returns
    -------
    Pandas Dataframe
        Dataset with features and a target.

    '''   
    return pd.read_csv(filename)
 
def print_kernel_matrix(kernel_matrix):
    '''
    Parameters
    ----------
    kernel_matrix : Numpy array
        Function prints the kernel matrix.

    Returns
    -------
    None.

    '''
    fig, axs = plt.subplots(1, figsize=(10, 5))
    axs.imshow(
        np.asmatrix(kernel_matrix), interpolation="nearest", origin="upper", cmap="Blues"
    )
    axs.set_title("Train kernel matrix")
    plt.show()
    
    return



#####Plot graph score, features, computing time, etc....



######Gaim access  

#load credentials
'''
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')

#get backend of quantum computer
ibmq_quito = provider.get_backend('ibmq_quito')
#plot_error_map(ibmq_jakarta)

'''
######################

#Import file
filename = "/Users/lee/Desktop/hot_weather_enjoyers_project/heart.csv"

#Read file using Pandas

df = pd.read_csv(filename)
infor = df.describe()
df2 = df[df.trestbps < infor.loc["mean", "trestbps"] + 3 * infor.loc["std", "trestbps"]]
df3 = df2[df.chol < infor.loc["mean", "chol"] + 3 * infor.loc["std", "chol"]]
df4 = df3[df.thalach > infor.loc["mean", "thalach"] - 3 * infor.loc["std", "thalach"]]
df_new = df4[df.oldpeak < infor.loc["mean", "oldpeak"] + 3 * infor.loc["std", "oldpeak"]]
df_new.head()


#Create new columns from categorical features
pd.set_option('display.max_columns', None)
df_new.cp = df_new.cp.map({0:"asymptomatic", 1: "typical angina", 2:"atypical angina", 3:"non-anginal pain"})
df_new = pd.get_dummies(df_new, columns=['cp'])
df_new.insert(16,'target',df_new.pop('target'))


#Split dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(df.iloc[115:215,:8], df.iloc[115:215,-1:], stratify=df.iloc[115:215,-1:], random_state=0)

#create circuit 
quantum_circuit = RawFeatureVector(8)

quantum_circuit = quantum_circuit.assign_parameters(x_train.iloc[0])

simulator = Aer.get_backend('qasm_simulator')
shots = 1000

#simulator or ibmq_jakarta
qinst = QuantumInstance(simulator, shots, 1420)

pauli_map = PauliFeatureMap(feature_dimension=8, reps=1, entanglement='linear')

#Building the kernel matrix
pauli_kernel = QuantumKernel(feature_map=pauli_map, quantum_instance=qinst)

train_matrix= pauli_kernel.evaluate(x_vec=x_train)
test_matrix = pauli_kernel.evaluate(x_vec=x_test, y_vec=x_train)


model = SVC(kernel= pauli_kernel.evaluate)

#Fiting the SVM model according to the given training data.
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print(f'Callable kernel classification test score: {score}')
