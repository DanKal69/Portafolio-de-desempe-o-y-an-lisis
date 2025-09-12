import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dataset de números del 0 al 4 en formato de matriz 3x3
cero = [1, 1, 1,
        1, 0, 1,
        1, 1, 1]

uno = [0, 1, 0,
       0, 1, 0,
       0, 1, 0]

dos = [1, 1, 0,
       0, 1, 0,
       0, 1, 1]

tres = [1, 1, 0,
        1, 1, 0,
        1, 1, 0]

cuatro = [1, 0, 1,
          1, 1, 1,
          0, 0, 1]

cinco =  [0, 1, 1,
          0, 1, 0,
          1, 1, 0]

y = [[0,0,0,0,0,1],
     [0,0,0,0,1,0], 
     [0,0,0,1,0,0],
     [0,0,1,0,0,0],
     [0,1,0,0,0,0],
     [1,0,0,0,0,0]]

# Convertir Dataset en un array de numpy
x = np.array([np.array(cero).reshape(1,9), 
              np.array(uno).reshape(1,9), 
              np.array(dos).reshape(1,9), 
              np.array(tres).reshape(1,9), 
              np.array(cuatro).reshape(1,9),
              np.array(cinco).reshape(1,9)])
y = np.array(y)

# ------- FUNCIÓN PARA AUMENTAR DATOS --------
def agregar_ruido(x, y, repeticiones=5, ruido=0.2):
    X_au, Y_au = [], []
    for i in range(len(x)):
        X_au.append(x[i])
        Y_au.append(y[i])
        for _ in range(repeticiones):
            ruido_sample = x[i] + np.random.normal(0, ruido, x[i].shape) # Agregar ruido gaussiano
            ruido_sample = np.clip(ruido_sample, 0, 1) # Mantener en rango [0,1]
            X_au.append(ruido_sample) # Agregar muestra ruidosa
            Y_au.append(y[i]) 
    return np.array(X_au), np.array(Y_au)

# Generar datos aumentados
x_total, y_total = agregar_ruido(x, y, repeticiones=40, ruido=0.3)
print("Tamaño dataset aumentado:", x_total.shape, y_total.shape)

# ------ SPLIT MANUAL -----
def train_val_test_split(X, Y, data_train=0.7, data_val=0.2, data_test=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    n_total = len(X)
    n_train = int(data_train * n_total)
    n_val = int(data_val * n_total)
    n_test = n_total - n_train - n_val
    X_train = X[:n_train]
    Y_train = Y[:n_train]
    X_val = X[n_train:n_train+n_val]
    Y_val = Y[n_train:n_train+n_val]
    X_test = X[n_train+n_val:]
    Y_test = Y[n_train+n_val:]
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

X_train, Y_train, X_val, Y_val, X_test, Y_test = train_val_test_split(x_total, y_total, data_train=0.7, data_val=0.2, data_test=0.1, seed=50)

print("Tamaño Train:", X_train.shape)
print("Tamaño Validation:", X_val.shape)
print("Tamaño Test:", X_test.shape)

# ------ RED NEURONAL ------
inputNeurons = 9
hiddenNeurons = 12
outputNeurons = 6

b1 = np.zeros((1, hiddenNeurons))
b2 = np.zeros((1, outputNeurons))

# Función de activación sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Definición de la función softmax para el uso de la capa de salida
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True) if x.ndim > 1 else e_x / e_x.sum()

# Crear FeedForward
def FeedForward(x,w1,b1,w2,b2):
    Z1 = x.dot(w1) + b1
    A1 = sigmoid(Z1)
    Z2 = A1.dot(w2) + b2
    A2 = softmax(Z2)
    return A1, A2

# Inicializar pesos
def weights(x,y):
    return np.random.randn(x,y)*0.1 

# Función de pérdida MSE (Mean Squared Error)
def MSE(out, Y):
    s = (np.square(out - Y))
    s = np.sum(s)/ len(Y)
    return s 

# ------- BACKPROPAGATION -------
def Backpropagation(x, y, w1, b1, w2, b2, alpha):
    # Capa oculta
    Z1 = x.dot(w1) + b1        
    A1 = sigmoid(Z1)
    # Capa de salida      
    Z2 = A1.dot(w2) + b2       
    A2 = softmax(Z2)      

    # Backward  
    error = y - A2                      
    deltaK = error * (A2 * (1 - A2))    
    deltaN = (deltaK.dot(w2.T)) * (A1 * (1 - A1))  

    # Gradientes
    w2_adj = A1.T.dot(deltaK)           
    w1_adj = x.T.dot(deltaN)            
    b2_adj = np.sum(deltaK, axis=0, keepdims=True)
    b1_adj = np.sum(deltaN, axis=0, keepdims=True)

    # Actualización del peso y bias
    w1 = w1 + alpha * w1_adj
    w2 = w2 + alpha * w2_adj
    b1 += alpha * b1_adj
    b2 += alpha * b2_adj

    return w1, b1, w2, b2

# ------ ENTRENAMIENTO -------
def train(X_train, Y_train, X_val, Y_val, w1, b1, w2, b2, alpha=0.1, epoch=30):
    acc_train = []
    acc_val = []
    loss_train = []
    for j in range(epoch):
        l = []
        # Entrenamiento  
        for i in range(len(X_train)):
            _, out = FeedForward(X_train[i], w1, b1, w2, b2)
            # Cálculo de pérdida
            l.append(MSE(out, Y_train[i]))
            # Backpropagation
            w1, b1, w2, b2 = Backpropagation(X_train[i], Y_train[i], w1, b1, w2, b2, alpha)

        # Pérdida por época
        loss_epoch = sum(l)/len(X_train)
        loss_train.append(loss_epoch)

        # Accuracy Train
        correct_train = 0
        for i in range(len(X_train)):
            _, out = FeedForward(X_train[i], w1, b1, w2, b2)
            if np.argmax(out) == np.argmax(Y_train[i]):
                correct_train += 1
        acc_train.append((correct_train / len(X_train)) * 100)

        # Accuracy Validation
        correct_val = 0
        for i in range(len(X_val)):
            _, out = FeedForward(X_val[i], w1, b1, w2, b2)
            if np.argmax(out) == np.argmax(Y_val[i]):
                correct_val += 1
        acc_val.append((correct_val / len(X_val)) * 100)

        print(f"Epoch {j+1}: Loss={loss_epoch:.4f}, TrainAcc={acc_train[-1]:.2f}%, ValAcc={acc_val[-1]:.2f}%")
    
    return acc_train, acc_val, loss_train, w1, b1, w2, b2

# Inicializar pesos
w1 = weights(inputNeurons, hiddenNeurons)
w2 = weights(hiddenNeurons, outputNeurons)

# Entrenar
acc_train, acc_val, loss_train, w1, b1, w2, b2 = train(X_train, Y_train, X_val, Y_val, w1, b1, w2, b2, alpha=0.1, epoch=40)

# Graficar
plt.plot(acc_train, label='Train Accuracy')
plt.plot(acc_val, label='Validation Accuracy')
plt.xlabel('Épocas')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

plt.plot(loss_train, label='Train Loss')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ------ TEST FINAL -------
print("\n------ TEST FINAL ------")
correct_test = 0
for i in range(len(X_test)):
    # Predicción
    _, out = FeedForward(X_test[i], w1, b1, w2, b2)
    pred = np.argmax(out)
    real = np.argmax(Y_test[i])
    print(f"Entrada {i}: Pred={pred}, Real={real}")
    if pred == real:
        correct_test += 1

accuracy_test = (correct_test / len(X_test)) * 100
print(f"\nAccuracy en test: {accuracy_test:.2f}%")

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ------ MATRIZ DE CONFUSIÓN ------
y_true = []
y_pred = []

for i in range(len(X_test)):
    _, out = FeedForward(X_test[i], w1, b1, w2, b2)
    pred = np.argmax(out)
    real = np.argmax(Y_test[i])
    y_true.append(real)
    y_pred.append(pred)

# Crear matriz de confusión
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2,3,4,5], yticklabels=[0,1,2,3,4,5])
plt.xlabel("Predicciones")
plt.ylabel("Reales")
plt.title("Matriz de Confusión en Test")
plt.show()

# ------ Métricas de clasificación -------
print(classification_report(y_true,y_pred))
