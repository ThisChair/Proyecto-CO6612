import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical, plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras import backend
from keras import optimizers
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

parser = argparse.ArgumentParser(
    description='Training a classifier for Cleveland dataset.'
)

group = parser.add_mutually_exclusive_group()

group.add_argument("-v", "--verbose", action="store_true")

group.add_argument(
    '--mlp',
    action='store_true',
    help='Train a Multi-Layer Perceptron.'
)

group.add_argument(
    '--svm',
    action='store_true',
    help='Train a Support Vector Machine.'
)

parser.add_argument(
    '--matrix',
    action='store_true',
    help='Prints the confussion matrix.'
)

parser.add_argument(
    '--plot',
    action='store_true',
    help='Network topology.'
)

parser.add_argument(
    '--graphs',
    action='store_true',
    help='Plots loss and accuracy over epoch.'
)

parser.add_argument(
    '--adam',
    action='store_true',
    help='Uses Adam optimization method instead of SDG.'
)

parser.add_argument(
    '--norm',
    action='store_true',
    help='Normalizes by MinMax instead of standarization.'
)

parser.add_argument(
    '--epochs',
    type=int,
    help='Number of epochs. Default: 50.',
    default=50
)

parser.add_argument(
    '--eta',
    type=float,
    help='Learning rate. Default: 0.01.',
    default=0.01
)

parser.add_argument(
    '--hidden',
    type=int,
    help='Size of hidden layer. Default: 10.',
    default=10
)

parser.add_argument(
    '--momentum',
    type=float,
    help='Momentum. Default: 0.0.',
    default=0.
)

parser.add_argument(
    '--batch',
    type=int,
    help='Batch size. Default: 1.',
    default=1
)

parser.add_argument(
    '--test',
    type=float,
    help='Percentage of data  to use as test. Default: 0.2.',
    default=0.2
)

parser.add_argument(
    '--validation',
    type=float,
    help='Percentage of training data to use as validation. Default: 0.2.',
    default=0.2
)

parser.add_argument(
    '--kernel',
    type=str,
    choices=['linear','rbf','poly','sigmoid'],
    help='Kernel function used for transforming space. Default: linear.',
    default='linear'
)

parser.add_argument(
    '--gamma',
    type=str,
    choices=['auto','scale'],
    help='Kernel coefficient for rbf, poly and sigmoid. Default: auto (1/n).',
    default='auto'
)

parser.add_argument(
    '--coef',
    type=float,
    help='Independent term in poly and sigmoid. Default: 0.0.',
    default=0.0
)

parser.add_argument(
    '--degree',
    type=int,
    help='Degree of kernel function for poly. Default: 3.',
    default=3
)

parser.add_argument(
    '--c',
    type=float,
    help='Penalty parameter C. Default: 1.0.',
    default=1.0
)

parser.add_argument(
    '--layers',
    type=int,
    help='Number of hidden layers. Default: 1.',
    default=1
)

parser.add_argument(
    '--n',
    type=int,
    help='Number of tests to make. Default: 1.',
    default=1
)

args = parser.parse_args()

# Fix seed for rng
# np.random.seed(5)

# Load the dataset
dataframe = pd.read_csv("processed.cleveland.data", header=None)

# Replace ? for the mode, for each feature
mode = dataframe.mode().iloc[0]
dataframe = dataframe.replace(to_replace="?",value=mode)

# Create input and target data
dataset = dataframe.values
X = dataset[:,0:-1].astype(float)
Y = dataset[:,-1].astype(int)
dy = to_categorical(Y)


# Standarize input
scaler = StandardScaler().fit(X)
if args.norm:
    scaler = MinMaxScaler().fit(X)
X = scaler.transform(X)

if args.mlp:
    t_train_acc = []
    t_train_pre = []
    t_train_rec = []
    t_test_acc = []
    t_test_pre = []
    t_test_rec = []
    t_acc = []
    t_pre = []
    t_rec = []
    t_loss = []
    t_val_loss = []
    t_test_loss = []
    t_mat = []

    for _ in range(args.n):
        # Split data between test and training
        X_train, X_test, y_train, y_test = train_test_split(X,dy,test_size=args.test)

        # Create the MLP
        model = Sequential()
        model.add(Dense(args.hidden,input_dim=X.shape[1],activation='sigmoid'))
        for _ in range(args.layers - 1):
            model.add(Dense(args.hidden,activation='sigmoid'))
        model.add(Dense(dy.shape[1], activation='softmax'))
        optimizer = optimizers.SGD(lr=args.eta,momentum=args.momentum)
        if args.adam:
            optimizer = optimizers.Adam()
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        # Train the MLP
        history = model.fit(
            X_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch,
            validation_split=args.validation
        )

        t_loss.append(history.history['loss'][-1])
        t_val_loss.append(history.history['val_loss'][-1])

        # Evaluate test data in the MLP
        test_acc = model.evaluate(
            X_test,
            y_test,
            batch_size=args.batch,
            verbose=0
        )
        
        print("Test loss:", test_acc[0])
        t_test_loss.append(test_acc[0])
        print("Test accuracy:", test_acc[1])

        # Evaluate train data in the MLP
        train_pred = np.argmax(model.predict_on_batch(X_train),axis=-1)
        y_train = np.argmax(y_train,axis=-1)
        t_train_acc.append(accuracy_score(y_train,train_pred))
        t_train_pre.append(precision_score(y_train,train_pred,average='micro'))
        t_train_rec.append(recall_score(y_train,train_pred,average='micro'))

        # Evaluate test data in the MLP
        test_pred = np.argmax(model.predict_on_batch(X_test),axis=-1)
        y_test = np.argmax(y_test,axis=-1)
        t_test_acc.append(accuracy_score(y_test,test_pred))
        t_test_pre.append(precision_score(y_test,test_pred,average='micro'))
        t_test_rec.append(recall_score(y_test,test_pred,average='micro'))

        # Evaluate all data in MLP
        pred = np.argmax(model.predict_on_batch(X),axis=-1)
        t_acc.append(accuracy_score(Y,pred))
        t_pre.append(precision_score(Y,pred,average='micro'))
        t_rec.append(recall_score(Y,pred,average='micro'))

        t_mat.append(confusion_matrix(Y,pred))

        if args.matrix:
            print("Confusion matrix:")
            print(confusion_matrix(Y,pred))


    print("Mean results of {} tests:".format(args.n))
    train_loss = np.mean(t_loss)
    val_loss = np.mean(t_val_loss)
    test_loss = np.mean(t_test_loss)
    print("Train loss: {} | Validation loss: {} | Test loss: {}".format(
        train_loss,
        val_loss,
        test_loss
    ))
    train_acc = np.mean(t_train_acc)
    train_pre = np.mean(t_train_pre)
    train_rec = np.mean(t_train_rec)

    print(
        "Train accuracy: {} | Train precision: {} | Train recall: {}".format(
            train_acc,
            train_pre,
            train_rec
        )
    )

    test_acc = np.mean(t_test_acc)
    test_pre = np.mean(t_test_pre)
    test_rec = np.mean(t_test_rec)

    print(
        "Test accuracy: {} | Test precision: {} | Test recall: {}".format(
            test_acc,
            test_pre,
            test_rec
        )
    )

    acc = np.mean(t_acc)
    pre = np.mean(t_pre)
    rec = np.mean(t_rec)

    print(
        "Accuracy: {} | Precision: {} | Recall: {}".format(
            acc,
            pre,
            rec
        )
    )
    print("Confusion matrix:")
    mat = np.mean(np.array(t_mat),axis=0)
    print(mat)



    if args.plot:
        name = 'graphs/model-{}_layers-{}_neurons.png'.format(
            args.layers,
            args.hidden
        )
        plot_model(model, to_file=name, show_shapes=True, show_layer_names=False)

    if args.graphs:
        name = ''
        if args.adam:
            name = 'graphs/acc-{}-adam-{}_layers-{}_neurons.png'.format(
                args.epochs,
                args.layers,
                args.hidden
            )
        else:
            name = 'graphs/acc-{}-eta_{}-mom_{}-{}_layers-{}_neurons.png'.format(
                args.epochs,
                args.eta,
                args.momentum,
                args.layers,
                args.hidden
            )
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(name)
        plt.clf()


        if args.adam:
            name = 'graphs/loss-{}-adam-{}_layers-{}_neurons.png'.format(
                args.epochs,
                args.layers,
                args.hidden
            )
        else:
            name = 'graphs/loss-{}-eta_{}-{}_layers-{}_neurons.png'.format(
                args.epochs,
                args.eta,
                args.layers,
                args.hidden
            )
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(name)
        plt.clf()

if args.svm:
    t_train_acc = []
    t_train_pre = []
    t_train_rec = []
    t_test_acc = []
    t_test_pre = []
    t_test_rec = []
    t_acc = []
    t_pre = []
    t_rec = []
    t_mat = []

    for _ in range(args.n):
        # Split data between test and training
        X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=args.test)


        # Create the SVM
        model = SVC(
            kernel=args.kernel,
            C=args.c,
            degree=args.degree,
            gamma=args.gamma,
            coef0=args.coef
        )
        
        # Train the SVM
        model.fit(X_train,y_train)

        # Evaluate train data in the SVM
        train_pred = model.predict(X_train)
        t_train_acc.append(accuracy_score(y_train,train_pred))
        t_train_pre.append(precision_score(y_train,train_pred,average='micro'))
        t_train_rec.append(recall_score(y_train,train_pred,average='micro'))

        # Evaluate test data in the SVM
        test_pred = model.predict(X_test)
        t_test_acc.append(accuracy_score(y_test,test_pred))
        t_test_pre.append(precision_score(y_test,test_pred,average='micro'))
        t_test_rec.append(recall_score(y_test,test_pred,average='micro'))

        # Evaluate all data in SVM
        pred = model.predict(X)
        t_acc.append(accuracy_score(Y,pred))
        t_pre.append(precision_score(Y,pred,average='micro'))
        t_rec.append(recall_score(Y,pred,average='micro'))

        t_mat.append(confusion_matrix(Y,pred))
        if args.matrix:
            print("Confussion matrix:")
            print(confusion_matrix(Y,pred))

    print("Mean results of {} tests:".format(args.n))

    train_acc = np.mean(t_train_acc)
    train_pre = np.mean(t_train_pre)
    train_rec = np.mean(t_train_rec)

    print(
        "Train accuracy: {} | Train precision: {} | Train recall: {}".format(
            train_acc,
            train_pre,
            train_rec
        )
    )

    test_acc = np.mean(t_test_acc)
    test_pre = np.mean(t_test_pre)
    test_rec = np.mean(t_test_rec)

    print(
        "Test accuracy: {} | Test precision: {} | Test recall: {}".format(
            test_acc,
            test_pre,
            test_rec
        )
    )

    acc = np.mean(t_acc)
    pre = np.mean(t_pre)
    rec = np.mean(t_rec)

    print(
        "Accuracy: {} | Precision: {} | Recall: {}".format(
            acc,
            pre,
            rec
        )
    )
    print("Confusion matrix:")
    mat = np.mean(np.array(t_mat),axis=0)
    print(mat)