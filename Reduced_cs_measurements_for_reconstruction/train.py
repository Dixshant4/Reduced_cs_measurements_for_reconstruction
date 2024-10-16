import Model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import Data_setup
import time
from torch.utils.data import DataLoader
import os



def accuracy(model, dataset, device):
    """
    Compute the accuracy of `model` over the `dataset`.
    We will take the **most probable class**
    as the class predicted by the model.

    Parameters:
        `model` - A PyTorch MLPModel
        `dataset` - A data structure that acts like a list of 2-tuples of
                  the form (x, t), where `x` is a PyTorch tensor of shape
                  [400,1] representinga pattern,
                  and `t` is the corresponding binary target label

    Returns: a floating-point value between 0 and 1.
    """

    correct, total = 0, 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=10)
    for pattern, t in loader:
        # X = img.reshape(-1, 784)
        pattern = pattern[:,:30].to(device)
        t = t.to(device)
        z = model(pattern)
        y = torch.sigmoid(z)
        pred = (y >= 0.5).int()
        # pred should be a [N, 1] tensor with binary
        # predictions, (0 or 1 in each entry)

        correct += int(torch.sum(t == pred))
        total += t.shape[0]
    # if total == 0:
    #     return 0.0
    return correct / total



def train_model(model,                # an instance of MLPModel
                train_data,           # training data
                val_data,             # validation data
                learning_rate=0.01,
                batch_size=32,
                num_epochs=8000,
                plot_every=50,        # how often (in # iterations) to track metrics
                plot=True,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):           # whether to plot the training curve
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True) # reshuffle minibatches every epoch
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.004)
    model = model.to(device)

    # these lists will be used to track the training progress
    # and to plot the training curve
    iters, train_loss, train_acc, val_acc = [], [], [], []
    iter_count = 0 # count the number of iterations that has passed

    try:
        for e in range(num_epochs):
            for i, (patterns, labels) in enumerate(train_loader):
                start = time.time()
                patterns = patterns[:,:30].to(device)

                # print(patterns[:,:200].shape)
                labels = labels.to(device)

                z = model(patterns).float()
                loss = criterion(z, labels.float())

                loss.backward() # propagate the gradients
                optimizer.step() # update the parameters
                optimizer.zero_grad() # clean up accumualted gradients


                iter_count += 1
                if iter_count % plot_every == 0:
                    iters.append(iter_count)
                    ta = accuracy(model, train_data, device)
                    va = accuracy(model, val_data, device)
                    train_loss.append(float(loss))
                    train_acc.append(ta)
                    val_acc.append(va)
                    end = time.time()
                    time_taken = round(end - start, 3)
                    print(iter_count, "Loss:", float(loss), "Train Acc:", ta, "Val Acc:", va, 'Time taken:', time_taken)
    finally:
        if plot:
            plt.figure()
            plt.plot(iters[:len(train_loss)], train_loss)
            plt.title("Loss over iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.savefig('training_loss.png')

            plt.figure()
            plt.plot(iters[:len(train_acc)], train_acc)
            plt.plot(iters[:len(val_acc)], val_acc)
            plt.title("Accuracy over iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Accuracy")
            plt.legend(["Train", "Validation"])
            plt.savefig('accuracy.png')

train_data = Data_setup.train_data
validation_data = Data_setup.val_data
test_data = Data_setup.test_data

model = Model.MLPModel()
train_model(model, train_data, validation_data)

test_accuracy = accuracy(model, test_data, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print(test_accuracy)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def get_prediction(model, data, sample=1000):
    loader = torch.utils.data.DataLoader(data, batch_size=sample, shuffle=True)
    for X, t in loader:
        z = model(X[:,:30])
        y = torch.sigmoid(z)
        break
    y = y.detach().numpy()
    t = t.detach().numpy()
    return y, t

y, t = get_prediction(model, validation_data)
y = y > 0.5
cm = confusion_matrix(t, y)
cmp = ConfusionMatrixDisplay(cm, display_labels=["0", "1"])
cmp.plot()
plt.title("Confusion Matrix (Val Data)")
plt.savefig('confusion_matrix.png')