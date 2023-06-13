import csv
import math
import os

import numpy as np
from matplotlib import pyplot as plt
from torch import Size
from torch.utils.data import TensorDataset, DataLoader
import MyDataset
import torch
import torch.nn as nn
import torch.optim as optim


class FeedForward(nn.Module):
    def __init__(self, input, hidden):
        super(FeedForward, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input, hidden[0]))
        self.layers.append(nn.Tanh())
        for i in range(len(hidden)-1):
            self.layers.append(nn.Linear(hidden[i], hidden[i+1]))
            self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(hidden[-1], 1))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


def train_classifier(filename_data, data_split, dim_emb, list_hidden, learning_rate, num_epochs):
    train_dataload, test_dataload = prep_datasets(filename_data, data_split)

    y_loss = {}  # loss history
    y_loss['train'] = []
    y_loss['test'] = []
    x_epoch = []
    y_acc = {}  # loss history
    y_acc['train'] = []
    y_acc['test'] = []

    fig = plt.figure()

    ax0 = fig.add_subplot(121, title="BCE Loss")
    ax1 = fig.add_subplot(122, title="Accuracy")

    fig.tight_layout(w_pad=3)
    fig.set_size_inches(9, 7)

    model = FeedForward(3*dim_emb, list_hidden)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # Training loop
    for epoch in range(num_epochs):
        train(model, train_dataload, optimizer, criterion, epoch, y_loss, y_acc)
        test(model, test_dataload, criterion, epoch, y_loss, y_acc)

        draw_curve(epoch, x_epoch, y_loss, y_acc, fig, ax0, ax1, filename_data)

    # Evaluate the model on the test data
    #accuracy, precision, recall = evaluate(model, test_dataload)
    #print('Accuracy: {:.4f}'.format(accuracy))
    #print('Precision: {:.4f}'.format(precision))
    #print('Recall: {:.4f}'.format(recall))

    last_dot_index = filename_data.rfind('.')
    save_model(model, filename_data[:last_dot_index])
    save_data(y_loss, y_acc, filename_data)

def prep_datasets(filename_data, split):
    data_tens_train = MyDataset.load_data("Train/" + filename_data)
    if os.path.exists("Daten/Test/" + filename_data):
        data_tens_test = MyDataset.load_data("Test/" + filename_data)
    else:
        data_tens_test = torch.empty(0)

    if (data_tens_test.numel() == 0):
        data_tens_train = data_tens_train[torch.randperm(data_tens_train.size()[0])]
        #Wie viel Prozent des ganzen Datensatzes werden verwendet
        part = math.ceil(data_tens_train.size()[0] * split)
        data_tens_train = data_tens_train[:part]
        # In Trainings- und Testdaten aufteilen
        border = math.ceil(data_tens_train.size()[0] * 0.7)
        data_train = data_tens_train[:border]
        data_test = data_tens_train[border:]
    else:
        data_train = data_tens_train[torch.randperm(data_tens_train.size()[0])]
        data_test = data_tens_test[torch.randperm(data_tens_test.size()[0])]

    # Feature und Label Tensoren erstellen
    data_train_features = data_train[:, :3]
    data_train_features = torch.flatten(data_train_features, 1, 2)
    data_train_labels = data_train[:, 3, 0]
    data_train_labels = torch.unsqueeze(data_train_labels, 1)
    #print("Train:", data_train_labels.size())
    data_test_features = data_test[:, :3]
    data_test_features = torch.flatten(data_test_features, 1, 2)
    data_test_labels = data_test[:, 3, 0]
    data_test_labels = torch.unsqueeze(data_test_labels, 1)

    # Trainings- und Testdatenset erstellen
    data_train = TensorDataset(data_train_features, data_train_labels)
    data_test = TensorDataset(data_test_features, data_test_labels)
    # Trainings- und Testdaten wrappen
    train_dataloader = DataLoader(data_train, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(data_test, batch_size=16, shuffle=True)

    return train_dataloader, test_dataloader

def train(model, train_dataloader, optimizer, criterion, epoch, y_loss, y_acc):

    model.train()

    running_loss = 0.0
    running_corrects = 0.0

    for i, (inputs_batch, labels_batch) in enumerate(train_dataloader):
        #Reset all gradients
        optimizer.zero_grad()
        #Forward pass
        outputs = model(inputs_batch)
        #Compute loss
        loss = criterion(outputs, labels_batch)

        #Backward pass
        loss.backward()
        #Update all weights
        optimizer.step()

        predicted = outputs.detach().clone()

        for i in range(predicted.size()[0]):
            if (predicted[i] >= 0.5):
                predicted[i] = 1
            else:
                predicted[i] = 0


        running_loss += loss.item() * inputs_batch.size(0)

        del loss
        running_corrects += float(torch.sum(predicted == labels_batch.data))

    epoch_loss = running_loss / (train_dataloader.__len__()*16)
    epoch_acc = running_corrects / (train_dataloader.__len__()*16)

    print('Epoch [{}], Train_Loss: {:.4f}, Train_Accuracy: {:.4f}'.format(epoch, epoch_loss, epoch_acc))

    y_loss['train'].append(epoch_loss)
    y_acc['train'].append(epoch_acc)

def test(model, test_dataloader, criterion, epoch, y_loss, y_acc):
    model.eval()

    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    running_loss = 0.0
    running_corrects = 0.0

    with torch.no_grad():
        for features, labels in test_dataloader:

            outputs = model(features)
            predicted = outputs.detach().clone()

            for i in range(predicted.size()[0]):
                if (predicted[i] >= 0.5):
                    predicted[i] = 1
                else:
                    predicted[i] = 0

            total += labels.size(0)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * features.size(0)
            del loss
            running_corrects += float(torch.sum(predicted == labels.data))

            for i in range(len(labels)):
                if labels[i] == 1:
                    if predicted[i] == 1:
                        true_positive += 1
                    else:
                        false_negative += 1
                else:
                    if predicted[i] == 1:
                        false_positive += 1

            correct = total - false_negative - false_positive

    #print("TP:", true_positive, "FP:", false_positive, "FN:", false_negative, "TP+TN:", correct, "Total:", total)
    accuracy = correct / total
    if ((true_positive + false_positive) != 0):
        precision = true_positive / (true_positive + false_positive)
    else:
        print("Precision wurde nicht berechnet, Divison durch 0")
        precision = -1000
    if ((true_positive + false_negative) != 0):
        recall = true_positive / (true_positive + false_negative)
    else:
        print("Recall wurde nicht berechnet, Division durch 0")
        recall = -1000


    epoch_loss = running_loss / (test_dataloader.__len__()*16)
    epoch_acc = running_corrects / (test_dataloader.__len__()*16)

    y_loss['test'].append(epoch_loss)
    y_acc['test'].append(epoch_acc)

    print('Epoch [{}], Test_loss: {:.4f}, Test_accuracy: {:.4f}, Test_precision: {:.4f}, Test_Recall: {:.4f}'.format(epoch, epoch_loss, epoch_acc, precision, recall))

def evaluate(model, data_loader):

        model.eval()

        correct = 0
        total = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0

        with torch.no_grad():
            for features, labels in data_loader:
                #print("Labels", labels)
                outputs = model(features)
                predicted = outputs.detach().clone()
                #print("Outputs", outputs)

                for i in range(predicted.size()[0]):
                    if (predicted[i] >= 0.5):
                        predicted[i] = 1
                    else:
                        predicted[i] = 0

                #print("Predictions", predicted)

                total += labels.size(0)

                for i in range(len(labels)):
                    if labels[i] == 1:
                        if predicted[i] == 1:
                            true_positive += 1
                        else:
                            false_positive += 1
                    else:
                        if predicted[i] == 1:
                            false_negative += 1

                correct = total - false_negative - false_positive

        print("TP:", true_positive, "FP:", false_positive, "FN:", false_negative,"TP+TN:" ,correct, "Total:", total)
        accuracy = correct / total
        if ((true_positive + false_positive) != 0):
            precision = true_positive / (true_positive + false_positive)
        else:
            print("Precision wurde nicht berechnet, Divison durch 0")
            precision = -1000
        if ((true_positive + false_negative) != 0):
            recall = true_positive / (true_positive + false_negative)
        else:
            print("Recall wurde nicht berechnet, Division durch 0")
            recall = -1000

        return accuracy, precision, recall

def draw_curve(current_epoch, x_epoch, y_loss, y_acc, fig, ax0, ax1, filename):

    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo', label='train', ms=2)
    ax0.plot(x_epoch, y_loss['test'], 'ro', label='test', ms=2)
    ax0.set_xlabel("Epoche")
    ax0.set_ylabel("Loss")
    ax1.plot(x_epoch, y_acc['train'], 'bo', label='train', ms=2)
    ax1.plot(x_epoch, y_acc['test'], 'ro', label='test', ms=2)
    ax1.set_xlabel("Epoche")
    ax1.set_ylabel("Accuracy")

    if current_epoch == 0:
        ax0.legend()
        ax1.legend()

    if "naiv" in filename:
        path = "Abbildungen/Abbildung_Loss/naiv/" + filename + ".png"
    elif "force" in filename:
        path = "Abbildungen/Abbildung_Loss/force/" + filename + ".png"
    else:
        path = "Abbildungen/Abbildung_Loss/n2v/" + filename + ".png"
    fig.savefig(path)

def save_data(loss, acc, filename):
    dict_list = [("Loss", loss), ("Acc", acc)]

    filename_cvs = "Cvs Files Training/" + filename + ".cvs"

    with open(filename_cvs, "w", newline="") as csvfile:
        # Define the CSV file writer
        writer = csv.DictWriter(csvfile, fieldnames=["dict name", "train", "test"])

        # Write the header row
        writer.writeheader()

        # Write the dictionary values as rows
        for dict_name, d in dict_list:
            row = {"dict name": dict_name, **d}
            writer.writerow(row)

#Parameter des trainierten Modells speichern
def save_model(model, filename):
    path = "Modell_FeedForward/" + filename + ".pth"
    torch.save(model, path)

#Gespeicherte Parameter des Modells laden
def load_model(filename):
    path = "Modell_FeedForward/" + filename + ".pth"
    loaded_model = torch.load(path)

    return loaded_model








