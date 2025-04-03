import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from data import *
from model import *

class Config:
    hidden_size = 256
    dropout = 0.2
    layer = 1
    learning_rate = 1e-3
    epoch = 100
    batch_size = 32
    pos_weight = 2
    threshold = 0.5

def main():
    '''Training the final EPP model'''
    ### 0. Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### 1. Load Input Data and Label Y
    ag_path ="./data/traindata_esm_ag.pt"
    ab_path = "./data/traindata_esm_ab.pt"
    label_path = './data/tensor_label_y.pt'

    train_data_ag = torch.load(ag_path)
    train_data_ab = torch.load(ab_path)
    train_data_y = torch.load(label_path)

    # Partition data set
    X_train_ag, X_val_ag, X_train_ab, X_val_ab, y_train, y_val = train_test_split(train_data_ag, train_data_ab, train_data_y, test_size=test_size, random_state=42)
    print("Data read successfully.")

    ### 2. Define Model
    input_size = 1280  # Input feature dimension
    hidden_size = Config.hidden_size 
    dropout_rate = Config.dropout
    num_layers= Config.layer

    # model = BiLSTMMerge(input_size, hidden_size, num_layers, dropout_rate).to(device)
    model_ag = BiLSTM(input_size, hidden_size, num_layers, dropout_rate).to(device)
    model_ab = BiLSTM(input_size, hidden_size, num_layers, dropout_rate).to(device)
    
    # Define loss functions and optimizers
    pos_weight = torch.tensor([Config.pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    print("Model construction Successful.")


    ### 3. Train Model
    num_epochs = Config.epoch
    batch_size = Config.batch_size
    threshold = Config.threshold
    print("Start model training")
    
    
    # Convert the Dataset into a PyTorch Dataset object
    train_dataset = TensorDataset(X_train_ag, X_train_ab, y_train)
    val_dataset = TensorDataset(X_val_ag, X_val_ab, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    train_model(
        train_loader, val_loader, model, optimizer, criterion, num_epochs,batch_size, threshold, device)
    print("Model training over.")
   
    
def train_model(train_loader, val_loader, model, optimizer, criterion, num_epochs,batch_size, threshold, device):
    '''train model'''
    train_losses = []       # Training set loss
    val_losses = []         # validation set loss
    train_accuracies = []   # Training set accuracy
    val_accuracies = []     # Validation set accuracy
    train_x_accuracies = [] # Training set x-axis accuracy
    train_y_accuracies = [] # Training set y-axis accuracy
    val_x_accuracies = []   # Validation set x-axis accuracy
    val_y_accuracies = []   # Validation set y-axis accuracy

    for epoch in range(num_epochs):
        # train
        model.train()

        train_losses_epoch = []
        train_accuracies_epoch = []
        train_x_accuracies_epoch = []
        train_y_accuracies_epoch = []
        
        for X_ag, X_ab, y_batch in train_loader:
            X_ag = X_ag.to(device)
            X_ab = X_ab.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            
            # result = model(X_ag, X_ab)
            
            output_ag = model_ag(X_ag)
            output_ab = model_ab(X_ab)
            result = torch.matmul(output_ag, output_ab.transpose(0, 1))
            
            result_prob = torch.sigmoid(result)
            loss = criterion(result, y_batch)
            loss.backward()
            optimizer.step()

            train_losses_epoch.append(loss.item())

            # Calculate the training set accuracy
            train_accuracy = accuracy(y_batch, result_prob, device, threshold)
            train_accuracies_epoch.append(train_accuracy)
            # Calculate x-axis and y-axis accuracy
            train_x_accuracy, train_y_accuracy = calculate_axis_accuracy(y_batch, result_prob,  device)
            train_x_accuracies_epoch.append(train_x_accuracy)
            train_y_accuracies_epoch.append(train_y_accuracy)
            

        train_losses.append(sum(train_losses_epoch) / len(train_losses_epoch))
        train_accuracies.append(sum(train_accuracies_epoch) / len(train_accuracies_epoch))
        train_x_accuracies.append(sum(train_x_accuracies_epoch) / len(train_x_accuracies_epoch))
        train_y_accuracies.append(sum(train_y_accuracies_epoch) / len(train_y_accuracies_epoch))        

        # Validate
        model.eval()

        val_losses_epoch = []
        val_accuracies_epoch = []
        val_x_accuracies_epoch = []
        val_y_accuracies_epoch = []
        y_scores_epoch = []
        y_true_epoch = []

        with torch.no_grad():
            for X_ag_val, X_ab_val, y_val_batch in val_loader:
                X_ag_val = X_ag_val.to(device)
                X_ab_val = X_ab_val.to(device)
                y_val_batch = y_val_batch.to(device)
                # result_val = model(X_ag_val, X_ab_val)
                output_ag_val = model_ag(X_ag_val)
                output_ab_val = model_ab(X_ab_val)
                result_val = torch.matmul(output_ag_val, output_ab_val.transpose(0, 1))
                result_val_prob = torch.sigmoid(result_val)

                # Converts the output result to a probability value  -for pr curve
                y_scores_batch = result_val_prob.cpu().detach().numpy()      
                # Converts the output result to a probability value  -for pr curve
                y_true_batch = y_val_batch.cpu().detach().numpy()

                y_scores_epoch.append(y_scores_batch)
                y_true_epoch.append(y_true_batch)

                val_loss = criterion(result_val, y_val_batch)
                val_losses_epoch.append(val_loss.item())
                # Calculate x-axis and y-axis accuracy
                val_x_accuracy, val_y_accuracy = calculate_axis_accuracy(y_val_batch, result_val_prob, device)
                val_x_accuracies_epoch.append(val_x_accuracy)
                val_y_accuracies_epoch.append(val_y_accuracy)

                # Calculate the validation set accuracy
                val_accuracy = accuracy(y_val_batch, result_val_prob, device, threshold)
                val_accuracies_epoch.append(val_accuracy)

        val_losses.append(sum(val_losses_epoch) / len(val_losses_epoch))
        val_accuracies.append(sum(val_accuracies_epoch) / len(val_accuracies_epoch))
        val_x_accuracies.append(sum(val_x_accuracies_epoch) / len(val_x_accuracies_epoch))
        val_y_accuracies.append(sum(val_y_accuracies_epoch) / len(val_y_accuracies_epoch))

        if epoch % 10 == 0:
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), epoch, 'loss:',train_losses[-1],'  acc:',train_accuracies[-1])
       
    return model

if __name__ == "__main__":
    main()
