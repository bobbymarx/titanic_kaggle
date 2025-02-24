import pandas as pd
import joblib
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import config
import os 
import argparse
import model_dispatcher
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
from sklearn.ensemble import RandomForestRegressor

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def Cabins_converter(x_tmp):
    # Step 1: Extract first letter or assign 'U' if NaN
    x_tmp['Letter'] = x_tmp['Cabin'].astype(str).str[0].where(x_tmp['Cabin'].notna(), 'U')
    # Step 2: Define mapping and convert to numbers
    letter_mapping = {letter: idx for idx, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZU', start=1)}
    x_tmp['Letter_Encoded'] = x_tmp['Letter'].map(letter_mapping)
    #Step 3: Clean up
    x_tmp.drop(['Letter', 'Cabin'], axis=1, inplace=True)
    return x_tmp

def Names_handler(x_train, x_valid):
    # Step 1: Extract last names
    x_train['Last_Name'] = x_train['Name'].str.split(',').str[0]
    x_valid['Last_Name'] = x_valid['Name'].str.split(',').str[0]

    # Step 2: Fit LabelEncoder on training data
    le = LabelEncoder()
    x_train['Last_Name_Encoded'] = le.fit_transform(x_train['Last_Name'])

    # Step 3: Create a mapping dictionary
    name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    # Step 4: Transform valid set and assign new unique labels for unknown values
    next_label = max(name_mapping.values()) + 1
    x_valid['Last_Name_Encoded'] = x_valid['Last_Name'].map(name_mapping)

    # Assign new unique labels to previously unseen last names
    new_names = x_valid['Last_Name'][x_valid['Last_Name_Encoded'].isna()].unique()
    new_mapping = {name: i for i, name in enumerate(new_names, start=next_label)}
    name_mapping.update(new_mapping)
    x_valid['Last_Name_Encoded'] = x_valid['Last_Name'].map(name_mapping).astype(int)

    # Define title mapping
    title_mapping = {
        'Dr': 1, 
        'Mr': 2, 
        'Miss': 3, 
        'Mrs': 4, 
        'Master': 5, 
        'Lady': 6, 
        'Major': 7, 
        'Rev': 8, 
        'Don': 9,
        'Jonkheer': 10, 
        'Sir': 11, 
        'Countess': 12, 
        'Capt': 13, 
        'Col': 14, 
        'Mlle': 15, 
        'Ms': 16, 
        'Mme': 17
    }

    # Extract titles and map them
    x_train['Title'] = x_train['Name'].str.extract(r' ([A-Za-z]+)\.').iloc[:, 0]
    x_valid['Title'] = x_valid['Name'].str.extract(r' ([A-Za-z]+)\.').iloc[:, 0]
    
    # Map titles to numbers, with 0 for unknown titles
    x_train['Title_Encoded'] = x_train['Title'].map(title_mapping).fillna(0).astype(int)
    x_valid['Title_Encoded'] = x_valid['Title'].map(title_mapping).fillna(0).astype(int)

    # Drop original name columns and title
    x_train.drop(["Name", "Last_Name", "Title"], axis=1, inplace=True)
    x_valid.drop(["Name", "Last_Name", "Title"], axis=1, inplace=True)

    return x_train, x_valid

def impute_age(df, features=['Pclass', 'Sex', 'SibSp', 'Parch']):
    df = df.copy()  # Work on a copy to avoid modifying the original DataFrame

    # Select rows where Age is not missing
    known_age = df[df['Age'].notna()]
    unknown_age = df[df['Age'].isna()]

    # If there are no known ages, use the global mean age (around 30 for Titanic dataset)
    if len(known_age) == 0:
        df.loc[df['Age'].isna(), 'Age'] = 30
        return df

    # Define the features for predicting age
    X_train = known_age[features]
    y_train = known_age['Age']

    # Train a RandomForest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict Age for rows where Age is missing
    X_test = unknown_age[features]
    predicted_ages = model.predict(X_test)

    # Assign the predicted ages
    df.loc[df['Age'].isna(), 'Age'] = predicted_ages

    return df

def feature_work(x_train, x_valid):
    #do some feature engineering
    x_train["Sex"]=x_train["Sex"].map({"male":0,"female":1})
    x_valid["Sex"]=x_valid["Sex"].map({"male":0,"female":1})

    x_train["Embarked"] = x_train['Embarked'].replace({'C': 1, 'S': 2, 'Q': 3}).fillna(-1)
    x_valid["Embarked"] = x_valid['Embarked'].replace({'C': 1, 'S': 2, 'Q': 3}).fillna(-1)

    x_train['Ticket'] = x_train['Ticket'].str.extract(r'(\d+)')  # Extracts first number in the string
    x_valid['Ticket'] = x_valid['Ticket'].str.extract(r'(\d+)')  # Extracts first number in the string

    # Convert to numeric type
    x_train['Ticket'] = pd.to_numeric(x_train['Ticket'])
    x_valid['Ticket'] = pd.to_numeric(x_valid['Ticket'])

    #deal with Cabins
    x_train=Cabins_converter(x_train)
    x_valid=Cabins_converter(x_valid)
    
    #Deal with Names
    x_train, x_valid= Names_handler(x_train, x_valid)


    """
    #deal with NAN's
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    x_train['Age']=imp_mean.fit_transform(x_train[['Age']])
    x_valid['Age']=imp_mean.transform(x_valid[['Age']])
    """

    x_train=impute_age(x_train)
    x_valid=impute_age(x_valid)
    
    #scaling
    scaler=StandardScaler()

    scaler.fit(x_train[["Age","Fare","Ticket", "Last_Name_Encoded"]])

    x_train[["Age","Fare","Ticket","Last_Name_Encoded"]]=scaler.transform(x_train[["Age","Fare","Ticket", "Last_Name_Encoded"]])
    x_valid[["Age","Fare","Ticket","Last_Name_Encoded"]]=scaler.transform(x_valid[["Age","Fare","Ticket", "Last_Name_Encoded"]])

    #debugging
    nan_rows = x_valid[x_valid.isna().any(axis=1)]
    nan_counts = x_valid.isna().sum()

    #(nan_counts)
    #print(nan_rows)

    x_train.fillna(-1, inplace=True)
    x_valid.fillna(-1, inplace=True)

    return x_train, x_valid

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.bn1 = nn.BatchNorm1d(11)
        self.layer1 = nn.Linear(11, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(p=0.3)  # Reduced dropout
        self.layer2 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(p=0.3)
        self.output = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(self.layer1(x))
        x = self.bn2(x)
        x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        x = self.bn3(x)
        x = self.dropout2(x)
        x = self.output(x)
        return self.sigmoid(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)  # prevents nans when probability 0
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

def run(fold, test):
    if test==True:
        train_df=pd.read_csv(config.Training_File)
        #train_df=df.drop("kfold", axis=1)
        valid_df=pd.read_csv(config.Test_File)
        x_valid=valid_df.drop(["PassengerId"],axis=1)

    else:
        df=pd.read_csv(config.Training_File)
        train_df=df[df.kfold!= fold].reset_index(drop=True)
        valid_df=df[df.kfold==fold].reset_index(drop=True)
        x_valid=valid_df.drop(["Survived", "PassengerId", "kfold"],axis=1)
        y_valid=valid_df.Survived.values
        

    x_train=train_df.drop(["Survived", "PassengerId", "kfold"],axis=1)
    y_train=train_df.Survived.values

    x_train, x_valid= feature_work(x_train, x_valid)

    x_train, x_valid=x_train.values, x_valid.values

    #Set up NN
    x_train = np.array(x_train, dtype=np.float32)
    x_valid = np.array(x_valid, dtype=np.float32)


    #convert to tensors
    X_train = torch.tensor(x_train, dtype=torch.float32)
    Y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

    model = NeuralNetwork()
    loss_fn = FocalLoss(alpha=0.25, gamma=3)  # Modified alpha
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)  # Added weight decay and increased lr

    n_epochs = 400
    batch_size = 32  # Increased batch size
    early_stopping_patience = 30
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # Convert validation data to tensors
    X_valid_tensor = torch.tensor(x_valid, dtype=torch.float32)
    Y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).reshape(-1, 1)

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Shuffle training data
        indices = torch.randperm(len(X_train))
        X_train_shuffled = X_train[indices]
        Y_train_shuffled = Y_train[indices]
        
        for i in range(0, len(X_train), batch_size):
            Xbatch = X_train_shuffled[i:i+batch_size]
            ybatch = Y_train_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            y_pred = model(Xbatch)
            loss = loss_fn(y_pred, ybatch)
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_valid_tensor)
            val_loss = loss_fn(val_predictions, Y_valid_tensor)
            val_accuracy = ((val_predictions > 0.5) == Y_valid_tensor).float().mean()
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Print metrics every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{n_epochs}:')
                print(f'Training Loss: {avg_train_loss:.4f}')
                print(f'Validation Loss: {val_loss:.4f}')
                print(f'Validation Accuracy: {val_accuracy:.4f}')
                print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}\n')
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_model_state = model.state_dict()
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_predictions = model(X_valid_tensor)
        final_accuracy = ((final_predictions > 0.5) == Y_valid_tensor).float().mean()
        print(f"\nFinal Validation Accuracy: {final_accuracy*100:.4f}")



if __name__ == "__main__":
    
    parser= argparse.ArgumentParser()

    parser.add_argument("--fold",
                        type=int,
                        default=0,
                        help="The number of the fold to be used for testing")

    
    parser.add_argument("--test",
                        type=bool,
                        default=False,
                        help="True: if predictions should be made for the test set, else false")
    
    #read arguments from the parser
    args= parser.parse_args()
    run(fold=args.fold, test=args.test)