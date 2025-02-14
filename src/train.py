import pandas as pd
import joblib
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import config
import os 
import argparse
import model_dispatcher
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
pd.set_option('future.no_silent_downcasting', True)


def feature_work(x_train, x_valid):
    #do some feature engineering
    x_train["Sex"]=x_train["Sex"].map({"male":0,"female":1})
    x_valid["Sex"]=x_valid["Sex"].map({"male":0,"female":1})

    x_train["Embarked"] = x_train['Embarked'].replace({'C': 1, 'S': 2, 'Q': 3})
    x_valid["Embarked"] = x_valid['Embarked'].replace({'C': 1, 'S': 2, 'Q': 3})

    x_train['Ticket'] = x_train['Ticket'].str.extract(r'(\d+)')  # Extracts first number in the string
    x_valid['Ticket'] = x_valid['Ticket'].str.extract(r'(\d+)')  # Extracts first number in the string

    # Convert to numeric type
    x_train['Ticket'] = pd.to_numeric(x_train['Ticket'])
    x_valid['Ticket'] = pd.to_numeric(x_valid['Ticket'])
    

    #Convert Names
    #Step 1 extract last names
    x_train['Last_Name']= x_train['Name'].str.split(',').str[0]
    x_valid['Last_Name']= x_valid['Name'].str.split(',').str[0]

    # Step 2: Fit LabelEncoder on training data
    le = LabelEncoder()
    x_train['Last_Name_Encoded'] = le.fit_transform(x_train['Last_Name'])

    #Step 3: Create mapping dic
    name_mapping=dict(zip(le.classes_, le.transform(le.classes_)))

    #Step 4: Transform valid set safely, assign -1 to unkown values
    x_valid['Last_Name_Encoded'] = x_valid['Last_Name'].map(name_mapping).fillna(-1).astype(int)

    #Step 5: Keep only encoding
    x_train.drop(["Name", "Last_Name"], axis=1, inplace=True)
    x_valid.drop(["Name", "Last_Name"], axis=1, inplace=True)


    scaler=MinMaxScaler()

    scaler.fit(x_train[["Age","Fare","Ticket", "Last_Name_Encoded"]])

    x_train[["Age","Fare","Ticket","Last_Name_Encoded"]]=scaler.transform(x_train[["Age","Fare","Ticket", "Last_Name_Encoded"]])
    x_valid[["Age","Fare","Ticket","Last_Name_Encoded"]]=scaler.transform(x_valid[["Age","Fare","Ticket", "Last_Name_Encoded"]])

    #deal with NAN's
    x_train.fillna(-1, inplace=True)
    x_valid.fillna(-1, inplace=True)

    return x_train, x_valid



def run(fold, model, test):
    if test==True:
        df=pd.read_csv(config.Training_File)
        train_df=df.drop("kfold", axis=1)
        valid_df=pd.read_csv(config.Test_File)
        x_valid=valid_df.drop(["PassengerId", "Cabin"],axis=1)

    else:
        df=pd.read_csv(config.Training_File)
        train_df=df[df.kfold!= fold].reset_index(drop=True)
        valid_df=df[df.kfold==fold].reset_index(drop=True)
        x_valid=valid_df.drop(["Survived", "PassengerId", "Cabin"],axis=1)
        y_valid=valid_df.Survived.values
        

    x_train=train_df.drop(["Survived", "PassengerId", "Cabin"],axis=1)
    y_train=train_df.Survived.values

    x_train, x_valid= feature_work(x_train, x_valid)

    x_train, x_valid=x_train.values, x_valid.values

    clf=model_dispatcher.models[model]

    clf.fit(x_train,y_train)

    preds=clf.predict(x_valid)

    if test==False:

        accuracy=metrics.accuracy_score(y_valid,preds)

        print(f"Fold={fold},Accuracy={accuracy}")
        print(confusion_matrix(y_valid,preds))
    
        #joblib.dump(clf,
        #            os.path.join(config.Model_Output, f"{model}_{fold}.bin"))
    
    else:
        output=pd.DataFrame({
            'PassengerId': valid_df['PassengerId'],
            'Survived': preds
         })
        
        output.to_csv(os.path.join(config.Output_File, f"predictions.csv"), index=False)
        print("Predictions saved to predictions.csv")

if __name__ == "__main__":
    
    parser= argparse.ArgumentParser()

    parser.add_argument("--fold",
                        type=int,
                        default=0,
                        help="The number of the fold to be used for testing")
    
    parser.add_argument("--model", 
                        type=str,
                        default="svc",
                        help="The model that should be used for the task")
    
    parser.add_argument("--test",
                        type=bool,
                        default=False,
                        help="True: if predictions should be made for the test set, else false")
    
    #read arguments from the parser
    args= parser.parse_args()
    run(fold=args.fold, model=args.model, test=args.test)