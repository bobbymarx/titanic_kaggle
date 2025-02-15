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
    next_label = max(name_mapping.values()) + 1  # Start labeling from the next available number
    x_valid['Last_Name_Encoded'] = x_valid['Last_Name'].map(name_mapping)

    # Assign new unique labels to previously unseen last names
    new_names = x_valid['Last_Name'][x_valid['Last_Name_Encoded'].isna()].unique()
    new_mapping = {name: i for i, name in enumerate(new_names, start=next_label)}

    # Update the mapping dictionary
    name_mapping.update(new_mapping)

    # Apply the updated mapping to `x_valid`
    x_valid['Last_Name_Encoded'] = x_valid['Last_Name'].map(name_mapping).astype(int)

    #check Titels in Names
    x_train['Title'] = x_train['Name'].str.extract(r' ([A-Za-z]+)\.')  # Looks for words followed by a dot
    x_train['Title']= x_train['Title'].fillna('Unknown')
    x_valid['Title'] = x_valid['Name'].str.extract(r' ([A-Za-z]+)\.')  
    x_train['Title']= x_valid['Title'].fillna('Unknown')

    #one-hot encode the Titles
    x_train = pd.get_dummies(x_train, columns=['Title'], prefix='Title')
    x_valid = pd.get_dummies(x_valid, columns=['Title'], prefix='Title')

    # Step 5: Keep only encoding
    x_train.drop(["Name", "Last_Name"], axis=1, inplace=True)
    x_valid.drop(["Name", "Last_Name"], axis=1, inplace=True)



    return x_train, x_valid

def impute_age(df, features=['Pclass', 'Sex', 'SibSp', 'Parch']):
    df = df.copy()  # Work on a copy to avoid modifying the original DataFrame

    # Select rows where Age is not missing
    known_age = df[df['Age'].notna()]
    unknown_age = df[df['Age'].isna()]

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



def run(fold, model, test):
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

    clf=model_dispatcher.models[model]

    clf.fit(x_train,y_train)

    #use different proba for rf to improve precision

    if model=='rf':
        y_preds= clf.predict_proba(x_valid)[:,1]
        threshold=0.43
        preds=(y_preds>=threshold).astype(int)
        


    else:
        preds=clf.predict(x_valid)

    if test==False:

        accuracy=metrics.accuracy_score(y_valid,preds)

        print(f"Fold={fold},Accuracy={accuracy}")
        print(confusion_matrix(y_valid,preds))

        counter=0
        for runner in range(len(y_valid)):
            if y_valid[runner]==1:
                if preds[runner]==0:
                    counter+=1
        
        print(counter)



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