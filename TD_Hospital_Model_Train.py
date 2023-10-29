import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from tensorflow import keras
from keras import layers
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from imblearn.over_sampling import RandomOverSampler
from keras import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.optimizers import Adam

def data_preprocessing(df):

   # col_to_keep = ['death', 'age', 'blood', 'reflex', 'bloodchem1', 'bloodchem2', 'psych1', 'glucose', 'temperature','heart', 'breathing']
  #  df = df[col_to_keep]
    df.replace('', 0, inplace=True)
    df.fillna(0, inplace=True)


    # # Get column names of non-numeric columns
    # non_numeric_columns = df.select_dtypes(exclude=['number']).columns
    #
    # # Drop non-numeric columns from the DataFrame
    # df = df.drop(columns=non_numeric_columns)


    # Get non-numeric columns
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns

    # One-hot encode non-numeric columns
    df = pd.get_dummies(df, columns=non_numeric_columns)

    return df
    
def split_feature_label(df):
    y = df['death']
    X = df.drop(columns=['death'])
    return y, X
    # print(X)
    # print(y)

    # death_0 = y.tolist().count(0)
    # death_1 = y.tolist().count(1)
    # percent_death_0 = 100 * death_0 / (death_0 + death_1)
    # percent_death_1 = 100 * death_1 / (death_0 + death_1)
    # print(f'Survived: {death_0}, or {percent_death_0:.2f}%')
    # print(f'Died: {death_1}, or {percent_death_1:.2f}%')

def standardize(X):
    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(X.select_dtypes(include=['float64']))
    X[X.select_dtypes(include=['float64']).columns] = X_numeric
    return X

def train_model(X, y):
    # Initialize K-Fold cross-validation with 5 folds
    kfold = KFold(n_splits=10, shuffle=True)

    model = None
    # Use K-Fold cross-validation to train and evaluate the Neural Network model
    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=40)  # You can adjust the number of components as needed
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Initialize the Neural Network model
        model = Sequential([
            Dense(170, activation='relu', input_shape=(X_train_pca.shape[1],)),
            Dropout(0.25),
            Dense(170, activation='relu'),
            Dropout(0.25),
            Dense(20),
            Dense(1, activation='sigmoid')
        ])
        # 128, 64 pca=10, no dropout
        # 170, 170, pca=20, no dropout
        # Compile the model
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train_pca, y_train, epochs=20, batch_size=32, verbose=0)

        # Predict using the model
        y_pred = (model.predict(X_test_pca) > 0.5).astype(int).reshape(-1)

        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Metrics for iteration:")
        print(f"F1-score: {f1}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Accuracy: {accuracy}")
        print("")
  #  model.fit(X,y)
    model.save('example.keras')


if __name__ == "__main__":
    data_path = 'TD_HOSPITAL_TRAIN.csv'
    df = pd.read_csv(data_path)
    cleaned_data = data_preprocessing(df)
    class_counts = df['death'].value_counts()
    y, X = split_feature_label(cleaned_data)
    X = standardize(X)
    train_model(X, y)
    