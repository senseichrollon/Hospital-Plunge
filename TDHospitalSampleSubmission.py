# Sample participant submission for testing
from flask import Flask, jsonify, request
import tensorflow as tf
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import load
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)


class Solution:
    def __init__(self):
        #Initialize any global variables here
       self.model = tf.keras.models.load_model('example.keras')
     #   self.model = load('random_forest_model.joblib')
        # import pickle

        # # Load the saved model using pickle
        # with open('gradientboost.pkl', 'rb') as file:
        #     self.model = pickle.load(file)


    import pandas as pd
    def data_preprocessing(self,df):
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
    def standardize(self,X):
        scaler = StandardScaler()
        X_numeric = scaler.fit_transform(X.select_dtypes(include=['float64']))
        X[X.select_dtypes(include=['float64']).columns] = X_numeric
        return X
    def split_feature_label(self,df):
        y = df['death']
        X = df.drop(columns=['death'])
        return y, X
    def calculate_death_prob(self, timeknown, cost, reflex, sex, blood, bloodchem1, bloodchem2, temperature, race,
                             heart, psych1, glucose, psych2, dose, psych3, bp, bloodchem3, confidence, bloodchem4,
                             comorbidity, totalcost, breathing, age, sleep, dnr, bloodchem5, pdeath, meals, pain,
                             primary, psych4, disability, administratorcost, urine, diabetes, income, extraprimary,
                             bloodchem6, education, psych5, psych6, information, cancer):
        labels = ['timeknown', 'cost', 'reflex', 'sex', 'blood', 'bloodchem1', 'bloodchem2', 'temperature', 'race',
                  'heart', 'psych1', 'glucose', 'psych2', 'dose', 'psych3', 'bp', 'bloodchem3', 'confidence',
                  'bloodchem4',
                  'comorbidity', 'totalcost', 'breathing', 'age', 'sleep', 'dnr', 'bloodchem5', 'pdeath', 'meals',
                  'pain',
                  'primary', 'psych4', 'disability', 'administratorcost', 'urine', 'diabetes', 'income', 'extraprimary',
                  'bloodchem6', 'education', 'psych5', 'psych6', 'information', 'cancer']

        # Convert all parameters to float and place them in a list
        values = [timeknown, cost, reflex, sex, blood, bloodchem1, bloodchem2,
                                                              temperature, race, heart, psych1, glucose, psych2, dose,
                                                              psych3, bp, bloodchem3, confidence, bloodchem4, comorbidity,
                                                              totalcost, breathing, age, sleep, dnr, bloodchem5, pdeath,
                                                              meals, pain, primary, psych4, disability, administratorcost,
                                                              urine, diabetes, income, extraprimary, bloodchem6, education,
                                                              psych5, psych6, information, cancer]

        # Create a dictionary for the DataFrame
        df_dict = {label: value for label, value in zip(labels, values)}
        import csv
        with open('parameters.csv', 'w', newline='') as csvfile:
            fieldnames = list(df_dict.keys())
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerow(df_dict)
   #     print(df_dict)
        # Create a DataFrame
        df = pd.read_csv('parameters.csv')

        data_path = 'TD_HOSPITAL_TRAIN.csv'
        df_train = pd.read_csv(data_path)
      #  print(df_train.shape)
        y, X = self.split_feature_label(df_train)
        combined = pd.concat([X, df])
        combined = self.data_preprocessing(combined)
        combined = self.standardize(combined)


        df_train = combined.iloc[:len(X)]
        df_test = combined.iloc[len(X):]
        pca = PCA(n_components=40)
        pca.fit(df_train)
        print(df_test)
        df_test = pca.transform(df_test)
        # df = self.data_preprocessing(df)
        # df = self.standardize(df)
        # print(df)
        # print(df.columns)
        # print(df.shape)
       # print(df_test.shape)
        prediction = self.model.predict(df_test)
        print(prediction)
        
        return 1-float(prediction[0][0])


# BOILERPLATE
@app.route("/death_probability", methods=["POST"])
def q1():
    solution = Solution()
    data = request.get_json()
    return {
        "probability": solution.calculate_death_prob(data['timeknown'], data['cost'], data['reflex'], data['sex'], data['blood'],
                                            data['bloodchem1'], data['bloodchem2'], data['temperature'], data['race'],
                                            data['heart'], data['psych1'], data['glucose'], data['psych2'],
                                            data['dose'], data['psych3'], data['bp'], data['bloodchem3'],
                                            data['confidence'], data['bloodchem4'], data['comorbidity'],
                                            data['totalcost'], data['breathing'], data['age'], data['sleep'],
                                            data['dnr'], data['bloodchem5'], data['pdeath'], data['meals'],
                                            data['pain'], data['primary'], data['psych4'], data['disability'],
                                            data['administratorcost'], data['urine'], data['diabetes'], data['income'],
                                            data['extraprimary'], data['bloodchem6'], data['education'], data['psych5'],
                                            data['psych6'], data['information'], data['cancer'])}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555)
