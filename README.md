# TAMU Datathon 2023: Hospital Plunge
## What it does
In our data-driven journey, we start by receiving patient data in the form of a CSV file. This treasure trove of information is then transformed into a pandas dataframe, a structured format that serves as the foundation for our predictive model.

But what's our model's purpose? To peer into the future and make a crucial call: will the patient, based on the information we have, live?

To carry out this vital task, we enlist the power of neural networks. These intricate systems of interconnected nodes are designed to sift through the data, recognize patterns, and deliver predictions that can be life-saving.

Our journey is one of data, transformation, and prediction. We're not just crunching numbers; we're making informed decisions that could make a world of difference in a patient's life.

## How we built it:
### Data Preprocessing:
#### Handling Missing Values and Data Cleaning:
To handle missing values, we replaced missing values and empty entries with zeros to ensure the trained model would be able to handle the data effectively.
Non-numeric columns were one-hot encoded to prepare categorical data for analysis.

#### Feature Engineering and Transformation:

The dataset was separated into feature vectors and the target label.
Numeric features were standardized using StandardScaler to maintain uniform scales for model training.

### Model Training and Evaluation:
#### Model Selection and Training:

The predictive models used in this analysis involved a Random Forest Classifier, Support Vector Machines (SVM), and a Neural Network implemented using TensorFlow and Keras.
The selected Neural Network model architecture included multiple dense layers with dropout for regularization and a final output layer with a sigmoid activation for binary classification.
#### Training Methodology:

K-Fold cross-validation with 5 folds was employed to train and evaluate the Neural Network model. PCA was utilized for dimensionality reduction in the feature space.
The model was trained for 20 epochs with a batch size of 32.
Performance Evaluation:

Model performance metrics including F1-score, Precision, Recall, and Accuracy were computed for each fold during cross-validation.
Results were printed for each iteration, showcasing the model's performance in predicting patient outcomes.
#### Model Persistence:

The trained Neural Network model was saved in a Keras format for this submission.

## Challenges we ran into

The most significant hurdle we encountered revolved around hyperparameter tuning. We extensively navigated through various aspects of the model's complexity, including the number and sizes of layers, the extent of dimensionality reduction using PCA's eigenvector components, the application of regularization through dropout, and the level of k-fold cross-validation.

## Accomplishments that we're proud of

We accomplished the development and evaluation of multiple machine learning models, including the Random Forest Classifier, Support Vector Machines (SVM), and a Neural Network built using TensorFlow and Keras. Our process involved meticulous data preprocessing, encompassing the handling of missing values, dataset cleaning, categorical feature encoding, and numerical data standardization, ensuring the dataset's readiness for model training. By employing K-Fold cross-validation with 5 folds and utilizing Principal Component Analysis (PCA) for dimensionality reduction, we effectively trained and evaluated the Neural Network model. We meticulously computed and reported vital performance metrics such as F1-score, Precision, Recall, and Accuracy for each fold during cross-validation, providing a comprehensive understanding of the model's predictive capabilities. Furthermore, we ensured the model's readiness for real-world application by saving it in a Keras format for future use or deployment.

## What we learned
Our journey through this data adventure has been enlightening. Real-world data isn't always as pristine as we'd hope. Instead, it often arrives in less-than-perfect form, with gaps and quirks that need our attention.

Our first lesson? Data wrangling. We learned how to handle missing values and outliers. Sometimes, we filled in the gaps with educated guesses. Other times, we had to let go of problematic outliers. The art of data preparation is all about making the information accessible and readable for our model.

But data isn't always just numbers. Often, we encountered categorical data, which our models couldn't directly comprehend. That's where one-hot encoding came into play. It's our secret sauce for translating categories into numbers, making it possible for our model to crunch the data.

Then, the spotlight shifted to the stars of our show – neural networks built with TensorFlow and Keras. We ventured into the realm of artificial intelligence, crafting models capable of making predictions, classifications, and uncovering hidden patterns in the data.

By far, our biggest takeaway from this Datathon was collaboration and learning. Together, we tackled real-world data puzzles, transformed them into actionable insights, and expanded our horizons in machine learning. The journey may not always be smooth, but the discoveries and growth along the way make it all worthwhile.

## What's next for Hospital Plunge
### Feature Importance Analysis:

Further exploration of feature importance could provide insights into which medical parameters significantly impact patient outcomes, thus allowing us to optimize our model by selecting the top *n* features from our data.

### Hyperparameter Tuning:

Fine-tuning the model's hyperparameters and architecture could potentially enhance predictive performance.

### Class Imbalance Handling:

Given the potential class imbalance in the target variable, advanced techniques for handling class imbalance like oversampling or undersampling could be explored to improve how well the model can generalize to unseen data.

### External Validation and Real-World Application:

Evaluating the model's performance on external datasets ("transfer learning") and considering its deployment within a real clinical environment would be crucial to verify its practical utility and robustness.
