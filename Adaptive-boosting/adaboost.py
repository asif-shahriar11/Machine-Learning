import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import MinMaxScaler

np.random.seed(1)  # For reproducibility

def telco_data_preprocessing(data_source, k = 10):
    data = pd.read_csv(data_source)
    # for each feature(column), chicking the unique values and their counts
    # for col in data.columns:
    #     print(data[col].value_counts())

    # customerID is unique for each customer, so we can drop it
    data.drop('customerID', axis=1, inplace=True)

    # train-test split
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # handling missing values: totalCharges has 11 blank values
    # we can replace them with the mean value
    train_data['TotalCharges'] = train_data['TotalCharges'].replace(' ', np.nan)
    test_data['TotalCharges'] = test_data['TotalCharges'].replace(' ', np.nan)
    train_data['TotalCharges'] = train_data['TotalCharges'].astype(float)
    test_data['TotalCharges'] = test_data['TotalCharges'].astype(float)
    #data['TotalCharges'] = data['TotalCharges'].astype(float)
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    #data['TotalCharges'] = imputer.fit_transform(data[['TotalCharges']])  
    train_data['TotalCharges'] = imputer.fit_transform(train_data[['TotalCharges']])
    test_data['TotalCharges'] = imputer.fit_transform(test_data[['TotalCharges']])

    return basic_preprocessing(train_data, test_data, k, 'Churn', 'Yes', 'No')

    # separating features and target
    # train_target = train_data['Churn']
    # train_data = train_data.drop('Churn', axis=1)
    # test_target = test_data['Churn']
    # test_data = test_data.drop('Churn', axis=1)

    # # numerical and categorical features
    # numeric_features = train_data.select_dtypes(include=['int64', 'float64']).columns
    # categorical_features = train_data.select_dtypes(include=['object']).columns

    # # normalizing numerical features
    # scaler = MinMaxScaler()
    # train_data[numeric_features] = scaler.fit_transform(train_data[numeric_features])
    # test_data[numeric_features] = scaler.fit_transform(test_data[numeric_features])

    # # one-hot encoding categorical features
    # train_data = pd.get_dummies(train_data, columns=categorical_features)
    # test_data = pd.get_dummies(test_data, columns=categorical_features)

    # # encoding target
    # train_target = train_target.replace({'Yes':1, 'No':0})
    # test_target = test_target.replace({'Yes':1, 'No':0})

    # # converting to numpy arrays
    # train_data = train_data.to_numpy()
    # train_target = train_target.to_numpy()
    # test_data = test_data.to_numpy()
    # test_target = test_target.to_numpy()

    # # feature selection using information gain
    # information_gain = mutual_info_classif(train_data, train_target)
    # # sorting the features based on their information gain
    # sorted_features = np.argsort(information_gain)[::-1]
    # # selecting top features
    # top_features = sorted_features[:k]
    # #top_features = np.argsort(information_gain)[::-1][:k]
    # #print('Top features: ', top_features)

    # # selecting top features
    # train_data = train_data[:, top_features]
    # test_data = test_data[:, top_features]


    
    # return train_data, train_target, test_data, test_target



def adult_data_preprocessing(k = 10):
    
    train_data = pd.read_csv(adult_train_data_source, header=None, sep=', ', engine='python')
    test_data = pd.read_csv(adult_test_data_source, header=None, sep=', ', engine='python', skiprows=1)
    # print(train_data.shape, test_data.shape) # (32561, 15) (16281, 15)
    # this dataset does not have column names, so we have to add them
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 
                    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    train_data.columns = column_names
    test_data.columns = column_names

    # for each feature(column), checking the unique values and their counts
    # for col in train_data.columns:
    #     print(train_data[col].value_counts())   
    # numerical: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
    # categorical: workclass (missing: ?), education, marital-status, occupation (?), relationship, race, sex, native-country (?), income (target)
    
    # handling missing values
    # workclass, occupation, native-country have missing values
    # replacing '?' with nan
    train_data = train_data.replace('?', np.nan)
    test_data = test_data.replace('?', np.nan)
    # replacing nan with the most frequent value using SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    train_data[['workclass']] = imputer.fit_transform(train_data[['workclass']])
    train_data[['occupation']] = imputer.fit_transform(train_data[['occupation']])
    train_data[['native-country']] = imputer.fit_transform(train_data[['native-country']])
    test_data[['workclass']] = imputer.fit_transform(test_data[['workclass']])
    test_data[['occupation']] = imputer.fit_transform(test_data[['occupation']])
    test_data[['native-country']] = imputer.fit_transform(test_data[['native-country']])

    # test data has a '.' at the end of the target values
    test_data['income'] = test_data['income'].str.rstrip('.')

    return basic_preprocessing(train_data, test_data, k, 'income', '<=50K', '>50K')

    # separating features and target
    # train_target = train_data['income']
    # train_data = train_data.drop('income', axis=1)
    # test_target = test_data['income']
    # test_data = test_data.drop('income', axis=1)

    # # encoding target
    # train_target = train_target.replace({'<=50K':0, '>50K':1})
    # test_target = test_target.replace({'<=50K.':0, '>50K.':1})

    # # normalizing numerical features
    # numeric_features = train_data.select_dtypes(include=['int64', 'float64']).columns
    # scaler = MinMaxScaler()
    # train_data[numeric_features] = scaler.fit_transform(train_data[numeric_features])
    # test_data[numeric_features] = scaler.fit_transform(test_data[numeric_features])

    # # one-hot encoding categorical features
    # categorical_features = train_data.select_dtypes(include=['object']).columns
    # train_data = pd.get_dummies(train_data, columns=categorical_features)
    # test_data = pd.get_dummies(test_data, columns=categorical_features)

    # # converting to numpy arrays
    # train_data = train_data.to_numpy()
    # train_target = train_target.to_numpy()
    # test_data = test_data.to_numpy()
    # test_target = test_target.to_numpy()

    # # feature selection using information gain
    # information_gain = mutual_info_classif(train_data, train_target)
    # # sorting the features based on their information gain
    # sorted_features = np.argsort(information_gain)[::-1]
    # # selecting top features
    # top_features = sorted_features[:k]

    # # selecting top features
    # train_data = train_data[:, top_features]
    # test_data = test_data[:, top_features]

    # return train_data, train_target, test_data, test_target



def credit_data_preprocessing(data_source, k = 10, smaller_data = True):
    data = pd.read_csv(data_source)
    # print(data.shape) # data size: (284807, 31)
    # if smaller_data: we work with a subset of the data, but we must keep all positives ie class = 1
    if smaller_data:
        data_0 = data[data['Class'] == 0] # 284315
        data_1 = data[data['Class'] == 1] # 492
        data_0 = data_0.sample(n=20000, random_state=42)
        data = pd.concat([data_0, data_1], axis=0)
    print(data.shape) # data size: (10292, 31)

    # for each feature(column), checking the unique values and their counts
    # for col in data.columns:
    #     print(data[col].value_counts())
    # missing values: none
    # numerical: all
    # categorical: none

    # train-test split
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # handling missing values: not required

    return basic_preprocessing(train_data, test_data, k, 'Class', '0', '1')

    # separating features and target
    # train_target = train_data['Class']
    # train_data = train_data.drop('Class', axis=1)
    # test_target = test_data['Class']
    # test_data = test_data.drop('Class', axis=1)

    # # normalizing numerical features
    # numeric_features = train_data.select_dtypes(include=['int64', 'float64']).columns
    # print(len(numeric_features))
    # scaler = MinMaxScaler()
    # train_data[numeric_features] = scaler.fit_transform(train_data[numeric_features])
    # test_data[numeric_features] = scaler.fit_transform(test_data[numeric_features])

    # # one-hot encoding categorical features: not required
    # categorical_features = train_data.select_dtypes(include=['object']).columns
    # print(len(categorical_features))

    # # converting to numpy arrays
    # train_data = train_data.to_numpy()
    # train_target = train_target.to_numpy()
    # test_data = test_data.to_numpy()
    # test_target = test_target.to_numpy()

    # # feature selection using information gain
    # information_gain = mutual_info_classif(train_data, train_target)
    # # sorting the features based on their information gain
    # sorted_features = np.argsort(information_gain)[::-1]
    # # selecting top features
    # top_features = sorted_features[:k]
    # # selecting data with top features
    # train_data = train_data[:, top_features]
    # test_data = test_data[:, top_features]

    # return train_data, train_target, test_data, test_target
    

def basic_preprocessing(train_data, test_data, top_features_k, target, target_0, target_1):
    """
    takes train and test data (missing-values must be handled)
    performs normalization, one-hot encoding, feature selection beased on information gain
    and returns preprocessed train and test data
    """
    # separating features and target
    train_target = train_data[target]
    train_data = train_data.drop(target, axis=1)
    test_target = test_data[target]
    test_data = test_data.drop(target, axis=1)

    # normalizing numerical features
    numeric_features = train_data.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_features) > 0:
        scaler = MinMaxScaler()
        train_data[numeric_features] = scaler.fit_transform(train_data[numeric_features])
        test_data[numeric_features] = scaler.fit_transform(test_data[numeric_features])

    # one-hot encoding categorical features
    categorical_features = train_data.select_dtypes(include=['object']).columns
    if len(categorical_features) > 0:
        train_data = pd.get_dummies(train_data, columns=categorical_features)
        test_data = pd.get_dummies(test_data, columns=categorical_features)

    # encoding target
    train_target = train_target.replace({target_0:0, target_1:1})
    test_target = test_target.replace({target_0:0, target_1:1})

    # converting to numpy arrays
    train_data = train_data.to_numpy()
    train_target = train_target.to_numpy()
    test_data = test_data.to_numpy()
    test_target = test_target.to_numpy()

    # feature selection using information gain
    information_gain = mutual_info_classif(train_data, train_target)
    # sorting the features based on their information gain
    sorted_features = np.argsort(information_gain)[::-1]
    # selecting top features
    top_features = sorted_features[:top_features_k]

    # selecting top features
    train_data = train_data[:, top_features]
    test_data = test_data[:, top_features]

    return train_data, train_target, test_data, test_target


def logistic_regression(X, y, learning_rate = 0.01, iterations = 5000, weak_learning=False, error_threshold=0.5):
    # X: training data, y: target

    # initializing weights
    weights = np.zeros(X.shape[1] + 1) # +1 for bias

    # adding bias to X
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    # scaler = preprocessing.StandardScaler()
    # X = scaler.fit_transform(X)
    # transform to float
    X = X.astype(float)
    y = y.astype(float)

    # gradient descent
    for i in range(iterations):
        # calculating the dot product of X and weights
        z = np.dot(X, weights)
        sigmoid = np.exp(z) / (1 + np.exp(z)) # y_predicted
        # calculating gradient of loss function w.r.t. weights
        # gradient = (X.T).(y_pred - y) / y.size
        gradient = np.dot(X.T, (sigmoid - y)) / y.size
        # updating weights
        weights -= learning_rate * gradient
        # calculating loss using mean squared error
        loss = np.mean(-y * np.log(sigmoid) - (1 - y) * np.log(1 - sigmoid))   

        # stopping the algorithm if the error is less than error_threshold
        if weak_learning and loss < error_threshold:
            break
    
    return weights
    

def predict(X, weights):
    # X: data, weights: weights
    # returns: predictions

    # adding bias to X
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    
    predictions = []

    for i in range(X.shape[0]):
        # calculating the dot product of X and weights
        z = np.dot(X[i], weights)
        sigmoid = np.exp(z) / (1 + np.exp(z))
        # predicting the class
        prediction = 1 if sigmoid >= 0.5 else 0
        predictions.append(prediction)
    
    return predictions



def adaptive_boosting(X, y, k=10, learning_rate = 0.01, iterations = 5000, weak_learning = False, error_threshold = 0.5):
    # X: training data, y: target
    # k: number of weak learners
    # local variables: h : a vector of weak learners, z : weights of weak learners
    # weak learner: logistic regression

    # initializing weights with 1/N
    weights = np.ones(X.shape[0]) / X.shape[0]
    h = []
    z = np.zeros(k)

    # training k weak learners
    for i in range(k):
        # resampling data
        indices = np.random.choice(X.shape[0], X.shape[0], p=weights)
        temp_X = X[indices]
        temp_y = y[indices]
        
        # examples = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        # data = examples[np.random.choice(examples.shape[0], examples.shape[0], replace=True, p=weights)]
        # data_X = data[:, :X.shape[1]]
        # data_y = data[:, -1:]

        # training a weak learner
        h.append(logistic_regression(temp_X, temp_y, learning_rate, iterations, weak_learning, error_threshold))
        # predicting the class
        prediction = predict(X, h[i])
        # calculating error
        error = 0
        for j in range(len(prediction)):
            if prediction[j] != y[j]: error += weights[j]
        # calculating weights
        #print(error)
        if error > 0.5: continue
        # updating weights
        for j in range(len(prediction)):
            if prediction[j] == y[j]: weights[j] = weights[j] * error / (1 - error)
        # normalizing weights
        weights = weights / np.sum(weights)
        z[i] = np.log((1 - error) / (error + np.finfo(float).eps))
    
    
    return h, z


def predict_single(datapoint, weight, weak_learning = False):
    """
    returns a prediction for a single data point
    """
    # datapoint: a single data point, weight: a single weight vector
    datapoint = np.concatenate([datapoint, [1]])
    z = np.dot(datapoint, weight)
    sigmoid = np.exp(z) / (1 + np.exp(z))
    prediction = np.round(sigmoid)

    return prediction


def predict_adaboost(X, h, z):
    # X: data, h: weak learners, z: weights of weak learners
    # returns: predictions based on weighted majority voting

    prediction = [] # a vector of predictions
    for i in range(X.shape[0]):
        sum = 0
        for j in range(len(h)):
            sum += z[j] * predict_single(X[i], h[j])
        prediction.append(np.sign(sum)) # signum function : returns -1 if sum < 0, 0 if sum = 0, 1 if sum > 0

    return prediction


def weighted_majority_predict(X, hypotheses, hypothesis_weights):
    num_samples = X.shape[0]
    num_hypotheses = len(hypotheses)
    
    # normalizing inputs X
    # X = normalize(X)
    X = X.astype(float)
    
    # calculating hypotheses
    y_predicteds = []
    
    for i in range(num_hypotheses):
        y_predicted = predict(X, hypotheses[i])
        y_predicteds.append([1 if y_pred == 1 else -1 for y_pred in y_predicted])
        
    y_predicteds = np.array(y_predicteds)
    
    # calculating weighted majority hypothesis and storing predictions
    weighted_majority_hypothesis = np.dot(y_predicteds.T, hypothesis_weights)
    predictions = [1 if y_pred > 0 else 0 for y_pred in weighted_majority_hypothesis]

    return predictions



def performance_metrics(prediction, target):
    """
    calculates accuracy, recall, specificity, precision and f1 score
    """
    # calculating true positive, true negative, false positive, false negative
    # true_positive = 0
    # true_negative = 0
    # false_positive = 0
    # false_negative = 0
    # for i in range(len(prediction)):
    #     if prediction[i] == 1 and target[i] == 1: true_positive += 1
    #     elif prediction[i] == 1 and target[i] == 0: false_positive += 1
    #     elif prediction[i] == 0 and target[i] == 1: false_negative += 1
    #     elif prediction[i] == 0 and target[i] == 0: true_negative += 1
    
    # # calculating accuracy, recall, specificity, precision and f1 score
    # accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    # recall = true_positive / (true_positive + false_negative)
    # specificity = true_negative / (true_negative + false_positive)
    # precision = true_positive / (true_positive + false_positive)
    # f1 = 2 * precision * recall / (precision + recall)

    # using sklearn.metrics
    accuracy = accuracy_score(target, prediction)
    recall = recall_score(target, prediction)
    specificity = recall_score(target, prediction, pos_label=0)
    precision = precision_score(target, prediction)
    false_discovery_rate = 1 - precision
    f1= f1_score(target, prediction)

    print('Accuracy: ', accuracy)
    print('Recall: ', recall)
    print('Specificity: ', specificity)
    print('Precision: ', precision)
    print('False Discovery Rate: ', false_discovery_rate)
    print('F1 Score: ', f1)
    
    return accuracy, recall, specificity, precision, false_discovery_rate, f1



def logistic_learning_test(data_type, data_path, learning_rate = 0.01, iterations = 5000, num_features = 10, weak_learning = False, error_threshold = 0.5, smaller_data=True):
    """
    evaluates the performance of logistic regression
    """
    print("\n============ Logistic Learning ============")
    if data_type == '1': # telco data
        print('-------- Telco Data --------')
        train_data, train_target, test_data, test_target = telco_data_preprocessing(data_path, num_features)
    elif data_type == '2': # adult data
        print('-------- Adult Data --------')
        train_data, train_target, test_data, test_target = adult_data_preprocessing(num_features)
    elif data_type == '3': # credit card data
        print('-------- Credit Card Data --------')
        train_data, train_target, test_data, test_target = credit_data_preprocessing(data_path, num_features, smaller_data)

    # training
    print(train_data.shape, train_target.shape)
    weights = logistic_regression(train_data, train_target, learning_rate, iterations, weak_learning, error_threshold)
    # prediction on train data
    prediction_train_data = predict(train_data, weights)
    print("---------- On train data ----------")
    performance_metrics(prediction_train_data, train_target)
    # prediction on test data
    prediction_test_data = predict(test_data, weights)
    print("---------- On test data ----------")
    performance_metrics(prediction_test_data, test_target)



def adaptive_boosting_test(data_type, data, k_vals = [5, 10, 15, 20], learning_rate = 0.01, iterations = 5000, num_features = 10, weak_learning = False, error_threshold=0.5, smaller_data = True):
    """
    evaluates the performance of adaboost
    """
    print("\n============ Adaptive Boosting ============")
    if data_type == '1': print('-------- Telco Data --------')
    elif data_type == '2': print('-------- Adult Data --------')
    elif data_type == '3': print('-------- Credit Card Data --------')
    #k_vals= [5, 10, 15, 20]
    for k in k_vals:
        if data_type == '1': # telco data
            train_data, train_target, test_data, test_target = telco_data_preprocessing(data, num_features)
        elif data_type == '2': # adult data
            train_data, train_target, test_data, test_target = adult_data_preprocessing(num_features)
        elif data_type == '3': # credit card data)
            train_data, train_target, test_data, test_target = credit_data_preprocessing(data, num_features, smaller_data)

        # training
        h, z = adaptive_boosting(train_data, train_target, k, learning_rate, iterations, weak_learning, error_threshold)
        print("\n\nk = ", k)
        # prediction on train data
        prediction_train_data = weighted_majority_predict(train_data, h, z)
        print("---------- On train data ----------")
        performance_metrics(prediction_train_data, train_target)
        # prediction on test data
        prediction_test_data = weighted_majority_predict(test_data, h, z)
        print("---------- On test data ----------")
        performance_metrics(prediction_test_data, test_target)


def adabost_test_2(data_type, data, k = 10, num_features = 10, learning_rate = 0.01, iterations = 5000, weak_learning = False, error_threshold = 0.5, smaller_data = True):

        print("\n============ Adaboost ============")
        if data_type == '1': # telco data
            print('-------- Telco Data --------')
            train_data, train_target, test_data, test_target = telco_data_preprocessing(data, num_features)
        elif data_type == '2': # adult data
            print('-------- Adult Data --------')
            train_data, train_target, test_data, test_target = adult_data_preprocessing(num_features)
        elif data_type == '3': # credit card data
            print('-------- Credit Card Data --------')
            train_data, train_target, test_data, test_target = credit_data_preprocessing(data, num_features, smaller_data)

        # training
        h, z = adaptive_boosting(train_data, train_target, k, learning_rate, iterations, weak_learning, error_threshold)
        # prediction on train data
        prediction_train_data = weighted_majority_predict(train_data, h, z)
        print("---------- On train data ----------")
        performance_metrics(prediction_train_data, train_target)
        # prediction on test data
        prediction_test_data = weighted_majority_predict(test_data, h, z)
        print("---------- On test data ----------")
        performance_metrics(prediction_test_data, test_target)


# using library functions
def library_logistic_learning(data_type, data, test_data = None, smaller_data = True):
    if data_type == '1': # telco data
        train_data, train_target, test_data, test_target = telco_data_preprocessing(data, k=10)
    elif data_type == '2': # adult data
        train_data, train_target, test_data, test_target = adult_data_preprocessing(data, test_data, k=10)
    elif data_type == '3': # credit card data
        train_data, train_target, test_data, test_target = credit_data_preprocessing(data, k=10, smaller_data=smaller_data)
    # training
    classifier = LogisticRegression()
    classifier.fit(train_data, train_target)
    # predicting
    prediction = classifier.predict(test_data)

    print("\n======= Using library functions on Test data =======")
    performance_metrics(prediction, test_target)


def library_adaptive_boost_test(data_type, data, test_data = None, smaller_data = True):
    if data_type == '1': # telco data
        train_data, train_target, test_data, test_target = telco_data_preprocessing(data, k=10)
    elif data_type == '2': # adult data
        train_data, train_target, test_data, test_target = adult_data_preprocessing(data, test_data, k=10)
    elif data_type == '3': # credit card data
        train_data, train_target, test_data, test_target = credit_data_preprocessing(data, k=10, smaller_data=smaller_data)
    # training
    classifier = AdaBoostClassifier(n_estimators=10, random_state=42)
    classifier.fit(train_data, train_target)
    # predicting
    prediction = classifier.predict(test_data)

    print("\n======= Using library functions on Test data =======")
    performance_metrics(prediction, test_target)




if __name__ == '__main__':

    # data - global variables
    global telco_data_source, adult_train_data_source, adult_test_data_source, credit_card_data_source

    # note: data is zipped in data.zip file

    telco_data_source = 'data/WA_Fn-UseC_-Telco-Customer-Churn.xls'
    adult_train_data_source = 'data/adult/adult.data'
    adult_test_data_source = 'data/adult/adult.test'
    credit_card_data_source = 'data/creditcard.csv/creditcard.csv'

    # parameters for logistic regression and adaboost
    learning_rate = 0.01
    iterations = 5000
    num_features = 15
    k_vals = [5, 10, 15, 20]
    weak_learning = False
    error_threshold = 0.5
    smaller_data = True

    # --------------------- our functions: logistic ---------------------
    logistic_learning_test('1', telco_data_source, learning_rate, iterations, num_features, weak_learning, error_threshold, smaller_data=False)
    #logistic_learning_test('2', adult_train_data_source, learning_rate, iterations, num_features, weak_learning, error_threshold, smaller_data=False)
    #logistic_learning_test('3', credit_card_data_source, learning_rate, iterations = 20000, num_features = 10, weak_learning=False, error_threshold=0.3, smaller_data=True)
    #logistic_learning_test('3', credit_card_data_source, learning_rate, iterations = 20000, num_features = 10, weak_learning=False, error_threshold=0.3, smaller_data=False)

    # --------------------- our functions: adaboost ---------------------
    #adaptive_boosting_test('1', telco_data_source, k_vals, learning_rate, iterations, num_features, weak_learning, error_threshold, smaller_data=False)
    #adaptive_boosting_test('2', adult_train_data_source, k_vals, learning_rate, iterations, num_features, weak_learning, error_threshold, smaller_data=False)
    #adaptive_boosting_test('3', credit_card_data_source, k_vals, learning_rate, iterations, num_features, weak_learning, error_threshold, smaller_data=True)
    #adaptive_boosting_test('3', credit_card_data_source, k_vals, learning_rate, iterations, num_features, weak_learning, error_threshold, smaller_data=False)
