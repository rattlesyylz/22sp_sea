import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

# models
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
# report
# from sklearn.metrics import roc_auc_score, confusion_matrix,
# f1_score, accuracy_score
from sklearn.metrics import classification_report


# Information about the dataset
def check_info(df):
    df=df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)
    print("---------------  Info ---------------")
    print(df.info())
    print("--------------- Shape ---------------")
    print(df.shape)
    print("--------------- Types ---------------")
    print(df.dtypes)
    print("--------------- Head ---------------")
    print(df.head())
    print("--------------- Tail ---------------")
    print(df.tail())
    print("--------------- NA ---------------")
    print(df.isnull().sum())
    print("--------------- Unique ---------------")
    print(df.nunique())
    return df


def geography(df):
    shp_df = gpd.read_file('/Users/NextStop/documents/project/geo_data \
                            /ne_110m_admin_0_countries.shp')
    europe_shp_df = shp_df[shp_df['CONTINENT'] == 'Europe']
    geo_df = europe_shp_df.merge(df, left_on='NAME_EN',
                                 right_on='Geography', how='left')
    churn_df = geo_df[['Geography', 'Exited', 'geometry']]

    churn = churn_df.dissolve(by='Geography', aggfunc='mean')
    fig, ax = plt.subplots(1, figsize=(15, 10))

    geo_df.plot(ax=ax, color='#EEEEEE', edgecolor='#FFFFFF')
    churn.plot(ax=ax, column='Exited', legend=True)
    plt.title('Churn by Geography')
    plt.show()


def visualization(df):
    # proportion of churn customers
    exist_cust = [df.Exited[df['Exited'] == 1].count(),
                  df.Exited[df['Exited'] == 0].count()]
    plt.subplots(figsize=(8, 8))
    plt.pie(exist_cust, labels=['Churn', 'Not Churn'], autopct='%.2f%%')
    plt.legend(labels=['Churn', 'Not Churn'], loc="upper right")
    plt.title('Proportion of Customer Churn', size=12)
    plt.show()


def heat_map(df):
    sns.heatmap(df.corr())
    plt.show()


def correlation_categorical(df):
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(20, 10))
    sns.countplot(ax=ax1, data=df, x='Geography', hue='Exited')
    sns.countplot(ax=ax2, data=df, x='Gender', hue='Exited')
    sns.countplot(ax=ax3, data=df, x='HasCrCard', hue='Exited')
    sns.countplot(ax=ax4, data=df, x='IsActiveMember', hue='Exited')
    plt.show()


def correlation_numerical(df):
    fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2, 3,
                                                           figsize=(20, 10))
    sns.boxplot(ax=ax1, data=df, y='CreditScore', x='Exited', hue='Exited')
    sns.boxplot(ax=ax2, data=df, y='Age', x='Exited', hue='Exited')
    sns.boxplot(ax=ax3, data=df, y='Tenure', x='Exited', hue='Exited')
    sns.boxplot(ax=ax4, data=df, y='Balance', x='Exited', hue='Exited')
    sns.boxplot(ax=ax5, data=df, y='NumOfProducts', x='Exited', hue='Exited')
    sns.boxplot(ax=ax6, data=df, y='EstimatedSalary', x='Exited', hue='Exited')
    # df.corr().at['Age', 'Exited']
    plt.show()


def features_labels(df):
    # one-hot encoding on categorical variables
    df=pd.get_dummies(data=df, columns=['Geography', 'Gender'])
    # combine features
    numerical_feature = ['CreditScore', 'Age', 'Tenure', 'Balance',
                         'NumOfProducts', 'EstimatedSalary']

    # min-max scaling
    min_vec = df[numerical_feature].min().copy()
    max_vec = df[numerical_feature].max().copy()
    df[numerical_feature] = (df[numerical_feature] - min_vec) / (max_vec
                                                                 - min_vec)
    features = df.loc[:, df.columns != 'Exited']
    labels = df['Exited']
    return df, features, labels


def logistic_regression(features_train, features_test,
                        labels_train, labels_test):
    # Hyperparameter turning
    lr_model = LogisticRegression(solver='liblinear')
    parameters = {'C': np.arange(0.5, 1.5, 0.1)}
    lr_model_grid = GridSearchCV(lr_model, parameters, cv=10,
                                 n_jobs=-1).fit(features_train, labels_train)
    print('Logistic Regression Best parameters:', lr_model_grid.best_params_)
    lr_para = lr_model_grid.best_params_
    print('Logistic Regression Cross validation score:',
          lr_model_grid.best_score_)
    # Prediction
    logistic_model = LogisticRegression(C=lr_para['C'], solver='liblinear')
    logistic_fit = logistic_model.fit(features_train, labels_train)
    logistic_prediction_train = logistic_fit.predict(features_train)
    # logistic_prediction_test = logistic_fit.predict(features_test)
    # Scores
    print(classification_report(labels_train, logistic_prediction_train))
    # feature importance
    logistic_importance = logistic_model.coef_[0]
    feature_importances(logistic_importance, features_train)


def random_forest(features_train, features_test, labels_train, labels_test):
    # Hyperparameter turning
    rf_model = RandomForestClassifier(n_jobs=-1)
    parameters = {'n_estimators': [100, 200, 300],
                  'max_depth': np.arange(3, 6, 1)}
    rf_model_grid = GridSearchCV(rf_model, parameters, cv=10,
                                 n_jobs=-1).fit(features_train, labels_train)
    rf_para = rf_model_grid.best_params_
    print('Random Forest Classifier Best parameters:', rf_para)
    print('Random Forest Classifier Cross validation score:',
          rf_model_grid.best_score_)
    # Prediction
    forest_model = RandomForestClassifier(n_estimators=rf_para['n_estimators'],
                                          max_depth=rf_para['max_depth'])
    forest_fit = forest_model.fit(features_train, labels_train)
    forest_prediction_train = forest_fit.predict(features_train)
    # forest_prediction_test = forest_fit.predict(features_test)
    # Scores
    print(classification_report(labels_train, forest_prediction_train))
    # feature importance
    forest_importance = forest_model.feature_importances_
    feature_importances(forest_importance, features_train)


def xgboost(features_train, features_test, labels_train, labels_test):
    # Hyperparameters turning
    xg_model = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1)
    parameters = {'n_estimators': [100, 200, 300], 'max_depth':
                  np.arange(3, 6, 1), 'learning_rate': [0.01, 0.1]}
    xg_model_grid = GridSearchCV(xg_model, parameters, cv=10,
                                 n_jobs=-1).fit(features_train, labels_train)
    xg_para = xg_model_grid.best_params_
    print('XGBoost Classifier Best parameters:', xg_para)
    print('XGBoost Classifier Cross validation score:',
          xg_model_grid.best_score_)
    # Prediction
    xgb_model = XGBClassifier(random_state=42, eval_metric='logloss',
                              n_estimators=xg_para['n_estimators'],
                              max_depth=xg_para['max_depth'],
                              learning_rate=xg_para['learning_rate'])
    xgb_fit = xgb_model.fit(features_train, labels_train)
    xgb_prediction_train = xgb_fit.predict(features_train)
    # xgb_prediction_test = xgb_fit.predict(features_test)
    # Scores
    print(classification_report(labels_train, xgb_prediction_train))
    # feature importance
    xgb_importance = xgb_model.feature_importances_
    feature_importances(xgb_importance, features_train)


def svm(features_train, features_test, labels_train, labels_test):
    # Hyperparameter turning
    support_model = SVC(random_state=42, kernel='linear')
    parameters = {'C': np.arange(0.8, 1.3, 0.1), 'degree': np.arange(3, 6, 1)}
    support_model_grid = GridSearchCV(support_model, parameters, cv=10,
                                      n_jobs=-1).fit(features_train,
                                                     labels_train)
    support_para = support_model_grid.best_params_
    print('SVM Classification Best parameters:', support_para)
    print('SVM Classification Cross validation score:',
          support_model_grid.best_score_)
    # Prediction
    svm_model = SVC(random_state=42, C=support_para['C'],
                    degree=support_para['degree'], kernel='linear')
    svm_fit = svm_model.fit(features_train, labels_train)
    svm_prediction_train = svm_fit.predict(features_train)
    # svm_prediction_test = svm_fit.predict(features_test)
    # Scores
    print(classification_report(labels_train, svm_prediction_train))
    # feature importance
    svm_importance = svm_model.coef_[0]
    feature_importances(svm_importance, features_train)


def feature_importances(imp, features):
    print('Model Feature Importance:')
    for i, v in zip(imp, features):
        print('Feature: ', v, ' Score:', i)
    imp, features = zip(*sorted(zip(imp, features)))
    plt.barh(range(len(features)), imp, align='center')
    plt.yticks(range(len(features)), features)
    plt.title('Feature Importance')
    plt.show()


def main():
    # read the file
    churn_df = pd.read_csv('/Users/NextStop/documents/project/churn.csv')
    # call functions
    df = check_info(churn_df)
    # visualization(df)
    # heat_map(df)
    # geography(df)
    # correlation_categorical(df)
    # correlation_numerical(df)
    # features and label
    df, features, labels = features_labels(df)
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    # logistic_regression(features_train, features_test,
    # labels_train, labels_test)
    # random_forest(features_train, features_test, labels_train, labels_test)
    # xgboost(features_train, features_test, labels_train, labels_test)
    svm(features_train, features_test, labels_train, labels_test)


if __name__ == '__main__':
    main()
