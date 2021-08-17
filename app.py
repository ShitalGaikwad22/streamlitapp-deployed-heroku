import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import warnings
warnings.filterwarnings("ignore")
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Credit Card Fraud Detection')

df =st.cache(pd.read_csv)('creditcard.csv')

#df = df.sample(frac=0.1, random_state = 48)


#SHAPE AND DESCRIPTION OF DATASET
if st.sidebar.checkbox('Take a look at the DataFrame'):
    st.header('Shape and description of data:')
    st.write(df.head(50))
    st.write('Shape of the dataframe: ',df.shape)
    st.write('Data description: \n',df.describe())


#GENUINE AND FRAUD TRANSACTIONS FROM DATASET
fraud = df[df['Class']==1]
genuine =df[df['Class']==0]

fraud_percentage=(df.Class.value_counts()[1]/df.Class.value_counts()[0])*100

if st.sidebar.checkbox("'Fraud'and 'Genuine' transaction details"):
    st.header('Transaction Details:')
    st.write('Fraudulent transactions are:{:.3f} %'.format(fraud_percentage))
    st.write('Fraud transactions: ',len(fraud))
    st.write('Genuine transactions ',len(genuine))


#target-y and features-X
X= df.drop(['Class'],axis=1)
y= df['Class']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
size = st.sidebar.slider('Set test-data size', min_value=0.2, max_value=0.4)

X_train,X_test ,y_train,y_test = train_test_split(X,y,test_size=size,random_state=42)

#print shape of training and testing data
if st.sidebar.checkbox('Shape of training and test data'):
    st.header('Shape of training and test data:')
    st.write('X_train: ',X_train.shape)
    st.write('y_train: ',y_train.shape)
    st.write('X_test: ',X_test.shape)
    st.write('y_test: ',y_test.shape)

#Import classification models and metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score

logreg=LogisticRegression()
svm=SVC()
knn=KNeighborsClassifier()
etree=ExtraTreesClassifier(random_state=42)
rforest=RandomForestClassifier(random_state=42)

features=X_train.columns.tolist()

#FEATURE ENGINEERING

#Selecting features through feature importance

@st.cache
def feature_sort(model,X_train,y_train):
    #feature selection
    mod=model
    # fit the model
    mod.fit(X_train, y_train)
    # get importance
    impList = mod.feature_importances_
    return impList

#Classifiers for feature importance
clf = ['Extra Trees','Random Forest']
featImpModel = st.sidebar.selectbox('Select model for feature importance', clf)

start_time = timeit.default_timer()

if featImpModel=='Extra Trees':
    model= etree
    important_features = feature_sort(model,X_train,y_train)
elif featImpModel=='Random Forest':
    model=rforest
    important_features= feature_sort(model,X_train,y_train)
#time required for the feature importance process
elapsed = timeit.default_timer() - start_time
st.write('Execution Time for feature selection: %.2f minutes'%(elapsed/60))


#Plot of feature importance
if st.sidebar.checkbox('Feature Importance Plot'):
    plt.bar([x for x in range(len(important_features))], important_features)
    plt.title('Feature Importance')
    plt.xlabel('Feature (Variable Number)')
    plt.ylabel('Importance')
    st.pyplot()

feature_imp = list(zip(features,  important_features))
features_sort = sorted(feature_imp, key=lambda x: x[1])

n_top_features = st.sidebar.slider('Number of top features', min_value=5, max_value=20)

top_features = list(list(zip(*features_sort[-n_top_features:]))[0])

if st.sidebar.checkbox('Show selected top features'):
    st.write('Top %d features in order of importance are: %s' % (n_top_features, top_features[::-1]))

X_train_imp = X_train[top_features]
X_test_imp = X_test[top_features]

X_train_imp_scaled = X_train_imp
X_test_imp_scaled = X_test_imp

# Import performance metrics, imbalanced rectifiers
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

np.random.seed(42)  # for reproducibility since SMOTE and Near Miss use randomizations

oversample = SMOTE()

nr = NearMiss()


def compute_performance(model, X_train, y_train, X_test, y_test):
    start_time = timeit.default_timer()
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
    'Accuracy: ', scores
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    'Confusion Matrix: ', cm
    cr = classification_report(y_test, y_pred)
    'Classification Report: ', cr
    elapsed = timeit.default_timer() - start_time
    'Execution Time for performance computation: %.2f minutes' % (elapsed / 60)


# Run different classification models with rectifiers
if st.sidebar.checkbox('Run the credit card fraud detection model'):

    alg = ['Extra Trees', 'Random Forest', 'k Nearest Neighbor', 'Support Vector Machine', 'Logistic Regression']
    classifier = st.sidebar.selectbox('Choose algorithm:', alg)
    rectifier = ['SMOTE', 'Near Miss', 'No Rectifier']
    imb_rect = st.sidebar.selectbox('Choose imbalanced class rectifier:', rectifier)

    if classifier == 'Logistic Regression':
        model = logreg
        if imb_rect == 'No Rectifier':
            compute_performance(model, X_train_imp_scaled, y_train, X_test_imp_scaled, y_test)
        elif imb_rect == 'SMOTE':
            rect = oversample
            st.write('Shape of imbalanced y_train data: ', np.bincount(y_train))
            X_train_bal, y_train_bal = rect.fit_resample(X_train_imp_scaled, y_train)
            st.write('Shape of balanced y_train: ', np.bincount(y_train_bal))
            compute_performance(model, X_train_bal, y_train_bal, X_test_imp_scaled, y_test)
        elif imb_rect == 'Near Miss':
            rect = nr
            st.write('Shape of imbalanced y_train: ', np.bincount(y_train))
            X_train_bal, y_train_bal = rect.fit_sample(X_train_imp_scaled, y_train)
            st.write('Shape of balanced y_train: ', np.bincount(y_train_bal))
            compute_performance(model, X_train_bal, y_train_bal, X_test_imp_scaled, y_test)


    elif classifier == 'k Nearest Neighbor':
        model = knn
        if imb_rect == 'No Rectifier':
            compute_performance(model, X_train_imp_scaled, y_train, X_test_imp_scaled, y_test)
        elif imb_rect == 'SMOTE':
            rect = oversample
            st.write('Shape of imbalanced y_train: ', np.bincount(y_train))
            X_train_bal, y_train_bal = rect.fit_resample(X_train_imp_scaled, y_train)
            st.write('Shape of balanced y_train: ', np.bincount(y_train_bal))
            compute_performance(model, X_train_bal, y_train_bal, X_test_imp_scaled, y_test)
        elif imb_rect == 'Near Miss':
            rect = nr
            st.write('Shape of imbalanced y_train: ', np.bincount(y_train))
            X_train_bal, y_train_bal = rect.fit_sample(X_train_imp_scaled, y_train)
            st.write('Shape of balanced y_train: ', np.bincount(y_train_bal))
            compute_performance(model, X_train_bal, y_train_bal, X_test_imp_scaled, y_test)

    elif classifier == 'Support Vector Machine':
        model = svm
        if imb_rect == 'No Rectifier':
            compute_performance(model, X_train_imp_scaled, y_train, X_test_imp_scaled, y_test)
        elif imb_rect == 'SMOTE':
            rect = oversample
            st.write('Shape of imbalanced y_train: ', np.bincount(y_train))
            X_train_bal, y_train_bal = rect.fit_resample(X_train_imp_scaled, y_train)
            st.write('Shape of balanced y_train: ', np.bincount(y_train_bal))
            compute_performance(model, X_train_bal, y_train_bal, X_test_imp_scaled, y_test)
        elif imb_rect == 'Near Miss':
            rect = nr
            st.write('Shape of imbalanced y_train: ', np.bincount(y_train))
            X_train_bal, y_train_bal = rect.fit_sample(X_train_imp_scaled, y_train)
            st.write('Shape of balanced y_train: ', np.bincount(y_train_bal))
            compute_performance(model, X_train_bal, y_train_bal, X_test_imp_scaled, y_test)

    elif classifier == 'Random Forest':
        model = rforest
        if imb_rect == 'No Rectifier':
            compute_performance(model, X_train_imp_scaled, y_train, X_test_imp_scaled, y_test)
        elif imb_rect == 'SMOTE':
            rect = oversample
            st.write('Shape of imbalanced y_train: ', np.bincount(y_train))
            X_train_bal, y_train_bal = rect.fit_resample(X_train_imp_scaled, y_train)
            st.write('Shape of balanced y_train: ', np.bincount(y_train_bal))
            compute_performance(model, X_train_bal, y_train_bal, X_test_imp_scaled, y_test)
        elif imb_rect == 'Near Miss':
            rect = nr
            st.write('Shape of imbalanced y_train: ', np.bincount(y_train))
            X_train_bal, y_train_bal = rect.fit_sample(X_train_imp_scaled, y_train)
            st.write('Shape of balanced y_train: ', np.bincount(y_train_bal))
            compute_performance(model, X_train_bal, y_train_bal, X_test_imp_scaled, y_test)

    elif classifier == 'Extra Trees':
        model = etree
        if imb_rect == 'No Rectifier':
            compute_performance(model, X_train_imp_scaled, y_train, X_test_imp_scaled, y_test)
        elif imb_rect == 'SMOTE':
            rect = oversample
            st.write('Shape of imbalanced y_train: ', np.bincount(y_train))
            X_train_bal, y_train_bal = rect.fit_resample(X_train_imp_scaled, y_train)
            st.write('Shape of balanced y_train: ', np.bincount(y_train_bal))
            compute_performance(model, X_train_bal, y_train_bal, X_test_imp_scaled, y_test)
        elif imb_rect == 'Near Miss':
            rect = nr
            st.write('Shape of imbalanced y_train: ', np.bincount(y_train))
            X_train_bal, y_train_bal = rect.fit_sample(X_train_imp_scaled, y_train)
            st.write('Shape of balanced y_train: ', np.bincount(y_train_bal))
            compute_performance(model, X_train_bal, y_train_bal, X_test_imp_scaled, y_test)