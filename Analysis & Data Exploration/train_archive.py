# import packages
# import for data loading and text handling
import sys
import pandas as pd 
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
#nltk library imports
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#sklearn libraries
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix,make_scorer
from sklearn.metrics import classification_report
#sklearn ML libraries
from sklearn.ensemble import RandomForestClassifier

def load_data(loc,table='categorizedMessages'):
    # read in file
    # clean data
    # load to database
    engine = create_engine(f'sqlite:///{loc}')
    conn=engine.connect()
    df = pd.read_sql_table(table_name=table, con=conn)
    # define features and label arrays
    X = df['message']
    cols=list(df.columns)
    for item in ['id','message','original','genre'] :
        cols.remove(item)
    Y = df[cols]
    return X, Y

def tokenize(text):
    '''function to create word tokens for text '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return tokens

def scoring_method(y_pred,y_test):
    '''function to score results by GridSearch'''
    return (y_test==y_pred).mean().min()


def build_model():
    # text processing and model pipeline
    pipeline = Pipeline([
        ('count',CountVectorizer(tokenizer=tokenize)),
        ('tfid',TfidfTransformer()),
        ('clf',RandomForestClassifier(n_estimators=200, min_samples_split=3)),
    ])
    #  parameters for GridSearchCV
    '''parameters= {
        'clf__n_estimators': [50, 100 ,], # 200,300,400,600],
        'clf__min_samples_split': [2, 3,]# 4],
    }'''
    # gridsearch object and return as final model pipeline
    '''pipeline=GridSearchCV(pipeline,
                    parameters,
                    scoring=make_scorer(scoring_method))'''

    return pipeline

def display_results( y_test, y_pred, cv=None):
    '''helper function to plot resultsc'''
    #print("\nBest Parameters:", cv.best_params_)
    accuracy = (y_pred == y_test).mean()
    print('Accuracy Results in each Category \n \n',accuracy)
    print('------------------------')
    for a,b in zip(y_test, y_pred.transpose()):
        confusion_mat = confusion_matrix(y_test[a].values, b)
        print("categories:"+str(a)+'\n',confusion_mat)
        print('\n')


def train(X, y, model):
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # fit model
    model.fit(X_train, y_train)
    # output model test results
    y_pred = model.predict(X_test)
    display_results( y_test, y_pred, cv=None)
    return model


def export_model(model,filename='model.pkl'):
    fp=open(filename,'wb')
    fp.write(pickle.dumps(model))
    fp.close()
    return True


def run_pipeline(data_file):
    X, y = load_data(data_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, y, model)  # train model pipeline
    export_model(model)  # save model
    return True


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline