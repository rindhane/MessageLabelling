import sys
from sklearn.model_selection import train_test_split
import json

# Note : Packages are imported within individaual functions
#This style of importing helps to reduce polluting global stack
#This reduces the size of pickle file

#important constant
tableName='categorizedMessages'

def load_data(database_filepath,table=tableName):
  #imports
  import pandas as pd 
  from sqlalchemy import create_engine
  engine = create_engine(f'sqlite:///{database_filepath}')
  conn=engine.connect()
  df = pd.read_sql_table(table_name=table, con=conn)
  # define features and label arrays
  X = df['message']
  cols=list(df.columns)
  for item in ['id','message','original','genre'] :
      cols.remove(item)
  Y = df[cols].values
  return X, Y, cols 


def tokenize(text):
  '''function to create word tokens for text '''
  import re
  import nltk
  from nltk.corpus import stopwords
  from nltk.stem.wordnet import WordNetLemmatizer
  from nltk.tokenize import word_tokenize
  nltk.download('punkt',quiet=True)
  nltk.download('stopwords',quiet=True)
  nltk.download('wordnet',quiet=True)

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
  from sklearn.pipeline import Pipeline
  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn.feature_extraction.text import TfidfTransformer
  from sklearn.model_selection import GridSearchCV
  from sklearn.metrics import make_scorer
  #sklearn ML libraries
  from sklearn.neighbors import KNeighborsClassifier
  #from sklearn.ensemble import RandomForestClassifier
  pipeline = Pipeline([
      ('count',CountVectorizer(tokenizer=tokenize)),
      ('tfid',TfidfTransformer()),
      ('clf', KNeighborsClassifier()),
  ])
  #  parameters for GridSearchCV
  parameters= {
      'clf__n_neighbors': [1,2,3,4,5], # 200,300,400,600],
  }
  # gridsearch object and return as final model pipeline
  pipeline=GridSearchCV(pipeline,
                  parameters,
                  scoring=make_scorer(scoring_method))
  return pipeline



def evaluate_model(model, X_test, Y_test, category_names):
    from sklearn.metrics import classification_report
    Y_pred=model.predict(X_test)
    print('--------------------------------')
    print('Evaluation Report')
    print('---------------------------------')
    for a,b,c in zip(Y_test.transpose(), Y_pred.transpose(),category_names):
        print("Category : "+str(c), ' & its result:'+'\n',
             classification_report(a,b))
        print('\n')
    return True


def save_model(model, model_filepath):
  import pickle
  fp=open(model_filepath,'wb')
  pickle.dump(model,fp)
  return True


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        model_details = {'model_filepath' :  model_filepath ,
                          'table_name':tableName,
                         'database_filepath': database_filepath,}
        fp=open('models/model_details.txt','w')
        json.dump(model_details,fp)
        fp.close()

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()