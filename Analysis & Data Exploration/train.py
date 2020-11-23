# Note : Packages are imported within individaual functions
#This style of importing helps to reduce polluting global stack
#This reduces the size of pickle file


def load_data(loc,table='categorizedMessages'):
  #imports
  import pandas as pd 
  from sqlalchemy import create_engine
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

def display_results( y_test, y_pred, cv=None):
  '''helper function to plot resultsc'''
  print("\nBest Parameters:", cv.best_params_)
  from sklearn.metrics import confusion_matrix
  from sklearn.metrics import classification_report
  accuracy = (y_pred == y_test).mean()
  print('Accuracy Results in each Category \n \n',accuracy)
  print('------------------------')
  for a,b in zip(y_test, y_pred.transpose()):
    confusion_mat = confusion_matrix(y_test[a].values, b)
    print("categories:"+str(a)+'\n',confusion_mat)
    print('\n')


def train(X, y, model):
  from sklearn.model_selection import train_test_split
  # train test split
  X_train, X_test, y_train, y_test = train_test_split(X, y)
  # fit model
  model.fit(X_train, y_train)
  # output model test results
  y_pred = model.predict(X_test)
  display_results( y_test, y_pred, cv=model)
  return model


def export_model(model,filename):
  import pickle
  fp=open(filename,'wb')
  pickle.dump(model,fp)
  return True


def run_pipeline(data_file,filename):
    X, y = load_data(data_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, y, model)  # train model pipeline
    export_model(model,filename)  # save model
    return True

def safe_list_get (l, idx, default):
  try:
    return l[idx]
  except IndexError:
    return default

if __name__ == '__main__':
  import sys
  data_file = sys.argv[1]  # get filename of dataset
  run_pipeline(data_file,safe_list_get(sys.argv,2,'model.pkl'))  # run data pipeline