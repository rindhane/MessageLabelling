import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
#from sklearn.externals import joblib
from sqlalchemy import create_engine
import pickle

#key values:
fp=open('../models/model_details.txt','r')
data=json.load(fp)
databaseTable=data['table_name']
databaseLoc=data['database_filepath']
modelLoc=data['model_filepath']
fp.close()

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine(f'sqlite:///../{databaseLoc}')
df = pd.read_sql_table(databaseTable, engine)

# load model
#model = joblib.load(f"../{modelLoc}")
model=pickle.load(open(f"../{modelLoc}",'rb'))

#get_columns_df
def get_columns(df):
    cols=list(df.columns)
    for item in ['id','message','original','genre'] :
        cols.remove(item)
    return cols

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    ##Generating data for pie chart
    def pi_data(df=df):
        '''helper function to generate
        counts of messages in particulat category'''
        cols=get_columns(df)
        pi_labels=cols
        pi_values=list()
        total=len(df)
        for col in cols:
            pi_values.append(
                df[df[col]==1][col].sum())
        return (pi_labels,pi_values)
    ##Generating data for stacked bar chart 
    def stacked_bar(df=df):
        cols=get_columns(df)
        x_axis=cols
        df=df[cols]
        tmp=[df[cols].groupby(df[cols]==0).count().tolist() 
                        for cols in df]
        associated = list()
        not_associated=list()
        for dat in tmp:
            if len(dat)==1:
                associated.append(0)
                not_associated.append(dat[0])
            else:
                associated.append(dat[0])
                not_associated.append(dat[1])
            
        return  x_axis, associated, x_axis, not_associated
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Pie(
                    values=pi_data()[1],
                    labels=pi_data()[0]
                )
            ],
            'layout': {
                'title': 'Composition of Messages under each category',
                'name' : 'Category'

            }
        },
        {
            'data': [
                Bar(
                    x=stacked_bar()[0],
                    y=stacked_bar()[1],
                    name= 'Associated',
                ),
                Bar(
                    x=stacked_bar()[2],
                    y=stacked_bar()[3],
                    name= 'Not Associated',
                ),
                
            ],

            'layout': {
                'barmode' : 'stack',
                'title': 'Availability of each Category in Message Datset ',
                'yaxis': {
                    'title': "Count of presence of each category"
                },
                'xaxis': {
                    'title': "Categories "
                }
            }
        },
    ]
    
    # encode plotly graphs in JSON
    ids = [["graph-{}".format(i),s['layout']['title']] for i, s in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)



# web page that handles user query and displays modelresults
@app. route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=8080, debug=True)


if __name__ == '__main__':
    main()