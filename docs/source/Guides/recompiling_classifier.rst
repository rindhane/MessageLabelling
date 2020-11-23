Recompiling classifier with updated data
=============================================

1. Put your data updated into an csv file for eg : 'messages.csv'
2. The categorically labelled details of this messages to be placed in another csv for eg : 'categories.csv'
3. In your terminal move to the project's root folder [Project]
4. Follow following instructions : 

### Basic Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/