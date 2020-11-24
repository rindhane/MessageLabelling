Recompiling classifier with updated data
=============================================

1. Put your data updated into an csv file for eg : 'messages.csv'
2. The categorically labelled details of this messages to be placed in another csv for eg : 'categories.csv'
3. In your terminal move to the project's root folder : *[Project]* 
4. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
    
    .. code-block:: bash
        
            python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

    - To run ML pipeline that trains classifier and saves it persistently: 

    .. code-block:: bash

            python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

5. Run the following command in the app's directory to run your web app.
    
    .. code-block:: bash
            
            python run.py

6. Go to http://0.0.0.0:3001/ in your browser.