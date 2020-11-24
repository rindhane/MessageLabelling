Machine Learning Pipeline:
=============================

This is the second part of the application in which the gathered data, present in the database file, is used to train a machine learning model. This trained model helps to map a provided text message into label(1 or 0) under each of the provided response categories. For eg: if a message is needs response from medical and fire team, it will label it 1 in both this category but mark it '0' in other category for eg: 'floods support'.     


A. The ML pipeline's subcomponent-functions are :
*************************************************** 
    - **load_data:** it loads data from the database and make it ready in array format to have it readily available for ML-model to consume and train on it.
    - **tokenize:** it is a helper function to convert a given text sentence into a python list of words in their root form. 
    - **build_model:** it creates the suitable machine learning model from the sklearn library wrapped in pipeline structure. This pipeline is also coupled with GridSearchCV inorder to select relevant parameters to tune the model with the provided data.
    - **evaluate_model:**  It provides performance report on the model against each category and its label.
    - **save_model:** It saves the model into the persistent memory for later use by the web application. 
            