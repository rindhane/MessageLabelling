ETL Pipeline:
====================

ETL pipepline is the component of the application which deals to extract raw data from CSV files, then process it and then put the data consolidatedly into the database within a single table . In concise, ETL pipeline takes in data from the two separate csv files called ``categories.csv`` and ``message.csv`` and processing on the fly , pushes it into the database called ``DisasterResposne.db`` 

A. The ETL pipeline subcomponents into following function :
************************************************************
    - **get_file_names:** to capture the filenames from the arguments passed on to the command line when running the ETL pipeline.
    - **read_categories:** it is to process the data captured from the ``categories.csv`` file and then clean it and make it in proper structure to get it readied for consumed into a pandas dataframe. 
    - **clean_data & remove_duplicates:** this components improves the quality of data by removing unessential and polluted data from the dataframe.
    - **create_merged_dataframe:**  this creates consolidated dataframe after merging data in one-to-one fashion from ``messages.csv`` and ``categories.csv``.
    - **save_to_db:** this pushes and save data into the database file. Here that file name is ``DisasterResposne.db`` 

B. The overview of provided data:
**************************************
    1. **Messages.csv:** Here the data is in pretty much in unicode text format and properly strucutred into the 4 columns.

    *Outlook:*  
        .. image:: ../_static/message_data.png

    2. **Categories.csv:** This provides labelled data under each category for every individual message. There are in total 36 categories where each category is related to specific type of distress or support catered by specific department. Eg: Medical help, fire-safety help & etc. The structured data presented below is achived only after the data in cleaned by the intermediary functions. (Check out the jupyter notebook ETL pipeline preparation.ipynb to see working details of cleaning functions )
    
    *Outlook:* 
         .. image:: ../_static/categories.png
    
    3. **Merged dataframe:** This is the end result as a single dataframe after merging the cleaned data from both the csv. The 'id' column present in the both csv helps to merge the related messages with their categories-label . 

    *Outlook:* 
        .. image:: ../_static/merge_data.png