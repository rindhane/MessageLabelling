import pandas as pd
import numpy as np
import sys
from sqlalchemy import create_engine


'''Global Variables defined: list to store the column data'''
columns=[]

def get_file_names():
    '''Capture file name arguments passed to command line 
    and transfer to relevant functions as their required arguments'''
    files=list()
    for arg in sys.argv:
        files.append(arg)
    if len(files)!=4:
        raise ValueError("Provide three filenames to the pipeline\
        in sequence messages.csv,\
        categories.csv , databasse.db")
    if files[1].split('.')[-1]!='csv' :
        raise ValueError (f"Provide .CSV filename at {files[1]} ")
    if files[2].split('.')[-1]!='csv' :
        raise ValueError (f"Provide .CSV filename at {files[2]} ")
    if files[3].split('.')[-1]!='db' :
        raise ValueError (f"Provide .db extension at {files[3]} ")
    return files[1:]


def column_update(l,entry) :
    '''helper function to manage column list '''
    if entry in l:
        pass
    else:
        l.append(entry)
        


def read_categories(path_csv):
    '''This function is to read the categories'
    matrix-data from the categories.csv '''
    #reading the given file 
    fp= open(path_csv,'r')
    while(True):
        a=fp.readline()
        if a =='':
            fp.close()
            break
            yield None
        else:
            a=a.replace('\n','')
            items=a.split(',')
            result=[]
            for item in items:
                splits=item.split(';')
                if item=='categories' :
                    continue
                if item=='id':
                    column_update(columns,item)
                    continue
                elif len(splits) >2 : 
                    for vals in splits:
                        tmp=vals.split('-')
                        column_update(columns,tmp[0])
                        result.append(int(tmp[-1]))
                else:
                    result.append(int(splits[0]))
            if len(result)>=1:
                yield result
            else:
                continue

def load_data(messages_filepath, categories_filepath):
    '''extracting and two csv file data'''
    messages = pd.read_csv(messages_filepath)
    categories = pd.DataFrame(read_categories(categories_filepath)
                              ,columns=columns)
    return messages, categories


def clean_data(df):
    df=df[~df.duplicated()]
    return df

def merge_dataframe(messages,categories):
    df = messages.merge(categories, on = 'id', how='left')
    return df

def save_data(df, database_filename,tableName):
    engine = create_engine(f'sqlite:///{database_filename}')
    conn=engine.connect()
    engine.execute(f"DROP TABLE IF EXISTS {tableName}")
    df.to_sql(tableName, conn, index=False)
    return True  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = get_file_names()

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        message,categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        message = clean_data(message)
        categories = clean_data(categories)
        df = merge_dataframe(message,categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        tableName='categorizedMessages'
        save_data(df, database_filepath,tableName)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()