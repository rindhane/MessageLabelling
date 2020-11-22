import pandas as pd
import numpy as np
import sys
from sqlalchemy import create_engine


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
    return files

#helper function to manage column list 
def column_update(l,entry) :
    if entry in l:
        pass
    else:
        l.append(entry)

columns=[] # list to store the column data

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

                
def create_merged_dataframe(csv_message,csv_categories):
    '''After extracting and cleaning the two csv file data,
    it creates single merged dataframe'''
    messages = pd.read_csv(csv_message)
    categories = pd.DataFrame(read_categories(csv_categories)
                              ,columns=columns)
    #removing duplicates
    messages=messages[~messages.duplicated()]
    categories=categories[~categories.duplicated()]
    df = messages.merge(categories, on = 'id', how='left')
    return df

def save_to_db(df,fileName, tableName):
    engine = create_engine(f'sqlite:///{fileName}')
    conn=engine.connect()
    engine.execute(f"DROP TABLE IF EXISTS {tableName}")
    df.to_sql(tableName, conn, index=False)
    return True

    
if __name__ =="__main__":
    _,messages,categories,db=get_file_names()
    df=create_merged_dataframe(messages,categories)
    tableName='categorizedMessages'
    a=save_to_db(df,db,tableName=tableName)
    if a : 
        print(f"Data has been saved into the {db} within \
the table '{tableName} '")

    
    
    
    