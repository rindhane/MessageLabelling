## Complete Structure of Project's Directory

```
Project [Root Folder]/
├── Analysis & Data Exploration/
│   ├── ETL Pipeline Preparation.ipynb
│   └── ML Pipeline Preparation.ipynb
├── app/
│   ├── run.py
│   └── templates/
│       ├── go.html
│       └── master.html
├── data/
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   ├── DisasterResponse.db
│   └── process_data.py
├── models/
│   ├── classifier.pkl
│   ├── model_details.txt
│   └── train_classifier.py
├── docs/
├── LICENSE
├── README.md
├── requirements.txt
├── .gitignore
└── .readthedocs.yml

```

**Details:**

+ *Analysis Analysis & Data Exploration:* Contains the jupyter notebooks which will give fair idea about the analysis of data & development steps of application.

+ *app:* It is the folder with all the files related to the web app made in flask.

+ *data:* Primary data files based on which application was trained upon and file `process_data.py` which contains the etl_pipeline.

+ *models:* Holds the ML pipepline file `train_classifier.py` which creates the model for text labelling.

+ *docs:* This files contains the files for holding the documentation.

+ *other files:* This are configuration files for application. 
