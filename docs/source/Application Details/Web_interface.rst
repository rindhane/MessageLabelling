Web Interface :
==================
To make the complete model accesible for a regular user, the model is made accessible through the web interface. The web interface is created using flask as web-framework and plotly for creating web charts.

The web interface is created as flask application. To run it , simply go to app folder location in the terminal and run the code::

    python run.py

It will run the development webserver to provide visualize the application , by using the browser. 

A. The web-interface sub-components are :
###################################################

    .. note:: Need to have an undersanding on flask & plotly for clear understanding of the sub-components

- **index:** It is entry point to web interface. It is the main function which renders the intial web page for the user. Through this user is able to see the metrics about the training data and a search bar through which it inputs the query to see its relation to response categories. 

- **go:** It is the search bar component which displays the results for the query of the provided messages and generates the webpage again to receive the query again.  
