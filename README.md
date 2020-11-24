# Message Labelling for Disaster Support Response Project
> **Introduction** : This project harness the power of the machine learning techniques to siphon the enormous messages into labelled categories inorder to transfer them to respective recipient for further responses & actions.  

### Live Demo
- The working demonstration of entire project can be checkout [here](google.com).

### Working details of the application: 
- The frontend application is rendered through the python based `flask` libaries. This library packages the content into a web application which can be served through nice web interface.
- In backend the provided text is processed through the machine learning pipeline which is prepared through `sklearn` library.
- The ML model is trained on the `Disaster Response messages` provided by 'Figure 8'. 

### Quick Starter guide to checkout the application locally:
- Clone the directory
```bash
git clone https://github.com/rindhane/MessageLabelling.git demo
```
- Using terminal go to to app folder
 ```bash
  cd demo/app
 ```
- Run the following command to start the local webserver to serve the application locally
```bash
python run.py
```
- Following application can be checked out in browser at http://127.0.0.1:3001/ 
- You may need to setup the python virtual environment and install the required dependencies. _For specific detail go [here]()_        

### Reference sources for further details : 
- Refer following documentation [**[Docs](https://messagelabelling.readthedocs.io)**] for further guidelines.
-Exploratory analysis and details of the data through which application was trainded to label the messages can be checked out here []
- Directory structure of entire code files is available **[here](#pendinghere)** 





