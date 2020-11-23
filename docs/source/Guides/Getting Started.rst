Getting Started
==================


Steps to install the demo project
    {Assumed to have basic knowledge to use a terminal}
    A. Making sure readiness with python 
    * Follow this guidelilnes if begineer in python [https://realpython.com/installing-python/]
    * Create a dir called "Project" . Its folder path will be referred as [Project] 
    B.Clone this repository into your [Project] folder. If unsure about Git, follow this refrence about git[https://docs.gitlab.com/ee/gitlab-basics/start-using-git.html]
    * `cd /path/to/[Project]`
    *  `git clone 'https://github.com/rindhane/MessageLabelling.git' labeller`
    C. Create Virtual Environment :'pyenv'. If unsure follow this guidelines [https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/]
    * python -m venv pyenv
    * source [Project]pyenv/bin/activate . (Detailed guidelines follow here [https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#activating-a-virtual-environment])
    D. Install dependencies : (be at [Project] folder in terminal)
    * pip install -r requirements.txt  
    E. Within terminal move to [project]/labeller/app & run following command: 
    * python run.py
    F. Now in browser type following url into address bar to see the demo : 
    * http://127.0.0.1:3001/

