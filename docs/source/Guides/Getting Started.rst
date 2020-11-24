Getting Started
==================

    .. note:: Assumed to have basic knowledge on using a terminal

Steps to install the demo project:
***************************************
   
    **A. Making sure readiness with python:** 
        
        * Follow the guidelilnes ,to install python, form the provided link if begineer in python >> `python support <https://realpython.com/installing-python/>`_
        
        * Create a dir called ``Project`` . Its folder path will be referred as *[Project]* 

    **B. Clone this repository into your [Project] folder:** 
        
        * If unsure about Git, follow this refrence link to get help about git >> `Git support <https://docs.gitlab.com/ee/gitlab-basics/start-using-git.html>`_
    
        .. code-block:: bash

            cd /path/to/[Project]
        
            git clone https://github.com/rindhane/MessageLabelling.git labeller

    **C. Create Virtual Environment [pyenv] :** 
        
        * If unsure follow this guidelines >> `venv help <https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/>`_
    
        .. code-block:: bash 

            python -m venv pyenv

            source pyenv/bin/activate  
            
        `Click here for Detailed guidelines. <https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#activating-a-virtual-environment>`_

    **D. Install dependencies:** 
    
    .. code-block:: bash

        pip install -r requirements.txt

    **E. Run webserver:**

        * Within terminal move to *[Project]/labeller/app* & run following command: 

        .. code-block:: bash
            
            python run.py

    **F. Checking the web-interface:**
        
        * Now in browser type following url into address bar to see the demo : *http://127.0.0.1:8080/*

