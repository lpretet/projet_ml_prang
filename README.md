Hello, welcome to the ATIAM Esling/Prang Machine Learning project, season 2017.
Authors are : Thomas Guennoc, Colin Lardier, Laure Prétet, Pierre Rampon.

### Here's how the project work :  
1. Make sure you've installed all the dependencies required to run the code (listed in `code/requirements`). NB : Python 3.5 was used for development.
2. By default, we assume you are runnign the code from a terminal and that you are located in the root folder of the project.  
3. You can run the code directly from the console using `$ python code/lib/pythonfile.py`
4. The test suite is presented as a bunch of files in the directory code/test. To run it, try `$ py.test code`. To run one single test, for example test_setup, type `$ py.test code/test/test_setup.py`  
NB. Unfortunaltely, We did not have time to implement the tests ...

### How to check our work :  
1. Build the dataset : `$ python toy/scripts/main.py`. It will build the midi files in the `toy/dataset/progressions` folder.
2. Visualize the results : `$ python code/lib/main.py`.
3. (Optional). The scripts that actually build the models and train them are `ex2_keras.py` and `ex3_word_embedding.py`.  
If you have time, you can try re-build these models by launching them. But they are already stored in `code/models`.

### Other useful informations

1. We used the autopep8 automatic formatting tool. 
2. You can have fun with our models by using the function predict_chords or visualize_embedding.
