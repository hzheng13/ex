This package contains the evaluation codes for spaCy bundled as an
Eclipse.  This project is pure python based.
1. you should download a python with latest version. you may also pip up some extra library if missing.
2. import this project into eclipse.
3. setup python evironment with eclipse to work with the python installed.
4. put sample data into data folder in the project.
5. run the python file directly inside eclipse.


Note: sample data is not included here which can be downloaded from wiki page provided.
      sample data for train model can use sample 1, then each test case there meet 100% expectation.

I also create a simple prototype rest API (api.py) to test the functonality based on default model
as following:
1. pip out flask library
2. python api.py
3. go to url: 'http://localhost:5000/api/v1/resources/entities/all' to see the whole result of ner search
      or url: 'http://localhost:5000/api/v1/resources/entities?label=ORG' to see the partial result of ner search by label(category/type)
      or go to postman or similar tool with POST method and text input as body using url: 'http://localhost:5000/api/v1/resources/entities/all'
         notice: Content-Type should be "text/plain" in the header