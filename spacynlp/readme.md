This package contains the evaluation codes for spaCy bundled as an
Eclipse.  This project is pure python based.
1. you should download a python with latest version. you may also pip up some extra library if missing.
2. import this project into eclipse.
3. setup python evironment with eclipse to work with the python installed.
4. put sample data into data folder in the project.
5. run the python file directly inside eclipse.

Note: sample data is not included here which can be downloaded from wiki page provided.
      sample data for train model can use sample 1, then each test case there meet expectation.

There are the following ways to run train ner script
1. if update an existing default model and then do a NER search
      run TrainNerEx().ner_search('data/data1.txt','en_core_web_lg','y')
2. if update an existing custom model and then do a NER search
      run TrainNerEx().ner_search('data/data1.txt','custommodel','y')
3. if only do a NER search against an existing custom model without creating or updating any model
      run TrainNerEx().ner_search('data/data1.txt')
4. if create a new custom model from blank and then do a NER search
      run TrainNerEx().ner_search('data/data1.txt',is_train='y')
5. if update an existing default model using Pseudo-rehearsal solution
      run TrainNerEx().ner_search('data/data1.txt','en_core_web_lg','y',is_rehearsal='y')
6. if update an existing custom model using Pseudo-rehearsal solution
      run TrainNerEx().ner_search('data/data1.txt','custommodel','y',is_rehearsal='y')


A prototype rest API (api.py) is also created to test the functonality based on default model
as following:
1. pip out flask library
2. python api.py
3. go to postman or similar tool with POST method and text input as body using urls:
    'http://localhost:5000/api/v1/resources/entities/all' if text body is english
    'http://localhost:5000/api/v1/resources/entities/all?lang=fr' if text body is french 
    'http://localhost:5000/api/v1/resources/entities/all?label=DATE' if need to do a category search

Note: Content-Type should be "text/plain" in the header. By default url, it is english content. Otherwise, attach query parameter lang=fr to url for french content.
      
You can also enter text input to view NER results
To view visually, select 'default' option
4. go to url: http://localhost:5000/ 

To send Rest requests
5. send request to localhost:5000/api/ner?text="YOUR TEXT" 


