from flask import Flask, request, jsonify, render_template, url_for, request
import re
import pandas as pd
from default_ner_example import DefaultNerEx
import spacy
from spacy import displacy
from cgitb import text

dataFile = 'data/data1.txt'
nlp = spacy.load('en_core_web_lg')
nlpFr = spacy.load('fr_core_news_md')

app = Flask(__name__)
app.config["DEBUG"] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/process', methods=["POST"])
def process():
    if request.method == 'POST':
        choice = request.form['taskoption']
        rawtext = request.form['rawtext']
        doc = nlp(rawtext)
        d = []
        for ent in doc.ents:
            d.append((ent.label_, ent.text))
            df = pd.DataFrame(d, columns=('named entity', 'output'))
            ORG_named_entity = df.loc[df['named entity'] == 'ORG']['output']
            PERSON_named_entity = df.loc[df['named entity'] == 'PERSON']['output']
            GPE_named_entity = df.loc[df['named entity'] == 'GPE']['output']
            MONEY_named_entity = df.loc[df['named entity'] == 'MONEY']['output']
            DATE_named_entity = df.loc[df['named entity'] == 'DATE']['output']
            CARDINAL_named_entity = df.loc[df['named entity'] == 'CARDINAL']['output']
        if choice == 'organization':
            results = ORG_named_entity
            num_of_results = len(results)
        elif choice == 'person':
            results = PERSON_named_entity
            num_of_results = len(results)
        elif choice == 'geopolitical':
            results = GPE_named_entity
            num_of_results = len(results)
        elif choice == 'money':
            results = MONEY_named_entity
            num_of_results = len(results) 
        elif choice == 'date':
            results = DATE_named_entity
            num_of_results = len(results) 
        elif choice == 'cardinal':
            results = CARDINAL_named_entity
            num_of_results = len(results)
             
    return render_template("index.html", results=results, num_of_results = num_of_results)  

""" 
    Return jsonified entity list 
    
    given arguments 'text' and 'lang'
"""         
@app.route('/api/ner', methods = ['GET']) 
def get_all_entities_from_msg():
    text = request.args.get('text')
    lang = request.args.get('lang')
    if lang is not None:
        if lang == 'fr':
            doc = nlpFr(text)
        else:
            doc = nlp(text)    
    else:
        doc = nlp(text)        
    d = []
    for ent in doc.ents:
        d.append((ent.text, ent.label_))
            
    return jsonify(d)
    
           
# A route to return all of the available entries from NER search.
@app.route('/api/v1/resources/entities/all', methods=['GET'])
def get_all_entities():
    lst=DefaultNerEx().ner_search(dataFile)
    keys=['entity_name', 'label']
    entity_list = []
    for ent in lst:
           ent1=[]
           ent1.append(str(ent[0]))
           ent1.append(ent[1])
           entity_list.append(dict(zip(keys,ent1)))
           
    return jsonify(entity_list)

# A route to return all of the available entries from NER search.
@app.route('/api/v1/resources/entities', methods=['GET'])
def get_entities_by_label():
    # Check if an label was provided as part of the URL.
    # If label is provided, assign it to a variable.
    # If no label is provided, display an error in the browser.
    if 'label' in request.args:
        label = str(request.args['label'])
    else:
        return "Error: No label field provided. Please specify an label."
    lst=DefaultNerEx().ner_search(dataFile)
    keys=['entity_name', 'label']
    entity_list = []
    for ent in lst:
        if ent[1] == label.upper():
           ent1=[]
           ent1.append(str(ent[0]))
           ent1.append(ent[1])
           entity_list.append(dict(zip(keys,ent1)))
    return jsonify(entity_list)

app.run()
