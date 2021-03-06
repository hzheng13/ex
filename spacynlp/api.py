from flask import Flask, request, jsonify, render_template, url_for, request
import re
import pandas as pd
from default_ner_example import DefaultNerEx
import spacy
from spacy import displacy
from cgitb import text
from flaskext.markdown import Markdown

nlp = spacy.load('en_core_web_lg')
nlpFr = spacy.load('fr_core_news_md')

app = Flask(__name__)
Markdown(app)
app.config["DEBUG"] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0    

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/process_ner', methods=["POST"])
def process_ner():
    if request.method == 'POST':
        choice = request.form['taskoption']
        lang = request.form['langoption']
        rawtext = request.form['rawtext']
        if lang is not None:
            if lang == 'fr':
                doc = nlpFr(rawtext)
            else:
                doc = nlp(rawtext)    
        else:
            doc = nlp(rawtext)
        #displacy.serve(doc, style="ent")
        #displacy.serve(doc, style="dep")
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
        elif choice == 'default':    
            html = displacy.render(doc, style='ent')
            results = html
            return render_template('results.html', rawtext=rawtext, results=results)
             
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
    


# A route to return all or part of the entries by input text as body from NER search
# with two optional query parameters: lang and label.
@app.route('/api/v1/resources/entities/all', methods=['POST'])
def handle_message():
    if request.headers['Content-Type'] == 'text/plain':
        lang = request.args.get('lang')
        label = request.args.get('label')
        message = request.get_data().decode("utf-8")
        lst = DefaultNerEx().ner_search_from_message(message, lang)
        keys = ['entity_name', 'label']
        entity_list = []
        for ent in lst:
           if label is None or ent[1] == label.upper():
              ent1 = []
              ent1.append(str(ent[0]))
              ent1.append(ent[1])
              entity_list.append(dict(zip(keys,ent1)))

        return jsonify(entity_list)
    else:
        return "415 Unsupported Media Type."
app.run()
