'''
Created on Mar. 26, 2020

@author: hong
'''
import spacy
import random
import json
import ast
from pprint import pprint
from spacy.util import minibatch, compounding
from pathlib import Path
import shutil
from spacy.util import decaying
from collections import defaultdict
from spacy.gold import GoldParse
from cytoolz import partition_all
import re


class TrainNerEx:
   #Pseudo-rehearsal solution for catastrophic forgetting problem: Use different and unrelated texts for the revision texts, not the same data as the train data and raw texts but should keep any contents containing entities with general entity types such as ORG, MONEY, DATE and so on.
   REVISION_TEXTS=[
'On 10 Dec 2018, John Li informed me by text messages that he was preparing an offer (property address: 1110 Southbend Road, West Vancouver C2V 9K7). Jong Li (Preferred name: John) is 28 years old, (DOB:24Jan1992), Male. He is    assigned, Canadian/Foreign Passport issued in China',
'He asked if it was legal when both the buyer and seller agreed to accept three million Canadian dollars as deposit and the deposit would be released to the seller directly on 11 May 2018 as agreed by both parties.', 
"I replied that both parties should find their own legal representatives and deposit could be submitted to the buyer's lawyer or our Brokerage.", 
'The release of the deposit would be handled by the lawyers.', 
'Our Brokerage received the deposit ($110,000) on 10 Nov 2016 and receipt was issued.',
"On the page 1 of 10   in the contract, it   states 'The Buyers agree to increase the deposit to TWOMILLION Dollars \non or before January 13th, 2013 and authorize the selling brokerage 'Amex Flamenco Realty' to release the TWO MILLION DOLLARS deposit directly to the Seller, no later than January 11th, 2013.",
'According to my realtor, the buyers realized the risk and the deposit would be held by the seller if the deal was not going to complete.',
'Therefore, I required our Realtor to add an addendum to mention Buyer and seller both agree that, if the buyer fails to complete the deal on Feb 25, 2013, the total deposit (two million Canadian dollars)', 
'That is, the total deposit will not be returned to the buyer and the seller will keep the deposit (two million Canadian dollars) as compensation after Feb 24, 2013.', 
'Both parties agree not to file any lawsuit upon the failure of deal completion.', 
'However, the buyers refused to sign the addendum and refused to appoint any legal representatives to handle the release of the deposit.',
'After discussing with my realtor, I  found the following points of suspicion: Buyers insisted not to find a lawyer to represent them and handle the large amount of deposit.  ', 
'They simply wanted to wire the deposit and the remaining amount (Four Million Dollars) directly to our trust account from overseas.',
'I refused and confirmed that our Brokerage can only accept the deposit by bank draft issued by local Canadian banks.',
"Since they refused to appoint the lawyers to release the deposit, I didn't approve the clause  written on page 2 of 6 in the contract.",  
'Attached you can find the contract and the emails concerning this deal.'
]
    
   def __init__(self):
       self.model_file = 'custommodel'
       self.train_data_file = 'data/train-data.txt'
       self.dropout = decaying(0.1, 0.0, 1e-4)
       self.iters=20
       self.batch_size=2
       
       #address
       self.address_label = "GPE"
       address_pattern='\d+[\w\s]+(?:avenue|ave|road|rd|boulevard|blvd|street|st|drive|dr|court|ct|highway|hwy|square|sq|park|parkway|pkwy|circle|cir|trail|trl)[,*\w\s]+([a-z][0-9][a-z]\s*[0-9][a-z][0-9](,*\s*canada)?)'
       self.address_pattern_object   = re.compile(address_pattern, re.IGNORECASE)
       #date
       self.date_label = "DATE"
       date_pattern='\d+(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september|oct|october|nov|november|dec|december)\d+'
       self.date_pattern_object             = re.compile(date_pattern, re.IGNORECASE)
       
   def pattern_matcher(self,doc): 
      label_pattern_objects={self.address_label:self.address_pattern_object, self.date_label:self.date_pattern_object}

      spans = []
      for label in label_pattern_objects:
         pattern_object = label_pattern_objects[label]
         for match in re.finditer(pattern_object, doc.text):
            start, end = match.span()
            span = doc.char_span(start, end, label=label)
            if span:  # Only add span if it's valid
               spans.append(span)
      # Add spans to the doc.ents
      doc.ents = list(doc.ents) + spans
      return doc
       
   def train_model(self, data, iters, model,is_rehearsal):

          TRAIN_DATA = data
          if model is not None:
             #load existing model
             nlp=spacy.load(model)
             print("Loaded model '%s'" % model)
          else:
             # create blank Language class
             nlp = spacy.blank('en')
             print("Created blank 'en' model")
    
          # create the built-in pipeline components and add them to the pipeline
          # nlp.create_pipe works for built-ins that are registered with spaCy
          if 'ner' not in nlp.pipe_names:
              ner = nlp.create_pipe('ner')
              nlp.add_pipe(ner, last=True)
          else:
              ner = nlp.get_pipe('ner')
       
          # add new labels
          for _, annotations in TRAIN_DATA:
               for ent in annotations.get('entities'):
                  ner.add_label(ent[2])

          if is_rehearsal is not None and is_rehearsal == 'y':
             # preparing the revision data
             revision_data = []
             for doc in nlp.pipe(self.REVISION_TEXTS):#list(zip(*TRAIN_DATA))[0]
                 tags = [w.tag_ for w in doc]
                 heads = [w.head.i for w in doc]
                 deps = [w.dep_ for w in doc]
                 entities = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
                 revision_data.append((doc, GoldParse(doc, tags=tags, heads=heads,
                                         deps=deps, entities=entities)))
             # preparing the fine_tune_data
             fine_tune_data = []
             for raw_text, entity_offsets in TRAIN_DATA:
                 doc = nlp.make_doc(raw_text)
                 gold = GoldParse(doc, entities=entity_offsets['entities'])
                 fine_tune_data.append((doc, gold))
             
          # get names of other pipes to disable them during training
          other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
          with nlp.disable_pipes(*other_pipes):  # only train NER
              if is_rehearsal is not None and is_rehearsal == 'y':
                 #batch_size = 2    
                 for iter in range(iters):
                    print("Starting iteration " + str(iter))                  
                    examples = revision_data + fine_tune_data
                    losses = {}
                    random.shuffle(examples)
                    for batch in partition_all(self.batch_size, examples):
                       docs, golds = zip(*batch)
                       nlp.update(docs, golds, drop=0.0, losses=losses)
                    print("Losses", losses)
              else:
                if model is None:
                   # initialize the weights randomly only if training a new model
                   optimizer = nlp.begin_training()
                else:
                   optimizer = nlp.resume_training() #nlp.entity.create_optimizer()
                  
                for iter in range(iters):
                    print("Starting iteration " + str(iter))
                    random.shuffle(TRAIN_DATA)
                    losses = {}
                    batches = minibatch(TRAIN_DATA, size=compounding(1., 4., 1.001))
                    for batch in batches:
                       texts, annotations = zip(*batch)
                      
                       # Updating the weights
                       nlp.update(texts,
                                 annotations,
                                 sgd=optimizer,  # callable to update weights
                                 drop=0.0,  # dropout - make it harder to memorise data
                                 losses=losses)
                    print("Losses", losses)
          return nlp
      
   def ner_search(self, file, model=None, is_train=None, is_rehearsal=None):
       
      # Read whole raw text
      with open(file,'r') as f:
         message = f.read()
      
      with open(self.train_data_file,"r") as tdf:
         TRAIN_DATA = tdf.read()

      TRAIN_DATA = ast.literal_eval(' '.join(TRAIN_DATA.split()))
      #TRAIN_DATA = self.TRAIN_DATA

      if is_train is not None and is_train == 'y':
         nlp =  self.train_model(TRAIN_DATA, self.iters, model, is_rehearsal)
         
         model_file_dir = Path(self.model_file)
         if model_file_dir.exists() and model_file_dir.is_dir() and self.model_file is not model:
            shutil.rmtree(model_file_dir)
         # Save our trained Model
         nlp.to_disk(self.model_file)
         #Test each sentence from train data
         for text, _ in TRAIN_DATA:
             nlp(text)
             
      if self.model_file is not None:
        model_file_dir = Path(self.model_file)
        if not model_file_dir.exists():
            print("Created a custom model first!")
            return []
      nlp = spacy.load(self.model_file)
      
      #beam search
      def _beam_search():
         with nlp.disable_pipes('ner'):
              doc = nlp(message)

         threshold = 0.0 #0.2
         (beams) = nlp.entity.beam_parse([ doc ], beam_width = 16, beam_density = 0.0001)

         entity_scores = defaultdict(float)
         for beam in beams:
             for score, ents in nlp.entity.moves.get_beam_parses(beam):
                for start, end, label in ents:
                   entity_scores[(start, end, label)] += score

         entities=[]
         for key in entity_scores:
             start, end, label = key
             score = entity_scores[key]
             if ( score > threshold):
                entities.append((doc[start:end], label, score))
         return entities
     
      #greedy search
      def _greedy_search():
         doc = nlp(message)

         entities=[]
         for entity in doc.ents:
                entities.append((entity.text, entity.label_, 1.0))
         return entities
          
      #custom search
      def _custom_search():
         #add the customized pipeline which cannot work together with default models somehow
         nlp.add_pipe(TrainNerEx().pattern_matcher, name="custom_pattern_matcher", last=True)
         
         #insert one space in order to identify the special date following 'DOB:' if there
         message1=re.sub('DOB:','DOB: ',message)
         
         with nlp.disable_pipes('ner','tagger', 'parser'):
             doc = nlp(message1)
         entities=[]      
         [entities.append((ent.text, ent.label_, 1.0)) for ent in doc.ents]
         return entities

      return _beam_search() + _custom_search()
       
 
      
if __name__ == '__main__':
   #entities=TrainNerEx().ner_search('data/data3.txt') # run against the existing model: custommodel
   #entities=TrainNerEx().ner_search('data/data1.txt',is_train='y') #run against the custommodel created just from blank
   #entities=TrainNerEx().ner_search('data/data1.txt','custommodel','y') # run against the existing custommodel just updated
   #entities=TrainNerEx().ner_search('data/data1.txt','en_core_web_lg','y') # run against the custommodel created just from en_core_web_lg with new update
 
   #entities=TrainNerEx().ner_search('data/data1.txt','custommodel','y',is_rehearsal='y') # run against the existing custommodel just updated using Pseudo-rehearsal solution
   entities=TrainNerEx().ner_search('data/data1.txt','en_core_web_lg','y',is_rehearsal='y') # run against the custommodel created just from en_core_web_lg using Pseudo-rehearsal solution
   
   # Find named entities from the message
   pprint([(entity) for entity in entities])
   
   #for entity in doc.ents:
   #    print(entity.text, entity.label_)
