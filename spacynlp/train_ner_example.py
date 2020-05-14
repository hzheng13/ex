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


class TrainNerEx:
   #TRAIN_DATA=[] #move the train data to a file already
   
   def __init__(self):
       self.model_file = 'custommodel'
       self.train_data_file = 'data/train-data.txt'
       self.dropout = decaying(0.1, 0.0, 1e-4)
       self.iters=20
       
   def train_model(self, data, iters, model):

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

          # get names of other pipes to disable them during training
          other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
          with nlp.disable_pipes(*other_pipes):  # only train NER
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
      
   def ner_search(self, file, model=None, is_train=None):
      # Read whole raw text
      with open(file,'r') as f:
         message = f.read()
      
      with open(self.train_data_file,"r") as tdf:
         TRAIN_DATA = tdf.read()

      TRAIN_DATA = ast.literal_eval(' '.join(TRAIN_DATA.split()))
      #TRAIN_DATA = self.TRAIN_DATA

      if is_train is not None:
         nlp =  self.train_model(TRAIN_DATA, self.iters, model)
         
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
      doc = nlp(message)

      entities=[]
      for ent in doc.ents:
          entities.append((ent.text, ent.label_))
      return entities
       

      
if __name__ == '__main__':
   entities=TrainNerEx().ner_search('data/data1.txt') # run against the existing model: custommodel
   #entities=TrainNerEx().ner_search('data/data1.txt',is_train='y') #run against the custommodel created just from blank
   #entities=TrainNerEx().ner_search('data/data1.txt','custommodel','y') # run against the existing custommodel just updated
   #entities=TrainNerEx().ner_search('data/data1.txt','en_core_web_lg','y') # run against the custommodel created just from en_core_web_lg with new update
   
   # Find named entities from the message
   pprint([(entity) for entity in entities])
   
   #for entity in doc.ents:
   #    print(entity.text, entity.label_)
