'''
Created on Mar. 24, 2020

@author: hong
'''
import spacy
from pprint import pprint
from collections import defaultdict
import re

class DefaultNerEx:
   def __init__(self):
      #address
      self.address_label = "GPE"
      address_pattern='\d+[\w\s]+(?:avenue|ave|road|rd|boulevard|blvd|street|st|drive|dr|court|ct|highway|hwy|square|sq|park|parkway|pkwy|circle|cir|trail|trl)[,*\w\s]+([a-z][0-9][a-z]\s*[0-9][a-z][0-9](,*\s*canada)*)'
      self.address_pattern_object   = re.compile(address_pattern, re.IGNORECASE)
      #date
      self.date_label = "DATE"
      date_pattern='(\d+(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september|oct|october|nov|november|dec|december)\d+)'
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
  
   def ner_search_from_message(self, message, lang): 
      if lang is not None and lang == 'fr':
         # Load French
         nlp = spacy.load('fr_core_news_md')

      else:
         # Load English
         nlp = spacy.load('en_core_web_lg')

      #beam search
      def _beam_search():
         with nlp.disable_pipes('ner'):
              doc = nlp(message)

         threshold = 0.2
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

      #custom search
      def _custom_search():
         #add the customized pipeline which cannot work together with default models somehow
         nlp.add_pipe(DefaultNerEx().pattern_matcher, name="custom_pattern_matcher", last=True)
         with nlp.disable_pipes('ner','tagger', 'parser'):
             doc = nlp(message)
         entities=[]      
         [entities.append((ent.text, ent.label_, 1.0)) for ent in doc.ents]
         return entities
     
      if lang is not None and lang == 'fr':
         # French. to do customize.
         return _beam_search()
      else:
         # English
         return _beam_search() + _custom_search()


   def ner_search(self, file, lang):
      # Read whole documents
      with open(file,'r') as f:
         message = f.read()
      return self.ner_search_from_message(message, lang)

if __name__ == '__main__':
   #execute the action
   entities=DefaultNerEx().ner_search('data/data2.txt', 'en')
   
   print ('Entities and scores (detected with beam search)')
   # Find named entities with text, entity type, probability
   pprint([(entity) for entity in entities])
