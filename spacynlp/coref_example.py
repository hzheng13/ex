'''
This class is currently working only in spaCy-2.0.12 environment with model en-coref-lg-3.0.0.

Created on April 23, 2020.

@author: hong
'''

import spacy

class CorefEx:
   def coref_search(self,file):
      nlp = spacy.load('en_coref_lg')
      # Read whole documents
      f=open(file, "r")
      message =f.read()
      #doc = nlp('My sister has a dog. She loves him.')
      doc = nlp(message)

      if (doc._.has_coref):
          print ('\nHas coreference in the given text')

          print ('\nAll the clusters of corefering mentions in the given text')
          print(doc._.coref_clusters)

          print ('\nAll the "mentions" in the given text')
          for cluster in doc._.coref_clusters:
             print(cluster.mentions)

          print ('\nAll the spans in which each has at least one corefering mention in the given text')    
          for ent in doc.ents:
             if ent._.is_coref:
               print(ent._.coref_cluster)
    
          print ('\nPronouns and their references')
          for token in doc:
             if token.pos_ == 'PRON' and token._.in_coref:
               for cluster in token._.coref_clusters:
                  print (token.text + "==>" + cluster.main.text)
      else:
          print ('\nNo coreference in the given text')        
if __name__ == '__main__':
   #execute the action
   CorefEx().coref_search('data/data3.txt')
