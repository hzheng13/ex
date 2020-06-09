'''
Created on June. 09, 2020

@author: hong
'''
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pandas as pd


class FuzzyMatchingEx:

   def fuzzy_search(self,file):
      expected_org_list = ['High Top Brewing','Holiday Inn Hotel Washington']
      
      df = pd.read_csv(file)
      
      print(df.head(-1))
      print('')  
      org_data = [row[0] for row in df.values]
      
      process.dedupe(org_data, threshold=80)
      
      print("The expected entities: " + str(expected_org_list) + "\n")
      
      print('The companies are matched as follows.')      
      for query in expected_org_list:
        #scorer could be ratio, partial_ratio, token_sort_ratio or token_set_ratio
        #result = process.extract(query, org_data, scorer=fuzz.partial_ratio, limit=2)
        result = process.extractBests(query, org_data, scorer=fuzz.partial_ratio, score_cutoff=70, limit=2)
        
        print(result)

if __name__ == '__main__':
   #execute the action
   FuzzyMatchingEx().fuzzy_search('data/org_name.csv')