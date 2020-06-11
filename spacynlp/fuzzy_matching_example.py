'''
Created on June. 09, 2020

@author: hong
'''
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from rapidfuzz import fuzz as rapid_fuzz
from rapidfuzz import process as rapid_process
import pandas as pd


class FuzzyMatchingEx:
   #experiment using organization data with fuzzywuzzy library
   def fuzzy_search(self,file):
      expected_org_list = ['High Top Brewing','Holiday Inn Hotel Washington']
      
      df = pd.read_csv(file)
      
      print(df.head(-1))
      print('')  
      org_data = [row[0] for row in df.values]
      
      process.dedupe(org_data, threshold=80)
      
      print("The expected companies  : " + str(expected_org_list) + "\n")
      
      print('The companies are matched as follows.')      
      for query in expected_org_list:
        #scorer could be ratio, partial_ratio, token_sort_ratio or token_set_ratio
        #result = process.extract(query, org_data, scorer=fuzz.partial_ratio, limit=2)
        result = process.extractBests(query, org_data, scorer=fuzz.partial_ratio, score_cutoff=70, limit=2)
        
        print(result)
        
   #experiment using movie data with fuzzywuzzy library         
   def fuzzy_movie_search(self,file):
  
      df = pd.read_csv(file)
      
      print(df.head(-1))
      print('')  
      misspelled_film_list = [row[0] for row in df.values]
      movie_data = [row[1] for row in df.values]
      
      process.dedupe(movie_data, threshold=80)
      
      print("The misspelled movies: " + str(misspelled_film_list) + "\n")
      
      print('The movies are matched as follows.')      
      for query in misspelled_film_list:
        #scorer could be ratio, partial_ratio, token_sort_ratio or token_set_ratio
        result = process.extractBests(query, movie_data, scorer=fuzz.partial_ratio, score_cutoff=70, limit=2)
        
        print("\'" + query + "\' matches with the films with scores: " + str(result))
        
   #experiment  using movie data with the rapidfuzzy library     
   def rapid_fuzzy_movie_search(self,file):
  
      df = pd.read_csv(file)
      
      print(df.head(-1))
      print('')  
      misspelled_movie_list = [row[0] for row in df.values]
      movie_data = [row[1] for row in df.values]
      
      print("The misspelled movies: " + str(misspelled_movie_list) + "\n")
      
      print('The movies are matched as follows.')      
      for query in misspelled_movie_list:
        #scorer could be ratio, partial_ratio, token_sort_ratio or token_set_ratio
        result = rapid_process.extractBests(query, movie_data, scorer=rapid_fuzz.partial_ratio, score_cutoff=70, limit=2)
        
        print("\'" + query + "\' matches with the movies with scores: " + str(result))
        
if __name__ == '__main__': 
   #execute the action
   #FuzzyMatchingEx().fuzzy_search('data/org_name.csv')
   #FuzzyMatchingEx().fuzzy_movie_search('data/movies.csv')
   FuzzyMatchingEx().rapid_fuzzy_movie_search('data/movies.csv')