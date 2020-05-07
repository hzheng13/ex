import unittest
from default_ner_example import DefaultNerEx as DefaultNerExClass

#common features for other three classes
class CommonUnitTest(unittest.TestCase):
    @staticmethod
    def get_text(entity):
        return entity[0]
    @staticmethod    
    def get_type(entity, entity_type):
        return entity[1] == entity_type

#load first short message, catch entities
class DefaultNerData1Test(CommonUnitTest):
    expected_org_in_data1=['FinTRAC','Capital One Bank USA','Capital One', 'Capital One Bank','Walmart','Target','Quality Inn','Best Buy' ]
    expected_date_in_data1=['1/12/2017','3/15/2016','5/14/2016','5/12/2016','5/14/2016']
    expected_gpe_in_data1=['NA','Canada']
    expected_money_in_data1=['$8,098.76','$8,098.76','$30.03']
    
    @classmethod
    def setUpClass(cls):
        #fetch entities
        cls.entities = DefaultNerExClass().ner_search('data/data1.txt','en')

    @classmethod
    def tearDownClass(cls):
        #clear
        cls.entities=[]
        

    def test_ner_search_by_org(self):

        actual_org_list=[ str(super(DefaultNerData1Test, self).get_text(entity)) for entity in self.entities if super(DefaultNerData1Test, self).get_type(entity,'ORG') or  super(DefaultNerData1Test, self).get_type(entity,'GPE') ]
        
        lst=[value for value in self.expected_org_in_data1 if value in actual_org_list]
        actual_percentage=len(lst)/len(self.expected_org_in_data1)
        self.assertEqual(actual_percentage,50/100)

        
    def test_ner_search_by_gpe(self):
        actual_gpe_list=[str(super(DefaultNerData1Test, self).get_text(entity)) for entity in self.entities if super(DefaultNerData1Test, self).get_type(entity,'GPE') ]
            
        lst=[value for value in self.expected_gpe_in_data1 if value in actual_gpe_list]
        actual_percentage=len(lst)/len(self.expected_gpe_in_data1)
        self.assertEqual(actual_percentage,50/100)

    def test_ner_search_by_date(self):
        actual_date_list=[str(super(DefaultNerData1Test, self).get_text(entity)) for entity in self.entities if super(DefaultNerData1Test, self).get_type(entity,'DATE') ]
            
        lst=[value for value in self.expected_date_in_data1 if value in actual_date_list]
        actual_percentage=len(lst)/len(self.expected_date_in_data1)
        self.assertEqual(actual_percentage,80/100)
        
    def test_ner_search_by_money(self):
        actual_money_list=[str(super(DefaultNerData1Test, self).get_text(entity)) for entity in self.entities if super(DefaultNerData1Test, self).get_type(entity,'MONEY') ]
            
        lst=[value for value in self.expected_money_in_data1 if value[1:] in actual_money_list]
        actual_percentage=len(lst)/len(self.expected_money_in_data1)
        self.assertEqual(actual_percentage,100/100)
        
#load second intermediate message, catch entities
class DefaultNerData2Test(CommonUnitTest):
    expected_person_in_data2=['Jong CHEN','Jong CHEN','Jong CHEN','John','Jackie CHEN']
    expected_org_in_data2=['FINTRAC','HBCA','HBCA','HBCA']
    expected_date_in_data2=['21Dec2016','09Dec2018','14Jul2009','24Jan1992']
    expected_gpe_in_data2=['China','China','133 YOUNGE ST  Toronto ON M0G 7V8, Canada','18 STRANDHURST COURT Barrie ON L7F4L1, Canada' ] 

    @classmethod
    def setUpClass(cls):
        #fetch entities
        cls.entities = DefaultNerExClass().ner_search('data/data2.txt','en')

    @classmethod
    def tearDownClass(cls):
        #clear
        cls.entities=[]
        
    #
    def test_ner_search_by_person(self):
        actual_person_list=[str(super(DefaultNerData2Test, self).get_text(entity)) for entity in self.entities if super(DefaultNerData2Test, self).get_type(entity,'PERSON') ]
            
        lst=[value for value in self.expected_person_in_data2 if value in actual_person_list]
        actual_percentage=len(lst)/len(self.expected_person_in_data2)
        self.assertEqual(actual_percentage,100/100)

    def test_ner_search_by_org(self):

        actual_org_list=[ str(super(DefaultNerData2Test, self).get_text(entity)) for entity in self.entities if super(DefaultNerData2Test, self).get_type(entity,'ORG') ]
        
        lst=[value for value in self.expected_org_in_data2 if value in actual_org_list]
        actual_percentage=len(lst)/len(self.expected_org_in_data2)
        self.assertEqual(actual_percentage,100/100)

    def test_ner_search_by_date(self):
        actual_date_list=[str(super(DefaultNerData2Test, self).get_text(entity)) for entity in self.entities if super(DefaultNerData2Test, self).get_type(entity,'DATE') ]
            
        lst=[value for value in self.expected_date_in_data2 if value in actual_date_list]
        actual_percentage=len(lst)/len(self.expected_date_in_data2)
        self.assertEqual(actual_percentage,100/100)
        
    def test_ner_search_by_gpe(self):
        actual_gpe_list=[str(super(DefaultNerData2Test, self).get_text(entity)) for entity in self.entities if super(DefaultNerData2Test, self).get_type(entity,'GPE') ]
            
        lst=[value for value in self.expected_gpe_in_data2 if value in actual_gpe_list]
        actual_percentage=len(lst)/len(self.expected_gpe_in_data2)
        self.assertEqual(actual_percentage,100/100)

#load third longest message, catch entities     
class DefaultNerData3Test(CommonUnitTest): 
    expected_person_in_data3=['Min TZING','Sammy Sung']
    expected_org_in_data3=['Flamenco Realty','Amex Flamenco Realty']
    expected_date_in_data3=['13 Dec 2012','14 Dec 2012','17 Dec 2012','January 13th, 2013','Feb 25, 2013','Feb 24, 2013','06 Feb 2013']
    expected_gpe_fac_in_data3=['3330 Southbend Road, West Vancouver  C 2 V 9 K 7']
    expected_money_in_data3=['200,000','TWO MILLION Dollars','two million Canadian dollars','Four Million Dollars']

    @classmethod
    def setUpClass(cls):
        #fetch entities
        cls.entities = DefaultNerExClass().ner_search('data/data3.txt','en')

    @classmethod
    def tearDownClass(cls):
        #clear
        cls.entities=[]
        
    
    def test_ner_search_by_person(self):

        actual_person_list=[ str(super(DefaultNerData3Test, self).get_text(entity)) for entity in self.entities if super(DefaultNerData3Test, self).get_type(entity,'PERSON') ]
        
        lst=[value for value in self.expected_person_in_data3 if value in actual_person_list]
        actual_percentage=len(lst)/len(self.expected_person_in_data3)       
        self.assertEqual(actual_percentage,100/100)
        
    def test_ner_search_by_org(self):

        actual_org_list=[ str(super(DefaultNerData3Test, self).get_text(entity)) for entity in self.entities if super(DefaultNerData3Test, self).get_type(entity,'ORG') ]
        
        lst=[value for value in self.expected_org_in_data3 if value in actual_org_list]
        actual_percentage=len(lst)/len(self.expected_org_in_data3)
        self.assertEqual(actual_percentage,50/100)

        
    def test_ner_search_by_gpe_fac(self):
        actual_gpe_fac_list=[str(super(DefaultNerData3Test, self).get_text(entity)) for entity in self.entities if super(DefaultNerData3Test, self).get_type(entity,'GPE') or super(DefaultNerData3Test, self).get_type(entity,'FAC')]
            
        lst=[value for value in self.expected_gpe_fac_in_data3 if value in actual_gpe_fac_list]
        actual_percentage=len(lst)/len(self.expected_gpe_fac_in_data3)
        self.assertEqual(actual_percentage,0/100)

    def test_ner_search_by_date(self):
        actual_date_list=[str(super(DefaultNerData3Test, self).get_text(entity)) for entity in self.entities if super(DefaultNerData3Test, self).get_type(entity,'DATE') ]
            
        lst=[value for value in self.expected_date_in_data3 if value in actual_date_list]
        actual_percentage=len(lst)/len(self.expected_date_in_data3)
        self.assertTrue((actual_percentage == 100/100) or (round(actual_percentage,2) == 86/100) ) #unstable between somehow
        
    def test_ner_search_by_money(self):
        actual_money_list=[str(super(DefaultNerData3Test, self).get_text(entity)) for entity in self.entities if super(DefaultNerData3Test, self).get_type(entity,'MONEY') ]
            
        lst=[value for value in self.expected_money_in_data3 if value in actual_money_list]
        actual_percentage=len(lst)/len(self.expected_money_in_data3)
        self.assertEqual(actual_percentage,50/100)
        
if __name__ == '__main__':
    unittest.main()