
import edu.stanford.nlp.pipeline.*;

import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
* This class is to do a named entity search for CoreNLP analysis
*
* @author  Hong Zheng
* @version 1.0
* @since   2020-04-06
*/
public class DefaultNerExample {

  public static void main(String[] args) throws Exception{
	  // do a ner search and get its result
	  final List<Entity> entityList=getNerSearch("data/data1.txt");
	  
	  System.out.println("Entities found as following:");
	  for (Entity entity:entityList) {
		  System.out.println(entity.getName()+"\t"+entity.getType());
	  }
  }
 
  public static List<Entity> getNerSearch(String filePaths) throws Exception {
	    // set up pipeline properties
	    final Properties props = new Properties();
	    
	    // the model ner mandatorily bundled with some other models
	    props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner");
	    props.setProperty("ner.applyFineGrained", "false");
	    //props.setProperty("ner.combinationMode", "HIGH_RECALL");
	    //props.setProperty("coref.algorithm", "neural");
	    
	    // customize fine grained ner
	    // props.setProperty("ner.fine.regexner.mapping", "example.rules");
	    // props.setProperty("ner.fine.regexner.ignorecase", "true");

	    // add additional rules, customize TokensRegexNER annotator
	    // props.setProperty("ner.additional.regexner.mapping", "example.rules");
	    // props.setProperty("ner.additional.regexner.ignorecase", "true");

	    // add 2 additional rules files ; set the first one to be case-insensitive
	    // props.setProperty("ner.additional.regexner.mapping", "ignorecase=true,example_one.rules;example_two.rules");

	    // set document date to be a specific date (other options are explained in the document date section)
	    // props.setProperty("ner.docdate.useFixedDate", "2019-01-01");

	    // only run rules based NER
	    // props.setProperty("ner.rulesOnly", "true");

	    // only run statistical NER
	    // props.setProperty("ner.statisticalOnly", "true");

	    
	    // build file path
	    final ClassLoader loader =DefaultNerExample.class.getClassLoader();
	    final URL filePath=loader.getResource(filePaths);
	    final String file = filePath.getPath().substring(1);//remove first useless letter
	    
	    // read the file content into a string
	    final String message=readFileAsString(file);
	    
	    // set up pipeline
	    final StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
	    // make a document
	    final CoreDocument doc = new CoreDocument(message);

	    // annotate the document
	    pipeline.annotate(doc);
	    
	    Entity entity;
	    final List<Entity> entityList=new ArrayList<Entity>();

	    for (CoreEntityMention em : doc.entityMentions()) {
	    	entity=new Entity(em.text(),em.entityType());
	    	entityList.add(entity);
	     }
	    
	    return entityList;
  }
  
  private static String readFileAsString(String fileName)throws Exception 
  { 
    return new String(Files.readAllBytes(Paths.get(fileName))); 
  } 
}
