

import edu.stanford.nlp.ie.util.RelationTriple;
import edu.stanford.nlp.naturalli.*;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.PropertiesUtils;


import java.io.InputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Properties;


public class OpenIEExample {

  public static void main(String[] args) throws Exception{
	  // do a ner search and get its result
	  printCorefs("data3.txt");
  }
 
  public static void printCorefs(String filePaths) throws Exception {
	    // set up pipeline properties
	    final Properties props = new Properties();
	    
	    // the model ner mandatorily bundled with some other models
	    props.setProperty("annotators", "tokenize,ssplit,pos,lemma,depparse, natlog, openie");
	    //props.setProperty("coref.algorithm", "neural");
	    
	    // build file path
	    InputStream data = OpenIEExample.class.getClassLoader().getResourceAsStream(filePaths);
	    String message = org.apache.commons.io.IOUtils.toString(data, StandardCharsets.UTF_8.name());
	    // set up pipeline
	    final StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
	    // make a document
	    //final CoreDocument doc = new CoreDocument(message);
	    final Annotation document = new Annotation(message);
	    // annotate the document
	    pipeline.annotate(document);
	    
	 // Loop over sentences in the document
	    int sentNo = 0;
	    for (CoreMap sentence : document.get(CoreAnnotations.SentencesAnnotation.class)) {
	      System.out.println("Sentence #" + ++sentNo + ": " + sentence.get(CoreAnnotations.TextAnnotation.class));

	      // Print SemanticGraph
	      System.out.println(sentence.get(SemanticGraphCoreAnnotations.EnhancedDependenciesAnnotation.class).toString(SemanticGraph.OutputFormat.LIST));

	      // Get the OpenIE triples for the sentence
	      Collection<RelationTriple> triples = sentence.get(NaturalLogicAnnotations.RelationTriplesAnnotation.class);

	      // Print the triples
	      for (RelationTriple triple : triples) {
	    	  	System.out.println("------");
	    	  	System.out.println("triple");
	        System.out.println(triple.confidence + "\t" +
	            triple.subjectGloss() + "\t" +
	            triple.relationGloss() + "\t" +
	            triple.objectGloss());
	      }

	      // Alternately, to only run e.g., the clause splitter:
	      List<SentenceFragment> clauses = new OpenIE(props).clausesInSentence(sentence);
	      for (SentenceFragment clause : clauses) {
	        System.out.println(clause.parseTree.toString(SemanticGraph.OutputFormat.LIST));
	      }
	      System.out.println();
	    }
  }
 
}
