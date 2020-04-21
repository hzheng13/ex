

import edu.stanford.nlp.coref.CorefCoreAnnotations;
import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.coref.data.Mention;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

import java.io.InputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import org.apache.commons.io.IOUtils;


public class CorefExample {

  public static void main(String[] args) throws Exception{
	  // do a ner search and get its result
	  printCorefs("data3.txt");	
  }

  public static void printCorefs(String filePaths) throws Exception {
	// set up pipeline properties
	
	final Properties props = new Properties();
	
	// the model ner mandatorily bundled with some other models
	props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,coref");
	props.setProperty("ner.combinationMode", "NORMAL");
	props.setProperty("ner.applyFineGrained", "false");
	//props.setProperty("coref.algorithm", "neural");
	 
	// build file path
	InputStream data = CorefExample.class.getClassLoader().getResourceAsStream(filePaths);
	String message = IOUtils.toString(data, StandardCharsets.UTF_8.name());
	System.out.println("here");
	// set up pipeline
	final StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
	// make a document
	//final CoreDocument doc = new CoreDocument(message);
	Instant start = Instant.now();
	
	final Annotation document = new Annotation(message);
	// annotate the document
	pipeline.annotate(document);
	
	Instant finish = Instant.now();
	
	System.out.println("Annotation took: " + Duration.between(start, finish) + " second");
	System.out.println("-----");

	for (CorefChain cc : document.get(CorefCoreAnnotations.CorefChainAnnotation.class).values()) {
			System.out.println("\t" + cc);
			for (CorefChain.CorefMention mention : cc.getMentionsInTextualOrder()) {
				System.out.println("\t" + mention);
			}
	}
	for (CoreMap sentence : document.get(CoreAnnotations.SentencesAnnotation.class)) {
	    System.out.println("---");
	    System.out.println("mentions");
	    for (Mention m : sentence.get(CorefCoreAnnotations.CorefMentionsAnnotation.class)) {
	    		System.out.println("\t" + m);
	    }
	}
  }
  
}
