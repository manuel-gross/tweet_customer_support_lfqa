#this is the logger, may be useful
import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)
print("Logger initialized")

#here are all the other imports taken from haystack
from haystack.utils import print_documents
from haystack.nodes import Seq2SeqGenerator, DensePassageRetriever
from haystack.pipelines import GenerativeQAPipeline, DocumentSearchPipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.schema import EvaluationResult, MultiLabel, Label, Answer, Document
import re
import ast
import json
print("Imports complete")

#this here initializes the pipeline retriever
retriever = DensePassageRetriever(
    document_store=FAISSDocumentStore.load(index_path = "index", config_path = "config"),
    query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
    passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
)
print("Retriever set up")

#here is an example prompt, used for confirmation that the database is filled and indexed
p_retrieval = DocumentSearchPipeline(retriever)
res = p_retrieval.run(query="The device is not working", params={"Retriever": {"top_k": 10}})
print_documents(res, max_text_len=512)

#this initializes the generator, note that the files downloaded are in total at least 2gb in size
generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")
print("Generator set up")

#here the pipeline itself is generated
pipe = GenerativeQAPipeline(generator, retriever)
print("Pipeline initialized")

#set the amount of top k to return
setK = 3
print ("\n")
print ("How many top documents (top_k) should be listed? (Default = 3)")
print ("A higher number may increase the time it takes to answer a question.")
try:
    newK = int(input())
    setK = newK
except ValueError:
    print("Not a valid number, using default instead")
    
document_string = open("document.txt", encoding = "utf-8").read()
print('Evaluation document input is called "document.txt" by default')

output = open("eval_log.txt", "x", encoding="utf-8")
print("Log output will be written to eval_log.txt; be sure that file doesn't exist in the folder yet before running this.")

print("\nSystem is set up. You should be able to ask your questions now.")
#and this finally lets you ask
print("Reminder: You can always stop the running program by pressing CTRL-C\n")

while True:
    query = input("Query: ")
    answer = input("Expected answer: ")
    if((query != "") & (answer != "")):
        eval_labels = [
            MultiLabel(
                labels = [
                    Label(
                        query=query,
                        answer=Answer(
                            answer=answer
                        ),
                        document = Document(
                            content = document_string
                        ),
                        is_correct_answer=True,
                        is_correct_document=True,
                        origin="gold-label"
                    )
                ]
            )
        ]
        eval_result = pipe.eval(
            labels = eval_labels, params = {"Retriever": {"top_k":10}}, sas_model_name_or_path="cross-encoder/stsb-roberta-large"
        )
        
        print("\n")
        
        print(eval_result["Generator"]["answer"][0])
        print("F1-Score:")
        print(eval_result["Generator"]["f1"][0])
        print("Semantic Answer Similarity Score:")
        print(eval_result["Generator"]["sas"][0])
        
        output.write("Query:\n"+query+"\n\n")
        output.write("Expected answer:\n"+answer+"\n\n")
        output.write("Actual answer:\n"+str(eval_result["Generator"]["answer"][0])+"\n")
        output.write("F1-Score:\n"+str(eval_result["Generator"]["f1"][0])+"\n")
        output.write("Semantic Answer Similarity Score:\n"+str(eval_result["Generator"]["sas"][0])+"\n\n------------------------\n")
            
    else:
        print("Error: Please avoid typing in empty Strings")
        
    print("\n\n")