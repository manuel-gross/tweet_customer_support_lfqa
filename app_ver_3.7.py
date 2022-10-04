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

filesCreatedCounter = 0
#set whether to write an output file
doWrite = False
print("\n")
print("Should the system write the output to a separate file? (y/n)")
print("This may increase the time it takes to answer a question.")
doW = input()
if doW == 'y':
    doWrite = True

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


print("\nSystem is set up. You should be able to ask your questions now.")
#and this finally lets you ask
print("Reminder: You can always stop the running program by pressing CTRL-C\n")

while True:
    text = input("Query: ")
    if(text != ""):
        res = pipe.run(
            query=text, params={"Retriever": {"top_k": setK}}
        )
        
        print("\n")
        
        #Convert the dict output to something readable
        #because this thing is dumb and has angular brackets (these: <>) for SOME REASON
        out1 = res['answers']
        out2 = str(out1)
        out3 = re.sub(r"\[<Answer {", "{", out2)
        out4 = re.sub(r"]}}>]", "]}}", out3)
        val = ast.literal_eval(out4)
        
        print(val['answer'])
        
        if(doWrite == True):
            filename = "returnLogFull"
            filename += str(filesCreatedCounter)
            filename += ".txt"
            
            #I would've wanted to make it a json. Unfortunately it's not in json format
            f = open(filename, "w", encoding = "utf-8")
            f.write(str(res))
            f.close()
            filesCreatedCounter += 1
            
            print("\nFull return written to " + filename)
            
    else:
        print("Error: Please avoid typing in empty Strings")
        
    print("\n\n")