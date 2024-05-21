#################################################################################################################################
#IMPORTS
#################################################################################################################################
#These imports need to be installed for the code to run - to install them in your environment use pip install command in the terminal
#Heroku requires these libraries + dependencies in a .txt file. Use command pip freeze > requirements.txt to create it
import os
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from flask import Flask, request, jsonify, send_from_directory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI 
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores import DeepLake


#Loads the .env file - all the API keys are in this file for security reasons
load_dotenv()
  

#Here we call the API-Keys we need for the Vectorstore and for using Chatgpt and set them as environment variable
ACTIVELOOP_TOKEN = os.getenv("ACTIVELOOP_TOKEN")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')



#We define the OpenAI model - gpt 4 allows for the highest token size, but is also 10x more expensive than other options
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
model_name = 'text-embedding-ada-002'  

#We specify the embedding model - there are other options, ada-002 is the standard
embeddings = OpenAIEmbeddings(  
    model=model_name,  
    openai_api_key=OPENAI_API_KEY 
)  


#defining the vectorstore
dataset_path = "link to dataset"
vectorstore = DeepLake(dataset_path=dataset_path, read_only=True, embedding=embeddings) 

#We set up the self-query retriever, that means that the retrievers filters the vectors by metadata based on the question
#For that, first we need to tell the retriever what metadata there is to use as filter:
metadata_field_info = [
    AttributeInfo(
        name="Source",
        description="The Source of the data",
        type="string",
    ),
    AttributeInfo(
        name="key_concepts",
        description="Key concepts and important terms",
        type="string",
    ),
    AttributeInfo(
        name="tables",
        description="name of tables and illustrations",
        type="string",
    ),
    AttributeInfo(
        name="formula",
        description="name of math formulas",
        type="string"
    ),
]

document_content_description = "The course contents: a theory book on operations managements and tutorial videos"

#Now we implement the selfqueryretriever
retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info, verbose=True
)

#Sets how wide the search is
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['k'] = 5





#################################################################################################################################
#Templates
#################################################################################################################################
#Empty container for Chat history, in this case it is stored client-side
#I recommend storing the chat-history somewhere else when deploying the chatbot for real
#Refer to this for options https://python.langchain.com/docs/use_cases/chatbots/memory_management/
chat_history = []

#Response schema is for when we want to output multiple things, the schema tells the llm how to format the output
response_schemas = [
    ResponseSchema(
        name="answer",
        description="Put in a Html Linebreak <br> or <br><br> after sentences."),
    ResponseSchema(
        name="source",
        description="Decide based of the answer if a source is necessary:\
           if YES - compile the 3 most relevant Sources from the context. Information is found under metadata-Sources. Format it like this: <br><br><b>Related Course Materials:</b> <br>source1 <br>source2 <br> etc.\
           if NO - leave empty",
    ),
]

#Based on this schema we construct an output parser who formats the final output
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

#The template for the format prompt - we define the input variables
format_prompt = PromptTemplate(
    template="Follow the format instructions.\n{format_instructions}\nanswer:{answer}\ncontext: {context}",
    input_variables=["answer", "context"],
    partial_variables={"format_instructions": format_instructions},
)

#This template rewrites the prompt, so it takes the chat-history into account and send it to the retriever
retriever_template = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
retriever_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", retriever_template),
        MessagesPlaceholder("chat_history"), #messageplaceholder helps running the code even when chat-history is empty
        ("human", "{input}"),
    ]
)

#Second template: gets passed to the llm and returns the answer to our question
prompt_template = """You are the assistant for the course Operations Management. \
    Be concise, NOT more than 5 sentences.\
    Use the following pieces of retrieved context to answer the question. \


{context}"""#Context is where the retrieved content from the vectorstore is put
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        
    ]
)

#################################################################################################################################
#Chains
#################################################################################################################################

#Chain that retrieves our source documents
history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=retriever_prompt
)

#Normal Conversational chain for the llm
question_answer_chain = create_stuff_documents_chain(llm, prompt)

#Combines both chains above - runs the retrieval first and then passes the context to the qa chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

#Finally, this chain passes the answer and context to the llm once more and it returns a formatted answer.
#This step is extremely unefficient but it ensures that we get the correct answer format consistently
format_chain = (
    RunnablePassthrough.assign(context=rag_chain, answer=rag_chain)
    | format_prompt
    | llm
    | output_parser
)


"""
# this is here for quickly testing the chain - gives the answer in the terminal 
x = format_chain.invoke({"input": "vertriebsschätzung", "chat_history": chat_history})
print (x)
"""

# This function takes the question and chat_history as input parameters, and returns the llm response
def process_chat(question, chat_history):
    response = format_chain.invoke({
        "chat_history": chat_history,
        "input": question,
    })
    
    return f"{response['answer']} \n {response['source']}"


#################################################################################################################################
#WebAPP
#################################################################################################################################


app = Flask(__name__)


chat_history = []

@app.route('/')
def index():
    return open('indexnew.html').read()

#takes the message by the user and runs the process_chat funktion. Appends the messages into the chat_history container 
@app.route('/process_message', methods=['POST'])
def process_message():
    global chat_history
    data = request.json
    question = data.get('question')
    bot_response = process_chat(question, chat_history)

    chat_history.append(('user', question))
    chat_history.append(('ai', bot_response))

    chat_history = chat_history[-4:] 

    return jsonify({"bot_response": bot_response})

#This app route is solely for testing reasons, can be used to see the content of chat_history
@app.route('/chat_history')
def get_chat_history():
    return jsonify({"chat_history": chat_history})

#loads the images
@app.route('/user.png')
def serve_userimage():
    return send_from_directory(app.root_path, 'user.png')

@app.route('/bot.png')
def serve_botimage():
    return send_from_directory(app.root_path, 'bot.png')

if __name__ == '__main__':
    app.run(debug=True)
