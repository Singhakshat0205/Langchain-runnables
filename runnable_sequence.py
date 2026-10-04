

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
import os
load_dotenv()

prompt1= PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt2= PromptTemplate(
    template='Explain the following joke in funny way- {text}' ,
    input_variables=['text'] 
)

llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
     huggingfacehub_api_token=os.getenv(
        "HUGGINGFACEHUB_ACCESS_TOKEN")

)


model= ChatHuggingFace(llm= llm)

parser= StrOutputParser()

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)


print(chain.invoke({'topic':'AI'}))

