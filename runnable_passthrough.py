
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
import os
load_dotenv()


llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
     huggingfacehub_api_token=os.getenv(
        "HUGGINGFACEHUB_ACCESS_TOKEN")

)

model= ChatHuggingFace(llm=llm)



prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2= PromptTemplate(
    template= 'Generate a Linkedin post about {topic}',
    input_variables=['topic']
)


parser= StrOutputParser()

joke_gen_chain= RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'explanation':RunnableSequence(prompt2, model, parser)

})

final_chain= RunnableSequence(joke_gen_chain, parallel_chain)

result= final_chain.invoke({'topic':'Cricket'})

print(result)
