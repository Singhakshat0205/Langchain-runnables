
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
import os
load_dotenv()


llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
     huggingfacehub_api_token=os.getenv(
        "HUGGINGFACEHUB_ACCESS_TOKEN")

)

model= ChatHuggingFace(llm=llm)

parser= StrOutputParser()

prompt1 = PromptTemplate(
    template='Generate a detailed report about {topic}',
    input_variables=['topic']
)

prompt2= PromptTemplate(
    template='Summarize the following text in under 100 words\n {text}',
    input_variables=['text']
)


report_gen_chain= RunnableSequence(prompt1, model, parser)
report_gen_chain= prompt1 | model | parser


branch_chain= RunnableBranch(
    (lambda x:len(x.split()) > 100, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)


final_chain= RunnableSequence(report_gen_chain, branch_chain)



result= final_chain.invoke({'topic':'Russia Vs Ukraine'})

print(result)


