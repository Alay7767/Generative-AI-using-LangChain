from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableBranch

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a detailed report on a {topic}',
    input_variables=['topic'],
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text'],
)

model = ChatOpenAI(model="gpt-3.5-turbo")

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    # (condition, runnable),
    (lambda x : len(x.split())>500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

result = final_chain.invoke({'topic':'Argentina vs France 2022 FIFA World Cup Final'})

print(result)