from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic} \n",
    input_variables=['topic'],
    
)

model = ChatOpenAI(model="gpt-3.5-turbo")

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic':'Moon'})
print(result)

chain.get_graph().print_ascii()