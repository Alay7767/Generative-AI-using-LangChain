from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatOpenAI(model="gpt-3.5-turbo")

model2 = ChatAnthropic(model="claude-3-haiku-20240307")

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text'],
)

prompt2 = PromptTemplate(
    template='Generate 5 short question and answers from the following text \n {text}',
    input_variables=['text'],
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes --> {notes} and quiz --> {quiz}',
    input_variables=['notes', 'quiz'],
)

parser = StrOutputParser()

parallel_chain =RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser,
})

merge_chain = prompt3 | model2 | parser

chain = parallel_chain | merge_chain


text = """
Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy stored in glucose. It mainly occurs in the chloroplasts of plant cells. The green pigment chlorophyll absorbs sunlight, which drives the conversion of carbon dioxide (CO₂) and water (H₂O) into glucose (C₆H₁₂O₆) and oxygen (O₂).

This process consists of two main stages:

Light-dependent reactions: Occur in the thylakoid membranes where sunlight splits water molecules, releasing oxygen and producing ATP and NADPH.

Light-independent reactions (Calvin Cycle): Occur in the stroma where ATP and NADPH are used to fix CO₂ into glucose.

Photosynthesis is vital because it forms the basis of the food chain and maintains atmospheric oxygen levels.
"""

result = chain.invoke({'text':text})
print(result)

chain.get_graph().print_ascii()