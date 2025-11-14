from langchain.llms import openai
from langchain.prompts import PromptTemplate

llm = openai(model_name="gpt-3.5-turbo", temperature = 0.7)

prompt = PromptTemplate(
    input_variables=['topic'],
    template='Suggest a catchy blog title about {topic}',
)

topic = input('Enter name of topic :')

formatted_prompt = prompt.format(topic=topic)

blog_title = llm.predict(formatted_prompt)

print("Generate Blog title :", blog_title)