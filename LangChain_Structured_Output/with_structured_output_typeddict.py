from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = ChatOpenAI()

# schema
class Review(TypedDict):
    summary : str
    sentiment : str
    
structured_model = model.with_structured_output(Review)

result = structured_model.invoke("I recently bought the AeroBrew Smart Coffee Maker, and I have to say, it’s been a game-changer for my mornings. The app connectivity is surprisingly smooth—I can schedule my brew the night before, and by the time I get out of bed, my kitchen smells like a coffee shop. The build feels premium, and the sleek matte finish looks great on my counter.")

print(result)