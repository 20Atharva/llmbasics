from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model=ChatAnthropic(model='claude-5-sonnet')

result=model.invoke('What is capital of up')

print(result)