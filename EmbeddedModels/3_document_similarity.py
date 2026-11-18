from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding=OpenAIEmbeddings(model='text-embedding-3-large',dimensions=300)

sportspersons = [
  "Virat Kohli is an Indian cricketer known for his aggressive batting and consistency.",
  "Cristiano Ronaldo is a Portuguese footballer known for his goal-scoring ability and athleticism.",
  "Lionel Messi is an Argentine footballer famous for his dribbling and playmaking skills.",
  "Neeraj Chopra is an Indian javelin thrower and Olympic gold medalist.",
  "MS Dhoni is a former Indian cricket captain known for his leadership and wicketkeeping.",
  "Roger Federer is a Swiss tennis player known for his elegant playing style.",
  "Rafael Nadal is a Spanish tennis player famous for his dominance on clay courts.",
  "PV Sindhu is an Indian badminton player and Olympic medalist.",
  "LeBron James is an American basketball player known for his versatility and athleticism.",
  "Usain Bolt is a Jamaican sprinter and world-record holder in the 100m and 200m."
]

query='tell me about virat kohli'

doc_embedding=embedding.embed_documents(sportspersons)
query_embedding=embedding.embed_query(query)

scores=cosine_similarity([query_embedding],doc_embedding)[0]

index,score=sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]
print(query)
print(sportspersons[index])