import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


import cohere
co = cohere.Client(os.getenv('COHERE_API_KEY'))

import weaviate
auth_config = weaviate.auth.AuthApiKey(
    api_key=os.getenv('WEAVIATE_API_KEY'))

client = weaviate.Client(
    url="https://cohere-demo.weaviate.network/",
    auth_client_secret=auth_config,
    additional_headers={
        "X-Cohere-Api-Key": os.getenv('COHERE_API_KEY'),
    }
)

from utils import dense_retrieval


# In[78]:


query = "What is the capital of Canada?"


# In[126]:


################################################
# 2.1 Apply Dense Retrieval to a query
################################################
dense_retrieval_results = dense_retrieval(query, 
     client)


# In[127]:


from utils import print_result


# In[128]:


################################################
# 2.2 Print the result of the Dense Retrieval to 
#     a query
################################################
print_result(dense_retrieval_results)


################################################
# 3. Improving Keyword Search with ReRank
################################################

# In[84]:


from utils import keyword_search


# In[85]:


query_1 = "What is the capital of Canada?"


# In[112]:


################################################
# 3.1 Keyword Search with 3 results
################################################
query_1 = "What is the capital of Canada?"
results = keyword_search(query_1,
  client,
  properties=["text", "title", "url", "views", 
        "lang", 
        "_additional {distance}"],
  num_results=3
  )

for i, result in enumerate(results):
    print(f"i:{i}")
    print(result.get('title'))
    print(result.get('text'))


# In[ ]:


################################################
# 3.2 Keyword Search with 500 results
################################################
query_1 = "What is the capital of Canada?"
results = keyword_search(query_1,
   client,
   properties=["text", "title", "url", "views", 
               "lang", 
               "_additional {distance}"],
   num_results=500
   )

for i, result in enumerate(results):
    print(f"i:{i}")
    print(result.get('title'))
    #print(result.get('text'))


# In[113]:


################################################
# 3.3 ReRank of the Keyword Search results
################################################
def rerank_responses(query, responses, 
         num_responses=10):
    reranked_responses = co.rerank(
        model = 'rerank-english-v2.0',
        query = query,
        documents = responses,
        top_n = num_responses,
        )
    return reranked_responses


# In[114]:


texts = [result.get('text') for result in 
         results]
reranked_text = rerank_responses(query_1, 
         texts)


# In[115]:


for i, rerank_result in enumerate(reranked_text):
    print(f"i:{i}")
    print(f"{rerank_result}")
    print()

################################################
# 4. Improving Dense Retrieval with ReRank
################################################

# In[116]:


from utils import dense_retrieval


# In[117]:


query_2 = "Who is the tallest person in history?"


# In[130]:


################################################
# 4.1 Dense Retrieval of a new query
################################################
results = dense_retrieval(query_2,client)


# In[132]:


for i, result in enumerate(results):
    print(f"i:{i}")
    print(result.get('title'))
    print(result.get('text'))
    print()



################################################
# 4.2 ReRank the Dense Retrieval of a 
#     new query
################################################
texts = [result.get('text') for result 
         in results]
reranked_text = rerank_responses(query_2, 
         texts)



for i, rerank_result in enumerate(
        reranked_text):
    print(f"i:{i}")
    print(f"{rerank_result}")
    print()


