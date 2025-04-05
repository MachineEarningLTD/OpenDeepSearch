from opendeepsearch import OpenDeepSearchTool
import os
import requests

os.environ["SERPER_API_KEY"] = "16c5da0d588ea2065171cc4a8ee8b934a660101d"  # If using Serper
os.environ["JINA_API_KEY"] = "jina_150ccbf475584a92afe87a5b2d8981dcb8-_62YowjWdCr0C_nU7wM7T9cJf"
os.environ["FIREWORKS_API_KEY"]="fw_3ZWPyAMiMEbfV8vorFT2gpcV"

# Using Serper (default)
search_agent = OpenDeepSearchTool(
    model_name="fireworks_ai/accounts/fireworks/models/llama-v3-70b-instruct",
    reranker="jina"
)

if not search_agent.is_initialized:
    print("Initializing search_agent")
    search_agent.setup()
    
query = "Fastest land animal?"

cumDump = requests.post(
                "https://google.serper.dev/search",
                headers={
                    'X-API-KEY': "16c5da0d588ea2065171cc4a8ee8b934a660101d",
                    'Content-Type': 'application/json'
                    # What does content-type do?
                },
                json={
                "q": "Who is the current President of the united states",
                "num": 10,
                "gl": "us"
            },
            timeout=10
            )
dump = cumDump.json()
print(dump)
print("dump type: ")
print(type(dump))
print("\n\n\n\n")
print(dump["organic"])
print("\n\n\n")
for i in range(0,10):
    print(dump["organic"][i])
    print("\n\n\n")

result = search_agent.forward(query)
print(result)