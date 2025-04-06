from typing import Optional, Literal
from smolagents import Tool
from firecrawl import FirecrawlApp
import os
from dotenv import load_dotenv
from litellm import completion


"""
We also created this crawl tool, with which we started to be able
to actually go inside into different website contents. However, since
the challenge didn't permit us to use it, we didn't end up using it
to generate our final data frame.
"""


class CrawlTool(Tool):
    name = "scrape_site"
    description = """
    This tool accesses the provided web link, extracts all the text from the page and summarizes it.
    Call this tool to obtain information from a specific webpage.
    Call this tool if you also want to go inside into website contents.
    """
    inputs = {
        "link": {
            "type": "string",
            "description": "A valid url",
        },
        "query": {
            "type": "string",
            "description": "The query to be answered",
        },
    }
    output_type = "string"

    def forward(self, link: str, query: str):
        scrape_result = self.app.scrape_url(link, params={
            'formats': ['markdown'], 
            "onlyMainContent": True, 
            "excludeTags": ['a', 'img']
        })
        
        if scrape_result["metadata"]["statusCode"] == 200:
            # Get the markdown content
            markdown_content = scrape_result['markdown']
            
            # Truncate the markdown content to a safe size (approximately 20K tokens)
            # A rough estimate is about 4 characters per token
            max_chars = 100000  # This is approximately 20K tokens
            if len(markdown_content) > max_chars:
                markdown_content = markdown_content[:max_chars] + "\n\n[Content truncated due to length]"
            
            messages = [
                {"role": "system", "content": "Summarize the Context based on the query. Return all relevant information including relevant context and potential explanation."},
                {"role": "user", "content": f"Context:\n{markdown_content}\n\nQuestion: {query}"}
            ]
            
            # Get completion from LLM
            response = completion(
                model="fireworks_ai/accounts/fireworks/models/qwen2-vl-72b-instruct",
                messages=messages,
                temperature=0.2,
            )
            print(f"FIRECRAWL LLM: {response.choices[0].message.content}") 
            return response.choices[0].message.content
        return "Error: Website scraping unsuccessful."

    def setup(self):
        load_dotenv()
        self.app=FirecrawlApp(api_key=os.getenv('FIRECRAWL_API_KEY'))

"""
# Create an instance of the tool
tool = CrawlTool()

# Set up the tool (initialize the app)
tool.setup()

# Call the forward method with a query
result = tool.forward("https://firecrawl.dev")
print(result)
"""