from typing import Optional, Literal
from smolagents import Tool
from firecrawl import FirecrawlApp
import os
from dotenv import load_dotenv

class CrawlTool(Tool):
    name = "scrape_site"
    description = """
    Accesses the provided web link and extracts all the text from the page.
    This is useful for gathering information from a specific webpage."""
    inputs = {
        "query": {
            "type": "string",
            "description": "The web link to scrape",
        },
    }
    output_type = "string"

    def forward(self, query: str):
        scrape_result = self.app.scrape_url('query', params={'formats': ['markdown']})
        print(f"FIRECRAWL: {scrape_result}")
        if scrape_result["metadata"]["statusCode"] == 200:
            return scrape_result["markdown"]
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