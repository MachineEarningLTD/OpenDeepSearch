from dataclasses import dataclass
from typing import List, Optional, Tuple
from opendeepsearch.context_scraping.crawl4ai_scraper import WebScraper
from opendeepsearch.ranking_models.infinity_rerank import InfinitySemanticSearcher
from opendeepsearch.ranking_models.jina_reranker import JinaReranker
from opendeepsearch.ranking_models.chunker import Chunker 
from dotenv import load_dotenv
import os

load_dotenv()


from firecrawl import FirecrawlApp

@dataclass
class Source:
    link: str
    html: str = ""
    # Add other relevant fields here

class SourceProcessor:
    def __init__(
        self, 
        top_results: int = 5,
        strategies: List[str] = ["no_extraction"],
        filter_content: bool = True,
        reranker: str = "infinity"
    ):
        self.strategies = strategies
        self.filter_content = filter_content
        self.scraper = WebScraper(
            strategies=self.strategies, 
            filter_content=self.filter_content
        )
        self.top_results = top_results
        self.chunker = Chunker()
        
        # Initialize the appropriate reranker
        if reranker.lower() == "jina":
            self.semantic_searcher = JinaReranker()
            print("Using Jina Reranker")
        else:  # default to infinity
            self.semantic_searcher = InfinitySemanticSearcher()
            print("Using Infinity Reranker")

        self.firecrawl=FirecrawlApp(api_key=os.getenv('FIRECRAWL_API_KEY'))


    async def process_sources(
        self, 
        sources: List[dict], 
        num_elements: int, 
        query: str, 
        pro_mode: bool = False
    ) -> List[dict]:
        try:
            valid_sources = self._get_valid_sources(sources, num_elements)
            if not valid_sources:
                return sources

            #html_contents = await self._fetch_html_contents([s[1]['link'] for s in valid_sources])
            """
            html_contents = []

            for s in valid_sources:
                scrape_result = self.app.scrape_url(s[1]['link'], params={'formats': ['markdown']})
                print(f"FIRECRAWL: {scrape_result}")
                if scrape_result["metadata"]["statusCode"] == 200:
                    html_contents.append(scrape_result["markdown"])
                html_contents.append("Error: Website scraping unsuccessful.")

            """

            for s in valid_sources:
                print(f"LInks: {s[1]['link']}")


            html_contents = await self._fetch_html_contents_new([s[1]['link'] for s in valid_sources])

            print(f"Fetched HTML contents: {html_contents}")

            #print(f"Fetched HTML contents: {html_contents_new}")

            return self._update_sources_with_content(sources.data, valid_sources, html_contents, query)
        except Exception as e:
            print(f"Error in process_sources: {e}")
            return sources

    def _get_valid_sources(self, sources: List[dict], num_elements: int) -> List[Tuple[int, dict]]:
        return [(i, source) for i, source in enumerate(sources.data['organic'][:num_elements]) if source]
    
    async def _fetch_html_contents(self, links: List[str]) -> List[str]:
        print("Links to scrape: {links}")
        raw_contents = await self.scraper.scrape_many(links)
        return [x['no_extraction'].content for x in raw_contents.values()]
    
    async def _fetch_html_contents_new(self, links: List[str]) -> List[str]:
        print(f"Links to scrape: {links}")  # Fixed the f-string format
        html_contents = []
        #from src.opendeepsearch.context_scraping.utils import filter_quality_content
        
        for link in links:
            scrape_result = self.firecrawl.scrape_url(link, params={'formats': ['markdown']})
            print(f"FIRECRAWL: {scrape_result}")
            if scrape_result["metadata"]["statusCode"] == 200:
                html_contents.append(scrape_result["markdown"])#(filter_quality_content(scrape_result["markdown"]))
            else:
                html_contents.append("Error: Website scraping unsuccessful.")
        
        return html_contents

    def _process_html_content(self, html: str, query: str) -> str:
        if not html:
            return ""
        try:
            # Split the HTML content into chunks
            documents = self.chunker.split_text(html)
            
            # Rerank the chunks based on the query
            reranked_content = self.semantic_searcher.get_reranked_documents(
                query,
                documents,
                top_k=self.top_results
            )
            
            return reranked_content
        
        except Exception as e:
            print(f"Error in content processing: {e}")
            return ""

    def _update_sources_with_content(
        self, 
        sources: List[dict],
        valid_sources: List[Tuple[int, dict]], 
        html_contents: List[str],
        query: str
    ) -> List[dict]:
        for (i, source), html in zip(valid_sources, html_contents):
            source['html'] = html#self._process_html_content(html, query)
            # sources[i] = source
        return sources