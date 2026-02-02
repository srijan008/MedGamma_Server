from .services import llm
try:
    from duckduckgo_search import DDGS
except ImportError:
    from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
import traceback

def fetch_content(url: str, max_chars: int = 8000) -> str:
    """
    Fetches the URL and extracts text using heuristics to find the main content.
    """
    try:
        print(f"🕸️ Scraping: {url}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. Remove unwanted elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            script.decompose()
            
        # 2. Try to find the main content area
        content_node = soup.find('article')
        if not content_node:
            content_node = soup.find('main')
            
        # 3. Fallback: Look for divs with specific classes/ids
        if not content_node:
             possible_content_roots = soup.find_all('div', {"class": ["content", "main", "post-content", "article-body", "story-body"]})
             if possible_content_roots:
                 # Pick the one with the most text
                 content_node = max(possible_content_roots, key=lambda x: len(x.get_text()))
        
        # 4. Fallback to body if nothing specific found
        if not content_node:
            content_node = soup.body

        if not content_node:
            return ""

        # 5. Extract text with separator
        text_content = content_node.get_text(separator='\n', strip=True)
        
        # 6. Basic cleanup (collapsing multiple newlines)
        import re
        text_content = re.sub(r'\n{3,}', '\n\n', text_content)
        # print("text_content", text_content)
        return text_content[:max_chars]
    except Exception as e:
        print(f"⚠️ Scraping Failed for {url}: {e}")
        return ""

def run_web_search(query: str) -> str:
    try:
        print(f"🔎 Searching DDG News for: {query}")
        # Use .news() to get specific articles rather than homepages
        results = list(DDGS().news(query, max_results=5)) 
        
        if not results:
             # Fallback to text search if news fails
             print("⚠️ No news results, falling back to text search...")
             results = list(DDGS().text(query, max_results=5))
             
        if not results:
            return ""
            
        formatted_output = "**Search Highlights:**\n"
        
        # 1. Add snippets for top 3
        for r in results[:3]:
            title = r.get('title', 'No Title')
            link = r.get('href', r.get('url', '#'))
            body = r.get('body', '')
            formatted_output += f"- [{title}]({link}): {body}\n"
        
        # 2. Deep Dive: Try to find ONE good article (iterating through top 3)
        best_content = ""
        used_source = ""
        
        for i, result in enumerate(results[:3]):
            url = result.get('href', result.get('url'))
            title = result.get('title', 'Source')
            
            if not url:
                continue
                
            # Skip puzzles/games that often appear in "news"
            if any(x in title.lower() for x in ['crossword', 'puzzle', 'wordle', 'sudoku', 'connections']):
                print(f"⏩ Skipping puzzle result: {title}")
                continue

            print(f"🚀 Attempt {i+1}: Fetching deep content from: {url}")
            content = fetch_content(url)
            
            # heuristic: if content is > 500 chars, it's probably a full article
            if len(content) > 500:
                best_content = content
                used_source = title
                print(f"✅ Found good content ({len(content)} chars) from {title}")
                break
            elif len(content) > len(best_content):
                # keep the longest one we found so far even if it's short
                best_content = content
                used_source = title
        
        if best_content:
             formatted_output += f"\n\n**Detailed Concept from {used_source}:**\n{best_content}\n"
        else:
            formatted_output += "\n\n(Could not scrape full article content from top results. Rely on snippets above.)"

        return formatted_output
    except Exception as e:
        print(f"🔥 Web Search Error: {e}")
        traceback.print_exc()
        return ""

# async def route_query(query: str, chat_history_str: str) -> str:
#     """
#     Decides if the query needs web access. 
#     Returns 'WEB' or 'CHAT'.
#     """
#     router_prompt = f"""
#     You are a routing assistant. Decide if the latest user query requires external knowledge from the Internet (e.g. current events, specific facts not in conversation, up-to-date info) or if it can be answered from the chat history.
    
#     Conversation History Summary:
#     {chat_history_str}
    
#     Latest Query: {query}
    
#     Answer ONLY with 'WEB' if internet search is needed, or 'CHAT' if not needed.
#     """
#     try:
#         # Use a lower temp for deterministic routing
#         response = llm.invoke(router_prompt)
#         decision = response.content.strip().upper()
#         if "WEB" in decision:
#             return "WEB"
#         return "CHAT"
#     except Exception as e:
#         print(f"⚠️ Router Error: {e}")
#         return "CHAT"
