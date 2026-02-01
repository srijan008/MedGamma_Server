from .services import llm
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
import traceback

def fetch_content(url: str, max_chars: int = 2000) -> str:
    """
    Fetches the URL and extracts text from <p> tags.
    """
    try:
        print(f"🕸️ Scraping: {url}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text_content = "\n".join([p.get_text() for p in paragraphs])
        
        # Clean up whitespace
        text_content = " ".join(text_content.split())
        
        return text_content[:max_chars]
    except Exception as e:
        print(f"⚠️ Scraping Failed for {url}: {e}")
        return ""

def run_web_search(query: str) -> str:
    try:
        print(f"🔎 Searching DDG News for: {query}")
        # Use .news() to get specific articles rather than homepages
        # This prevents scraping generic "Index" pages like cnn.com
        results = list(DDGS().news(query, max_results=3)) 
        
        if not results:
             # Fallback to text search if news fails
             print("⚠️ No news results, falling back to text search...")
             results = list(DDGS().text(query, max_results=3))
             
        if not results:
            return ""
            
        formatted_output = ""
        
        # Format top 3 snippets
        formatted_output += "**Search Highlights:**\n"
        for r in results:
            title = r.get('title', 'No Title')
            link = r.get('href', r.get('url', '#'))
            body = r.get('body', '')
            formatted_output += f"- [{title}]({link}): {body}\n"
        
        # Deep Dive into the Top Result
        if results:
            top_result = results[0]
            top_url = top_result.get('href', top_result.get('url'))
            if top_url:
                print(f"🚀 Fetching deep content from: {top_url}")
                content = fetch_content(top_url)
                if content:
                    formatted_output += f"\n\n**Detailed Concept from Top Source ({top_result.get('title', 'Source')}):**\n{content}\n"
                else:
                    # Fallback to second result
                    if len(results) > 1:
                        sec_result = results[1]
                        sec_url = sec_result.get('href', sec_result.get('url'))
                        if sec_url:
                             content = fetch_content(sec_url)
                             if content:
                                  formatted_output += f"\n\n**Detailed Concept from source ({sec_result.get('title', 'Source')}):**\n{content}\n"

        return formatted_output
    except Exception as e:
        print(f"🔥 Web Search Error: {e}")
        traceback.print_exc()
        return ""

async def route_query(query: str, chat_history_str: str) -> str:
    """
    Decides if the query needs web access. 
    Returns 'WEB' or 'CHAT'.
    """
    router_prompt = f"""
    You are a routing assistant. Decide if the latest user query requires external knowledge from the Internet (e.g. current events, specific facts not in conversation, up-to-date info) or if it can be answered from the chat history.
    
    Conversation History Summary:
    {chat_history_str}
    
    Latest Query: {query}
    
    Answer ONLY with 'WEB' if internet search is needed, or 'CHAT' if not needed.
    """
    try:
        # Use a lower temp for deterministic routing
        response = llm.invoke(router_prompt)
        decision = response.content.strip().upper()
        if "WEB" in decision:
            return "WEB"
        return "CHAT"
    except Exception as e:
        print(f"⚠️ Router Error: {e}")
        return "CHAT"
