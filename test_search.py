from routers.web_helpers import run_web_search
import asyncio

def test_search():
    print("--- Testing Full Search with Scraping ---")
    # run_web_search is synchronous now, so we can just call it
    result = run_web_search("can you tell me what is happening in today world in detail")
    print(result)

if __name__ == "__main__":
    test_search()
