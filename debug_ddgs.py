from ddgs import DDGS
import json

def debug_news_keys():
    try:
        print("Fetching news...")
        results = list(DDGS().news("latest news", max_results=1))
        if results:
            print("Keys found in news result:", results[0].keys())
            print(json.dumps(results[0], indent=2))
        else:
            print("No news results found.")

        print("\nFetching text...")
        text_results = list(DDGS().text("latest news", max_results=1))
        if text_results:
            print("Keys found in text result:", text_results[0].keys())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_news_keys()
