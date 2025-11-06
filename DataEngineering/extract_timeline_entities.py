# import libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from datetime import datetime, timedelta, timezone

from sympy import false
from tqdm import tqdm
import json
import os
import re
from collections import Counter, defaultdict
from typing import List, Dict, Set,Any, Tuple
from difflib import SequenceMatcher
from openai import OpenAI
from pydantic import BaseModel


OPENAI_API_KEY="test"

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------
# 1. Extract timeline
# ---------------------------------------------------------------------
def extract_timeline(news):
    """
    Group articles by date and ask GPT-5 to summarize each day's major event.
    Returns:
        timeline_list = [
            {"date": "YYYY-MM-DD", "event": "..."},
            ...
        ]
    """
    # group by date
    by_date = defaultdict(list)
    for item in news:
        d = item.get("date")
        if d:
            by_date[d].append(item)

    timeline_list = []
    # process each date, oldest to newest,come up with one-line summary
    for d, items in sorted(by_date.items(), key=lambda x: x[0]):
        # prepare short text for the model
        joined = "\n\n".join([
            f"- Title: {it.get('title','')}\n  Text: {it.get('text','')}"
            for it in items
        ])

        system = (
            """
            You are an expert data summarizer. Given several news snippets 
            about events on the SAME date, produce ONE concise 
            timeline entry.
            """
            # """
            # You are given several text messages that were recorded on the same day.
            # Your task is to identify and extract the distinct events mentioned in these messages.
            # """
        )
        user = f"Date: {d}\nNews snippets:\n{joined}\n\nReturn only the final one-line event."

        response = client.responses.create(
            model="gpt-5",
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        content=response.output_text
        timeline_list.append({"date": d, "event": content})

    return timeline_list


# ---------------------------------------------------------------------
# 2. Extract entities
# ---------------------------------------------------------------------

def extract_entities(news):
    """
    Ask GPT-5 to extract and normalize people, organizations, and prizes
    from the entire dataset.
    Returns:
        entities_dict = {
            "people": [...],
            "organizations": [...],
            "prize": [...]
        }
    """

    entities_list = []
    for item in news:
        # concat limited sample of texts for prompt (avoid overly long input)
        joined = "\n\n".join([
            f"- Title: {item.get('title','')}\n  Text: {item.get('text','')}"
        ])
        user = (
            f"""
            Extract and normalize Nobel-related named entities from the following articles. 
            Return a JSON array where each element has fields: 'people', 'organizations', 'prize'.
            Each entry should contain unique canonical names.
            Ensure the output is strictly valid JSON\n\n
            "Articles:\n{joined}
            """
        )
        response = client.responses.create(
            model="gpt-5",
            input=[{"role": "user", "content": user}],
            # text_format=Entities
        )
        result = json.loads(response.output_text)
        entities_list.extend(result)

    return entities_list


if __name__ == '__main__':
    # Extract information
    with open("cleanned_news.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    # Extract timeline
    timeline = extract_timeline(data)
    # save results
    json.dump(timeline, open("timeline.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    # Extract entities
    entities = extract_entities(data)
    # save results
    json.dump(entities, open("entities.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)