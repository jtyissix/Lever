EXTRACT_TAG_PROMPT = """
- Target task -
You are an intelligent assistant that helps a human analyst determine which domains a text document belongs to.

- Goal -
Given a text document, summarize all domains that best describe the document.

- Requirements -
1. The summary must contain individual words only.
2. The summary should be concise and reveal the parent domain and topic.
3. Use one word if possible; use up to five words if needed.
4. Format the answer as (<domain_1>{tuple_delimiter}<domain_2>{tuple_delimiter}).
5. Return the output in English.

- Examples -
Example 1:
Input text document: **biological efficiency:** the percentage measurement of the yield of fresh mushrooms from the dry weight of the substrate.
Output: (Biology{tuple_delimiter}Agriculture{tuple_delimiter}Farming{tuple_delimiter})

Example 2:
Input text document: **Creamed Leek and Potato Soup** Pass the soup through a food mill. Thin with milk or water if needed, then add cream. Serve hot or chilled with finely chopped chives and parsley or chervil.
Output: (Cooking{tuple_delimiter})

Example 3:
Input text document: Title Company Indemnity, provided that the Purchaser determines during the Due Diligence Period that the Title Company refuses to delete the standard printed exception for liens as part of extended coverage.
Output: (Legal{tuple_delimiter}Law{tuple_delimiter})
"""


