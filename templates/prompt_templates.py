prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You're a financial expert and have been tasked with analysing news about the crpyto market from several reputable publications.
Based on the information provided, answer the user's questions based on the information you gained from analysing the following article.
The below information will start with the published date followed by the article. The aim of providing the publishing date is to give you
a temporal dimension of what event is occuring when. This will help you understand how market sentiments are changing over time.
Additionally, you have also been provided with tabular data of opening and closing prices of bitcoin for additional context.


### Published Date:
{}

### News Title:
{}

### News Article:
{}

### Question:
{}

### Data:
{}

### Response:
{}"""

question = """Based on the news articles and past 7 day price of bitcoin you have analysed so far,
give financial advice to a trading novice interested in investing in Bitcoin. Also predict the general 
market sentiement for the next week and explain why you think so.
Format your response into JSON. The keys will be thought process, advice and market sentiment."""
