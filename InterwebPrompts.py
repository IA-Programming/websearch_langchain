from datetime import datetime
from langchain.output_parsers.regex import RegexParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.prompts import PromptTemplate

# ====== SEARCH CHAIN PROMPT ======
# Here goes the prompt to make the search queries.

search_prompt_tmpl: str="""The output should be a numbered list of questions and each \
should have a question mark at the end: {question}


Here you have some tips that can help you to write better questions:

1.  **Quoted searches**: Use quotation marks to search for exact phrases, such as ""John Quincy Adams" to get results that contain that exact name or phrase, take into consideration that you cannot end your query with '"' character if you must do it you must add another extra one to the end of your query.
2.  **Minus operator**: Use the minus sign (-) to exclude specific words or phrases from your search results, such as "peanut butter cookies -peanut" to exclude pages that mention peanuts.
3.  **Site search**: Use the "site:" operator to search within a specific website, such as "site:github.com Dave PL" to find a user profile on GitHub.
4.  **Plus operator**: Use the plus sign (+) to ensure that a specific word or phrase is included in your search results, such as "site:amazon.com +24v LED strip" to find pages that include the phrase "24v LED strip".
5.  **Or operator**: Use the "or" operator to search for pages that contain either of two or more words or phrases, such as "Saskatchewan Melford or Humboldt" to find pages that include either "Melford" or "Humboldt".
6.  **Parentheses**: Use parentheses to group words or phrases and clarify your search intent, such as "(vanilla or chocolate) cake" to find pages that contain either "vanilla cake" or "chocolate cake".
7.  **And operator**: Use the "and" operator to ensure that two or more words or phrases are included in your search results, such as "-Saskatchewan and (Humboldt or Melford)" to find pages that do not include "Saskatchewan" but do include either "Humboldt" or "Melford".
8.  **Searching from the URL bar**: Use the URL bar to enter searches directly, and use the "site:" operator to search within a specific website.
9.  **Domain searching**: Use the "site:" operator to search within a specific website, such as "site:amazon.com" to search only within Amazon.
10.  **URL searches**: Use the "inurl:" operator to search for pages that contain a specific URL, such as "site:amazon.com inurl:ID" to find pages on Amazon that contain a specific ID.
11.  **File type searches**: Use the "filetype:" operator to search for files of a specific type, such as "filetype:pdf 1999 secret budget" to find PDF files related to secret budgets from 1999.
12.  **Wild cards**: Use the asterisk (\*) as a wild card to match any word or phrase, such as "the quick brown \* jumps over the \* dog" to find pages that contain a variation of that phrase.
13. **Date format searches**: Use quotation marks and wildcards to search for specific date formats. For example, to search for files containing dates in the MM/DD/YY format, you can use a search query like this: "??/??/??" (where ? represents any single digit). This will match patterns like 01/15/23, 12/31/99, etc.

Some additional,bonus tips to consider are:

-   Using the "in title:" operator to search only within the title of a document
-   Using the "text:" operator to search only within the text of a document
-   You cannot start a query with a number, if you must you have to add before the number a '_' character to your query

"""

system_prompt_tmpl: str = """You are an assistant tasked with improving Google search results. \
Generate THREE Google search queries that are similar to the user's question. \
You will have access to the history of previous interactions and your own answers to it.
For you able to access current events you will know which it's the current date. {date}"""

Search_ChatPrompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_tmpl),
    MessagesPlaceholder("memory"),
    ("human", search_prompt_tmpl)
])

# ====== STUFF AND MAP-REDUCE CHAIN PROMPTS ======
# The common used prompt is the combine prompt for stuff and map-reduce chain and the rest is the others.
system_map_prompt_tmpl = """You're a helpful question/answer assistant, You are aware that {date}"""

question_prompt_template = """Use the following portion of a long chunk of text extracted from a website to see if any of the text is relevant to answer the question. 
Return any relevant text verbatim.
{context}
Question: {question}
Relevant text, if any:"""
QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", system_map_prompt_tmpl),
    ("human", question_prompt_template)
])

combine_prompt_template = """You're a helpful assistant that has access to the converstaion history and knows that {date}.
Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.

QUESTION: What is the impact of climate change on polar bears?
=========
Content: Polar bears rely heavily on sea ice as a platform for hunting seals, their primary food source. With climate change causing the Arctic to warm twice as fast as the rest of the world, sea ice is melting earlier in the spring and forming later in the fall. This shortened hunting season has resulted in reduced fat reserves, which are critical for survival during the ice-free months. As a result, polar bear populations are experiencing declining birth rates and increasing mortality rates due to starvation and malnutrition.
Source: [Climate Change and Arctic Wildlife: Effects on Polar Bears](https://www.arcticwildlife.org/climate-change-effects-on-polar-bears)
Content: A significant effect of climate change on polar bears is their increased energy expenditure. As sea ice disappears, polar bears are forced to travel greater distances to find food. Studies have shown that polar bears are burning more calories than they can consume, leading to weight loss and reduced overall fitness. This not only affects their ability to reproduce but also makes them more vulnerable to disease and death.
Source: [Polar Bear Research and Conservation Studies](https://www.polarbearconservation.org/research-studies)
Content: Another impact of climate change on polar bears is the increasing encroachment of humans into their habitats. As sea ice melts, polar bears are spending more time on land, where they come into contact with human settlements. This has led to a rise in polar bear-human conflicts, as the bears search for food in areas occupied by people. These encounters often end in tragedy for the polar bears, as they are considered dangerous and may be killed.
Source: [Human-Wildlife Conflict in the Arctic](https://www.arcticconflicts.org/human-wildlife-polar-bears)
=========
FINAL ANSWER: Polar bears are increasingly impacted by climate change as rising global temperatures cause sea ice to melt, which is crucial for their hunting and survival. The decrease in sea ice limits their access to prey, primarily seals, leading to malnutrition and reduced reproductive success [`1`](https://www.arcticwildlife.org/climate-change-effects-on-polar-bears). In addition, as polar bears are forced to travel greater distances for food, their energy expenditure increases, further threatening their health and population numbers [`2`](https://www.polarbearconservation.org/research-studies). Climate change also affects polar bear habitats, pushing them closer to human settlements, increasing the likelihood of human-wildlife conflict [`3`](https://www.arcticconflicts.org/human-wildlife-polar-bears).
SOURCES:
[Climate Change and Arctic Wildlife: Effects on Polar Bears](https://www.arcticwildlife.org/climate-change-effects-on-polar-bears)
[Polar Bear Research and Conservation Studies](https://www.polarbearconservation.org/research-studies)
[Human-Wildlife Conflict in the Arctic](https://www.arcticconflicts.org/human-wildlife-polar-bears)

QUESTION: What did the president say about Michael Jackson?
=========
Content: Exercise stimulates the production of endorphins, which are natural mood elevators. Regular physical activity has been shown to decrease feelings of anxiety and depression by enhancing brain function and lowering stress hormones. These endorphins act as natural painkillers and help induce feelings of relaxation and well-being, making exercise an effective, non-pharmacological treatment for mental health issues.
Source: [Mental Health Benefits of Physical Activity](https://www.healthpsychology.org/mental-health-benefits-exercise)
Content: Physical activity also helps improve sleep, which is often disrupted in individuals suffering from mental health conditions. Research shows that people who engage in regular exercise tend to fall asleep faster and enjoy deeper sleep cycles, which in turn enhances cognitive function and emotional regulation. The improved sleep patterns associated with regular physical activity play a crucial role in maintaining mental health.
Source: [How Exercise Improves Sleep and Mental Health](https://www.sleepfoundation.org/exercise-mental-health)
Content: Exercise is also associated with improved self-esteem and confidence. People who regularly engage in physical activities tend to develop a more positive body image and feel a sense of accomplishment, which enhances their overall mental well-being. Additionally, the structured nature of regular physical activity provides a coping mechanism for dealing with stress and anxiety.
Source: [The Psychological Benefits of Exercise](https://www.psychologytoday.com/exercise-psychological-benefits)
Content: More support for patients and families. To get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. It’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  ARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more. A unity agenda for the nation. We can do this. My fellow Americans—tonight , we have gathered in a sacred space—the citadel of our democracy. In this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done great things. We have fought for freedom, expanded liberty, defeated totalitarianism and terror. And built the strongest, freest, and most prosperous nation the world has ever known. Now is the hour. Our moment of responsibility. Our test of resolve and conscience, of history itself. It is in this moment that our character is formed. Our purpose is found. Our future is forged. Well I know this nation.
Source: [The New York Times: President's Call for Unity and Scientific Innovation](https://www.nytimes.com/2022/03/01/us/politics/biden-state-of-the-union-address.html)
=========
FINAL ANSWER: Exercise is widely recognized as beneficial for mental health due to its ability to reduce symptoms of anxiety, depression, and stress. Physical activity promotes the release of endorphins, chemicals in the brain that elevate mood and create a sense of well-being [`1`](https://www.healthpsychology.org/mental-health-benefits-exercise). Additionally, regular exercise can improve sleep patterns, which are often disrupted in people with mental health conditions, leading to better cognitive function and emotional regulation [`2`](https://www.sleepfoundation.org/exercise-mental-health). It has also been shown that engaging in consistent physical activity enhances self-esteem and can provide a structured routine, helping individuals cope with daily stressors more effectively [`3`](https://www.psychologytoday.com/exercise-psychological-benefits).
SOURCES:
[Mental Health Benefits of Physical Activity](https://www.healthpsychology.org/mental-health-benefits-exercise)
[How Exercise Improves Sleep and Mental Health](https://www.sleepfoundation.org/exercise-mental-health)
[The Psychological Benefits of Exercise](https://www.psychologytoday.com/exercise-psychological-benefits)
"""
human_tmpl = """
QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""
COMBINE_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ('system', combine_prompt_template),
    MessagesPlaceholder("chat_history"),
    ('human', human_tmpl)
])

# ====== PROMPT DOCUMENTS ======
# This prompt works to process all the documents recieved from the retriever

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: [{title}]({source})",
    input_variables=["page_content", "source", "title"],
)

# ====== REFINE CHAIN PROMPTS ======
# Here goes the prompts for the refine chain.

system_tmpl = """You're an helpful assistant that's being task with help to the user to answer his internet web search using context retrieved from the internet, you will have message history to keep in track, And you know that {date}"""

DEFAULT_REFINE_PROMPT_TMPL = (
    "The original question is as follows: {question}\n"
    "We have provided an existing answer, including sources: {existing_answer}\n"
    "We have the opportunity to refine the existing answer"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_str}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question. "
    "If you do update it, please update the sources as well. "
    "If the context isn't useful, return the original answer."
)
DEFAULT_REFINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", system_tmpl),
    MessagesPlaceholder("chat_history"),
    ("human", DEFAULT_REFINE_PROMPT_TMPL)
])


DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {question}\n"
)
DEFAULT_TEXT_QA_PROMPT = ChatPromptTemplate.from_messages([
    ('system', system_tmpl),
    MessagesPlaceholder("chat_history"),
    ("human", DEFAULT_TEXT_QA_PROMPT_TMPL)
])

# ====== MAP-RERANK-PROMPT ======
# Here goes the custom prompt for this map-rerank chain.

output_parser = RegexParser(
    regex=r"(.*?)\nScore: (\d*)",
    output_keys=["answer", "score"],
)

system_maprerank_prompt_tmpl = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

In addition to giving an answer, also return a score of how fully it answered the user's question. This should be in the following format:

Question: [question here]
Helpful Answer: [answer here]
Score: [score between 0 and 100]

How to determine the score:
- Higher is a better answer
- Better responds fully to the asked question, with sufficient level of detail
- If you do not know the answer based on the context, that should be a score of 0
- Don't be overconfident!

Example #1

Context:
---------
Apples are red
---------
Question: what color are apples?
Helpful Answer: red
Score: 100

Example #2

Context:
---------
it was night and the witness forgot his glasses. he was not sure if it was a sports car or an suv
---------
Question: what type was the car?
Helpful Answer: a sports car or an suv
Score: 60

Example #3

Context:
---------
Pears are either red or orange
---------
Question: what color are apples?
Helpful Answer: This document does not answer the question
Score: 0

Begin!
"""

maprerank_prompt_template = """
Context:
---------
{context}
---------
Question: {question}
Helpful Answer:"""

maprerank_chatprompt = ChatPromptTemplate.from_messages([
    ("system", "You're a fully capable assistant, to follow instructions, and is able to keep track of the current conversation. You're time aware knowing that {date}"),
    MessagesPlaceholder("chat_history"),
    ("human", system_maprerank_prompt_tmpl+maprerank_prompt_template)
])

maprerank_chatprompt.output_parser = output_parser

# ====== DATE UTILS FUNCTIONS ======
# Here goes functions that helps you to populate some partial variables.

def get_current_date():
    """
    Returns the current date in the format 'Today is <day of the week> <day> <month> of <year>.'

    Returns:
        str: The current date in the desired format.
    """
    now = datetime.now()
    day_of_week = now.strftime("%A").lower()
    day = now.day
    month = now.strftime("%B").lower()
    year = now.year

    # Convert day of the week and month to title case
    day_of_week = day_of_week.capitalize()
    month = month.capitalize()

    # Format the date string
    date_string = f"Today is {day_of_week} {day:02d} {month} of {year}."

    return date_string

MAP_RERANK_PROMPT = maprerank_chatprompt.partial(date=get_current_date())