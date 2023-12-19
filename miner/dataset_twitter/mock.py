import json
import re

# Path to your JSON file
file_name = './miner/dataset_twitter/dataset_twitter.json'

async def mock_filter_tweets(query_string):
    """
    Filters tweets from a JSON file based on given query strings.

    :param file_name: Name of the JSON file containing tweet data.
    :param query_strings: List of query strings to filter tweets.
    :return: List of filtered tweets.
    """
    with open(file_name, 'r') as file:
        tweets = json.load(file)

    filtered_tweets = []

    for tweet in tweets:
        if 'text' in tweet:
            if check_match(tweet['text'],  query_string):
                filtered_tweets.append(tweet)
                break  # Avoid adding the same tweet multiple times
    return filtered_tweets

def check_match(tweet_query, filter_query):
    """
    Checks if the tweet query matches the filter query.

    :param tweet_query: The query string from the tweet.
    :param filter_query: The query string to filter by.
    :return: True if it matches, False otherwise.
    """
    if '"' in filter_query:
        # Handle phrase or keyword queries
        phrases = re.findall(r'"(.*?)"', filter_query)
        keywords = re.sub(r'"(.*?)"', '', filter_query).split()
        for phrase in phrases:
            if phrase in tweet_query:
                return True
        for keyword in keywords:
            if keyword in tweet_query:
                return True
    elif 'OR' in filter_query:
        # Handle logical operator queries
        parts = filter_query.split(' OR ')
        return any(part in tweet_query for part in parts)
    elif 'since:' in filter_query or 'until:' in filter_query:
        # Handle queries with additional parameters (this part is more complex and needs specific implementation based on your date format and logic)
        # As an example, we'll just check for the presence of "climate change"
        return "climate change" in tweet_query
    else:
        # Handle simple text queries
        return filter_query in tweet_query

    return False



# # Define your query strings
# query_strings = [
#     "\"latest trends\" \"artificial intelligence\" OR \"AI\"",
#     "\"bla bla bla\"",
#     "UN summit reactions OR UN conference feedback",
#     "\"climate change\" since:2023-02-01 until:2023-03-01 -filter:retweets"
#     "elon"
# ]


# # Call the method and get filtered tweets

# # Print or process the filtered tweets

# for query in ['#Elonmusk']:
#     filtered_tweets = mock_filter_tweets(query)
#     print(filtered_tweets)
