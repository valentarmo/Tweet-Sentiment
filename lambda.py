import os
import time
import json
import boto3
import requests
import traceback

TWITTER_TOKEN = os.environ['TWITTER_TOKEN']
TWITTER_FILTER_API_URL = os.environ['TWITTER_FILTER_API_URL']
TWITTER_FILTER_API_RULES_URL = os.environ['TWITTER_FILTER_API_RULES_URL']
MAX_TWEETS_ALLOWED = int(os.environ['MAX_TWEETS_ALLOWED'])


def lambda_handler(event, context):
    try:
        filter_statement = event['filter']
        auth_headers = {'Authorization': f'Bearer {TWITTER_TOKEN}'}
        set_filter(TWITTER_FILTER_API_RULES_URL, auth_headers, filter_statement)
        tweets = get_tweets(TWITTER_FILTER_API_URL, auth_headers, MAX_TWEETS_ALLOWED)
        comprehend = boto3.client('comprehend')
        languages = map_languages(comprehend, tweets)
        return {
            'Success': True,
            'Sentiments': detect_sentiment(comprehend, tweets, languages)
        }
    except Exception as e:
        traceback.print_exc()
        return {
            'Success': False,
            'Sentiments': []
        }


def set_filter(url, headers, rule):
    current_rules = get_current_rules(url, headers)
    if current_rules:
        remove_rules(url, headers, current_rules)
    add_rule(url, headers, make_filter_rule(rule))


def get_current_rules(url, headers):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    response_body = response.json()
    return response_body['data'] if 'data' in response_body else None


def make_filter_rule(rule):
    return [{'value': rule, 'tag': rule}]


def remove_rules(url, headers, rules):
    rule_ids = list(map(lambda rule: rule['id'], rules))
    response = requests.post(url, headers=headers, json={'delete': {'ids': rule_ids}})
    response.raise_for_status()


def add_rule(url, headers, rule):
    response = requests.post(url, headers=headers, json={'add': rule})
    response.raise_for_status()


def get_tweets(url, headers, max_tweets, timeout=10):
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()

    tweets = []
    start_time = time.time()
    number_of_tweets = 0
    for line in response.iter_lines():
        json_response = json.loads(line)
        tweets += [json_response['data']['text']]
        number_of_tweets += 1
        if number_of_tweets == max_tweets or time.time() - start_time >= timeout:
            response.close()
            break

    return tweets


def map_languages(comprehend_client, tweets):
    result_list = comprehend_client.batch_detect_dominant_language(TextList=tweets)['ResultList']
    return list(map(lambda result: result['Languages'][0]['LanguageCode'], result_list))


def detect_sentiment(comprehend_client, tweets, languages):
    if len(tweets) != len(languages):
        raise ValueError('Tweets and Languages differ in length')

    sentiments = []
    for i in range(len(tweets)):
        tweet = tweets[i]
        language = languages[i]
        sentiment = comprehend_client.detect_sentiment(Text=tweet, LanguageCode=language)['Sentiment']
        sentiments += [(tweet, sentiment)]

    return sentiments
