# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import traceback
import time
import bittensor as bt
import random
import asyncio
import re
import html
import random
from typing import List
from .config import RewardModelType
from .reward import BaseRewardModel, BaseRewardEvent, pattern_to_check
from neurons.validators.utils.prompts import (
    LinkContentPrompt,
)
from datura.protocol import (
    ScraperStreamingSynapse,
    TwitterScraperTweet,
)
from neurons.validators.apify.twitter_scraper_actor import TwitterScraperActor
from datura.services.twitter_api_wrapper import TwitterAPIClient
from neurons.validators.reward.reward_llm import RewardLLM
from neurons.validators.utils.prompts import LinkContentPrompt
from datura.utils import clean_text
import json
from datetime import datetime
import pytz
from pydantic import ValidationError
from datetime import datetime
import pytz

APIFY_LINK_SCRAPE_AMOUNT = 5


class TwitterContentRelevanceModel(BaseRewardModel):
    reward_model_name: str = "VMware/open-llama-7b-open-instruct"

    @property
    def name(self) -> str:
        return RewardModelType.twitter_content_relevance.value

    def __init__(self, device: str, scoring_type: None, llm_reward: RewardLLM):
        super().__init__()
        self.device = device
        self.reward_llm = llm_reward

        self.scoring_type = scoring_type
        self.tw_client = TwitterAPIClient()

    def clean_text(self, text):
        return clean_text(text)

    async def llm_process_validator_tweets(self, response: ScraperStreamingSynapse):
        if not response.validator_tweets:
            return {}

        start_llm_time = time.time()
        scoring_messages = []
        for validator_tweet in response.validator_tweets:
            val_text = validator_tweet.text
            val_tweet_id = validator_tweet.id
            result = self.get_scoring_text(
                prompt=response.prompt, content=val_text, response=None
            )
            if result:
                _, scoring_text = result
                scoring_messages.append({str(val_tweet_id): scoring_text})
        score_responses = await self.reward_llm.llm_processing(scoring_messages)

        end_llm_time = time.time()
        llm_duration_minutes = (end_llm_time - start_llm_time) / 60
        bt.logging.info(
            f"TwitterContentRelevanceModel LLM process validator tweets took {llm_duration_minutes} minutes."
        )
        return score_responses

    async def scrape_tweets_with_retries(
        self, urls: List[str], group_size: int, max_attempts: int
    ):
        fetched_tweets = []
        non_fetched_links = urls.copy()
        attempt = 1

        while attempt <= max_attempts and non_fetched_links:
            bt.logging.info(
                f"Attempt {attempt}/{max_attempts}, processing {len(non_fetched_links)} links."
            )

            url_groups = [
                non_fetched_links[i : i + group_size]
                for i in range(0, len(non_fetched_links), group_size)
            ]

            tasks = [
                asyncio.create_task(TwitterScraperActor().get_tweets(urls=group))
                for group in url_groups
            ]

            # Wait for tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Combine results and handle exceptions
            for result in results:
                if isinstance(result, Exception):
                    bt.logging.error(
                        f"Error in TwitterScraperActor attempt {attempt}: {str(result)}"
                    )
                    continue
                fetched_tweets.extend(result)

            # Update non_fetched_links
            fetched_tweet_ids = {tweet.id for tweet in fetched_tweets}
            non_fetched_links = [
                link
                for link in non_fetched_links
                if self.tw_client.utils.extract_tweet_id(link) not in fetched_tweet_ids
            ]

            if non_fetched_links:
                bt.logging.info(
                    f"Retrying fetching non-fetched {len(non_fetched_links)} tweets. Retries left: {max_attempts - attempt}"
                )
                await asyncio.sleep(3)

            attempt += 1

        return fetched_tweets, non_fetched_links

    async def process_tweets(self, responses: List[ScraperStreamingSynapse]):
        default_val_score_responses = [{} for _ in responses]

        try:
            non_fetched_links = {}
            start_time = time.time()

            all_links = [
                link
                for response in responses
                if response.completion_links
                for link in random.sample(
                    response.completion_links,
                    min(APIFY_LINK_SCRAPE_AMOUNT, len(response.completion_links)),
                )
            ]
            unique_links = list(set(all_links))
            if len(unique_links) == 0:
                bt.logging.info("No unique links found to process.")
                return default_val_score_responses

            bt.logging.info(f"Fetching {len(unique_links)} unique Twitter links.")

            tweets_list, non_fetched_links = await self.scrape_tweets_with_retries(
                unique_links, group_size=200, max_attempts=4
            )

            for response in responses:
                ids = [
                    self.tw_client.utils.extract_tweet_id(link)
                    for link in response.completion_links
                ]

                for tweet in tweets_list:
                    if tweet.id in ids:
                        response.validator_tweets.append(tweet)
            if len(unique_links) == 0:
                bt.logging.info("No unique links found to process.")
                return default_val_score_responses

            end_time = time.time()
            bt.logging.info(
                f"Fetched Twitter links method took {end_time - start_time} seconds. "
                f"All links count: {len(all_links)}, Unique links count: {len(unique_links)}, "
                f"APIFY fetched tweets links count: {len(tweets_list)}"
            )

            bt.logging.info(
                f"Twitter Links not fetched Amount: {len(non_fetched_links)}; List: {non_fetched_links}"
            )
            if len(non_fetched_links):
                bt.logging.info(
                    f"Unique Twitter Links Amount: {len(unique_links)}; List: {unique_links};"
                )

            val_score_responses_list = await self.process_response_items_in_batches(
                responses=responses,
                batch_size=20,
                process_function=self.llm_process_validator_tweets,
            )

            return val_score_responses_list
        except Exception as e:
            bt.logging.error(f"Error in process_tweets: {str(e)}")
            return default_val_score_responses

    def format_text_for_match(self, text):
        # Unescape HTML entities first
        text = html.unescape(text)
        # url shorteners can cause problems with tweet verification, so remove urls from the text comparison.
        text = re.sub(r"(https?://)?\S+\.\S+\/?(\S+)?", "", text)
        # Some scrapers put the mentions at the front of the text, remove them.
        text = re.sub(r"^(@\w+\s*)+", "", text)
        # And some trim trailing whitespace at the end of newlines, so ignore whitespace.
        text = re.sub(r"\s+", "", text)
        # The validator apify actor uses the tweet.text field and not the note_tweet field (> 280) charts, so only
        # use the first 280 chars for comparison.
        text = text[:280]
        return text

    def check_tweet_content(self, response: ScraperStreamingSynapse):
        try:
            tweet_score = 0

            completion = self.get_successful_twitter_completion(response=response)
            if not completion:
                return 0

            tweets_data = response.miner_tweets
            # miner_tweets_meta = miner_tweets.get('meta', {})
            tweets_amount = len(tweets_data)

            # Assign completion links and validator tweets from the response
            completion_links = response.completion_links

            if (
                not completion_links
                or len(completion_links) < 2
                or tweets_amount == 0
                or not response.validator_tweets
            ):
                # Ensure there are at least two twitter links provided by miners and check for the presence of miner and validator tweets
                return 0

            # Initialize a list to hold scores for each validator tweet
            tweet_scores = []
            # Iterate over all validator tweets instead of selecting a random one
            for val_tweet in response.validator_tweets:
                # Extract content, ID, and creation time of the validator tweet
                val_tweet_content = val_tweet.text
                val_tweet_id = val_tweet.id
                val_tweet_created_at = val_tweet.created_at

                # Find the corresponding miner tweet by ID
                tweet = next(
                    (tweet for tweet in tweets_data if tweet["id"] == val_tweet_id),
                    None,
                )

                # Initialize the score for this iteration
                tweet_score = 0

                if tweet:
                    if not self.is_valid_tweet(tweet):
                        tweet_scores.append(0)
                        continue

                    tweet_text = tweet["text"]

                    if not tweet_text or re.search(
                        pattern_to_check, tweet_text, flags=re.IGNORECASE
                    ):
                        tweet_scores.append(0)
                        continue

                    # Prepare texts for comparison by normalizing them
                    miner_text_compared = self.format_text_for_match(tweet_text)
                    validator_text_compared = self.format_text_for_match(
                        val_tweet_content
                    )

                    if miner_text_compared == validator_text_compared:
                        tweet_score = 1
                    else:
                        tweet_score = 0

                    converted_val_tweet_created_at = (
                        datetime.strptime(
                            val_tweet_created_at, "%a %b %d %H:%M:%S %z %Y"
                        )
                        .astimezone(pytz.UTC)
                        .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                        + "Z"
                    )

                    if not tweet.get("created_at") == converted_val_tweet_created_at:
                        tweet_score = 0

                    tweet_created_at_aware = datetime.strptime(
                        converted_val_tweet_created_at, "%Y-%m-%dT%H:%M:%S.%fZ"
                    ).replace(tzinfo=pytz.UTC, second=0, microsecond=0)

                    start_date = response.start_date
                    end_date = response.end_date

                    start_date = datetime.strptime(
                        start_date, "%Y-%m-%dT%H:%M:%SZ"
                    ).replace(tzinfo=pytz.utc)
                    end_date = datetime.strptime(
                        end_date, "%Y-%m-%dT%H:%M:%SZ"
                    ).replace(tzinfo=pytz.utc)

                    if (
                        tweet_created_at_aware < start_date
                        or tweet_created_at_aware > end_date
                    ):
                        tweet_score = 0

                tweet_scores.append(tweet_score)

            if tweet_scores:
                # Calculate the average score
                average_score = sum(tweet_scores) / len(tweet_scores)
            else:
                # If there are no scores, set average score to 0
                average_score = 0

            return average_score
        except Exception as e:
            bt.logging.error(f"check_tweet_content: {str(e)}")
            return 0

    def is_valid_tweet(self, tweet):
        try:
            _ = TwitterScraperTweet(**tweet)
        except ValidationError as e:
            bt.logging.error(f"Invalid miner tweet data: {e}")
            return False
        return True

    def get_scoring_text(
        self, prompt: str, content: str, response: bt.Synapse
    ) -> BaseRewardEvent:
        try:
            if response:
                completion = self.get_successful_twitter_completion(response=response)
                if not completion:
                    return None

            scoring_prompt = None

            scoring_prompt_text = None

            scoring_prompt = LinkContentPrompt()

            if content is None:
                bt.logging.debug("Twitter Content is empty")
                return None

            content = self.clean_text(content)

            scoring_prompt_text = scoring_prompt.text(prompt, content)

            return scoring_prompt, [
                {"role": "system", "content": scoring_prompt.get_system_message()},
                {"role": "user", "content": scoring_prompt_text},
            ]
        except Exception as e:
            error_message = f"Error in Prompt reward method: {str(e)}"
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            bt.logging.warning("\n".join(tb_str) + error_message)
            return None

    async def get_rewards(
        self, responses: List[bt.Synapse], uids
    ) -> List[BaseRewardEvent]:
        try:
            # completions: List[str] = self.get_successful_twitter_completions(responses)
            # bt.logging.debug(
            #     f"TwitterContentRelevanceModel | Calculating {len(completions)} rewards (typically < 1 sec/reward)."
            # )

            val_score_responses_list = await self.process_tweets(responses=responses)
            bt.logging.info(f"VAL_SCORE_RESPONSES: {val_score_responses_list}")

            scores = [self.check_tweet_content(response) for response in responses]

            reward_events = []
            scoring_prompt = LinkContentPrompt()

            grouped_val_score_responses = []

            for apify_score, response, val_score_responses, uid_tensor in zip(
                scores,
                responses,
                val_score_responses_list,
                uids,
            ):
                uid = uid_tensor.item()
                reward_event = BaseRewardEvent()
                reward_event.reward = 0
                if "Twitter Search" not in response.tools:
                    if response.completion_links:
                        reward_event.reward = 0
                    else:
                        reward_event.reward = 1
                    reward_events.append(reward_event)
                    grouped_val_score_responses.append({})
                    continue

                score_result = None
                response_scores = {}
                total_score = 0
                max_links_considered = (
                    len(response.validator_tweets)
                    if len(response.validator_tweets) > 10
                    else 10
                )

                unique_tweet_texts = {}
                for val_tweet in response.validator_tweets:
                    text = self.format_text_for_match(val_tweet.text)
                    if text not in unique_tweet_texts:
                        unique_tweet_texts[text] = val_tweet

                unique_validator_tweets = list(unique_tweet_texts.values())

                duplicate_tweets_count = len(response.validator_tweets) - len(
                    unique_validator_tweets
                )

                response.validator_tweets = unique_validator_tweets

                if len(response.validator_tweets):
                    for val_tweet in response.validator_tweets:
                        val_tweet_id = val_tweet.id
                        if val_score_responses:
                            score_result = val_score_responses.get(
                                str(val_tweet_id), None
                            )
                            if score_result is not None:
                                score = scoring_prompt.extract_score(score_result)
                                total_score += score / 10.0
                                response_scores[val_tweet_id] = score
                    if total_score > 0:
                        average_score = (
                            total_score / max_links_considered * apify_score
                        )  # len(response.validator_tweets)
                        reward_event.reward = self.calculate_adjusted_score(
                            links_count=len(response.completion_links),
                            score=average_score,
                            duplicate_tweets_count=duplicate_tweets_count,
                        )
                else:
                    bt.logging.info(f"UID '{uid}' has no validator tweets.")
                    reward_event.reward = 0  # Handle case with no validator tweets
                reward_events.append(reward_event)
                grouped_val_score_responses.append(response_scores)

            zero_scores = {}
            non_zero_scores = {}

            for (index, response), uid_tensor, reward_e in zip(
                enumerate(responses), uids, reward_events
            ):
                uid = uid_tensor.item()
                if reward_e.reward == 0:
                    zero_scores[uid] = reward_e.reward
                else:
                    non_zero_scores[uid] = reward_e.reward

            bt.logging.info(
                f"==================================Twitter Links Content scoring Zero Scores  ({len(zero_scores)} cases)=================================="
            )
            bt.logging.info(json.dumps(zero_scores))
            bt.logging.info(
                f"==================================Twitter Links Content scoring Non-Zero Scores ({len(non_zero_scores)} cases)=================================="
            )
            bt.logging.info(json.dumps(non_zero_scores))
            return reward_events, grouped_val_score_responses
        except Exception as e:
            error_message = f"Link Content Relevance get_rewards: {str(e)}"
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            bt.logging.error("\n".join(tb_str) + error_message)
            reward_events = []
            for response in responses:
                reward_event = BaseRewardEvent()
                reward_event.reward = 0
                reward_events.append(reward_event)
            return reward_events, {}
