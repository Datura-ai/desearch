import os
from typing import List, Optional

# DEALINGS IN THE SOFTWARE.p
import traceback
import bittensor as bt
from apify_client import ApifyClientAsync
from datura.protocol import (
    TwitterScraperTweet,
    TwitterScraperMedia,
    TwitterScraperUser,
)


APIFY_API_KEY = os.environ.get("APIFY_API_KEY")

# todo at ths moment just warning, later it will be required
if not APIFY_API_KEY:
    raise ValueError(
        "Please set the APIFY_API_KEY environment variable. See here: https://github.com/surcyf123/smart-scrape/blob/main/docs/env_variables.md"
    )


class TwitterScraperActor:
    def __init__(self) -> None:
        # Actor: https://apify.com/apidojo/tweet-scraper
        self.actor_id = "61RPP7dywgiy0JPD0"
        self.user_scraper_actor_id = "V38PZzpEgOfeeWvZY"
        self.client = ApifyClientAsync(token=APIFY_API_KEY)

    async def get_tweets(
        self, urls: List[str], add_user_info: bool = True
    ) -> List[TwitterScraperTweet]:
        if not APIFY_API_KEY:
            bt.logging.warning(
                "Please set the APIFY_API_KEY environment variable. See here: https://github.com/surcyf123/smart-scrape/blob/main/docs/env_variables.md. This will be required in the next release."
            )
            return []
        try:
            run_input = {
                "startUrls": urls,
            }

            run = await self.client.actor(self.actor_id).call(run_input=run_input)

            tweets: List[TwitterScraperTweet] = []

            async for item in self.client.dataset(
                run["defaultDatasetId"]
            ).iterate_items():
                media_list = item.get("extendedEntities", {}).get("media", [])

                media_list = [
                    TwitterScraperMedia(
                        media_url=media.get("media_url_https"), type=media.get("type")
                    )
                    for media in media_list
                ]

                author = item.get("author", {})

                tweet = TwitterScraperTweet(
                    id=item.get("id"),
                    text=item.get("text"),
                    reply_count=item.get("replyCount"),
                    retweet_count=item.get("retweetCount"),
                    like_count=item.get("likeCount"),
                    quote_count=item.get("quoteCount"),
                    url=item.get("url"),
                    created_at=item.get("createdAt"),
                    is_quote_tweet=item.get("isQuote"),
                    is_retweet=item.get("isRetweet"),
                    media=media_list,
                    user=TwitterScraperUser(
                        id=author.get("id"),
                        created_at=author.get("createdAt"),
                        description=author.get("description"),
                        followers_count=author.get("followers"),
                        favourites_count=author.get("favouritesCount"),
                        media_count=author.get("mediaCount"),
                        statuses_count=author.get("statusesCount"),
                        verified=author.get("isVerified"),
                        profile_image_url=author.get("profilePicture"),
                        url=author.get("url"),
                        name=author.get("name"),
                        username=author.get("userName"),
                    ),
                )

                tweets.append(tweet)

            return tweets
        except Exception as e:
            error_message = (
                f"TwitterScraperActor: Failed to scrape tweets {urls}: {str(e)}"
            )
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            bt.logging.warning("\n".join(tb_str) + error_message)
            return []

    async def get_tweets_advanced(
        self,
        urls: Optional[List[str]],
        author: Optional[str] = None,
        conversationIds: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        geocode: Optional[str] = None,
        geotaggedNear: Optional[str] = None,
        inReplyTo: Optional[str] = None,
        includeSearchTerms: Optional[bool] = None,
        maxItems: Optional[int] = None,
        mentioning: Optional[str] = None,
        minimumFavorites: Optional[str] = None,
        minimumReplies: Optional[str] = None,
        minimumRetweets: Optional[str] = None,
        onlyImage: Optional[bool] = None,
        onlyQuote: Optional[bool] = None,
        onlyTwitterBlue: Optional[bool] = None,
        onlyVerifiedUsers: Optional[bool] = None,
        onlyVideo: Optional[bool] = None,
        placeObjectId: Optional[str] = None,
        searchTerms: Optional[List[str]] = None,
        sort: Optional[str] = None,
        tweetLanguage: Optional[str] = None,
        twitterHandles: Optional[List[str]] = None,
        withinRadius: Optional[str] = None,
    ) -> dict:
        if not APIFY_API_KEY:
            error = "Please set the APIFY_API_KEY environment variable. See here: https://github.com/surcyf123/smart-scrape/blob/main/docs/env_variables.md. This will be required in the next release."
            bt.logging.warning(error)
            return {"error": error}
        try:
            run_input = {
                "startUrls": urls,
                "author": author,
                "conversationIds": conversationIds,
                "start": start,
                "end": end,
                "geocode": geocode,
                "geotaggedNear": geotaggedNear,
                "inReplyTo": inReplyTo,
                "includeSearchTerms": includeSearchTerms,
                "maxItems": maxItems,
                "mentioning": mentioning,
                "minimumFavorites": minimumFavorites,
                "minimumReplies": minimumReplies,
                "minimumRetweets": minimumRetweets,
                "onlyImage": onlyImage,
                "onlyQuote": onlyQuote,
                "onlyTwitterBlue": onlyTwitterBlue,
                "onlyVerifiedUsers": onlyVerifiedUsers,
                "onlyVideo": onlyVideo,
                "placeObjectId": placeObjectId,
                "searchTerms": searchTerms,
                "sort": sort,
                "tweetLanguage": tweetLanguage,
                "twitterHandles": twitterHandles,
                "withinRadius": withinRadius,
            }
            run_input = {k: v for k, v in run_input.items() if v is not None}

            run = await self.client.actor(self.actor_id).call(run_input=run_input)

            tweets: List[dict] = []

            async for item in self.client.dataset(
                run["defaultDatasetId"]
            ).iterate_items():
                media_list = item.get("extendedEntities", {}).get("media", [])

                media_list = [
                    TwitterScraperMedia(
                        media_url=media.get("media_url_https"), type=media.get("type")
                    )
                    for media in media_list
                ]

                author = item.get("author", {})

                tweet = TwitterScraperTweet(
                    id=item.get("id"),
                    text=item.get("text"),
                    reply_count=item.get("replyCount"),
                    retweet_count=item.get("retweetCount"),
                    like_count=item.get("likeCount"),
                    quote_count=item.get("quoteCount"),
                    impression_count=item.get("viewCount"),
                    bookmark_count=item.get("bookmarkCount"),
                    url=item.get("url"),
                    created_at=item.get("createdAt"),
                    is_quote_tweet=item.get("isQuote"),
                    is_retweet=item.get("isRetweet"),
                    media=media_list,
                    user=TwitterScraperUser(
                        id=author.get("id"),
                        created_at=author.get("createdAt"),
                        description=author.get("description"),
                        followers_count=author.get("followers"),
                        favourites_count=author.get("favouritesCount"),
                        media_count=author.get("mediaCount"),
                        statuses_count=author.get("statusesCount"),
                        verified=author.get("isVerified"),
                        profile_image_url=author.get("profilePicture"),
                        url=author.get("url"),
                        name=author.get("name"),
                        username=author.get("userName"),
                    ),
                )

                tweets.append(tweet.dict())

            return {"data": tweets}
        except Exception as e:
            error_message = (
                f"TwitterScraperActor: Failed to scrape tweets {searchTerms}: {str(e)}"
            )
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            bt.logging.warning("\n".join(tb_str) + error_message)
            return {
                "error": error_message,
            }

    async def get_user_by_id(
        self,
        id: str,
    ) -> dict:
        if not APIFY_API_KEY:
            error = "Please set the APIFY_API_KEY environment variable. See here: https://github.com/surcyf123/smart-scrape/blob/main/docs/env_variables.md. This will be required in the next release."
            bt.logging.warning(error)
            return {"error": error}
        try:
            run_input = {
                "twitterUserIds": [id],
                "maxItems": 1,
                "getFollowing": True,
                "getRetweeters": False,
                "getFollowers": False,
                "includeUnavailableUsers": False,
            }
            run_input = {k: v for k, v in run_input.items() if v is not None}

            run = await self.client.actor(self.user_scraper_actor_id).call(
                run_input=run_input
            )

            user = None
            async for item in self.client.dataset(
                run["defaultDatasetId"]
            ).iterate_items():
                if item.get("id") == id:
                    user = item

            return {"data": user}
        except Exception as e:
            error_message = f"TwitterScraperActor: Failed to scrape user {id}: {str(e)}"
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            bt.logging.warning("\n".join(tb_str) + error_message)
            return {
                "error": error_message,
            }

    async def get_user_by_username(
        self,
        username: str,
    ) -> dict:
        if not APIFY_API_KEY:
            error = "Please set the APIFY_API_KEY environment variable. See here: https://github.com/surcyf123/smart-scrape/blob/main/docs/env_variables.md. This will be required in the next release."
            bt.logging.warning(error)
            return {"error": error}
        try:
            run_input = {
                "twitterHandles": [username],
                "maxItems": 1,
                "getFollowing": True,
                "getRetweeters": False,
                "getFollowers": False,
                "includeUnavailableUsers": False,
            }
            run_input = {k: v for k, v in run_input.items() if v is not None}

            run = await self.client.actor(self.user_scraper_actor_id).call(
                run_input=run_input
            )

            user = None
            async for item in self.client.dataset(
                run["defaultDatasetId"]
            ).iterate_items():
                if item.get("userName") == username:
                    user = item

            return {"data": user}
        except Exception as e:
            error_message = (
                f"TwitterScraperActor: Failed to scrape user {username}: {str(e)}"
            )
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            bt.logging.warning("\n".join(tb_str) + error_message)
            return {
                "error": error_message,
            }

    async def get_user_followings(
        self,
        id: str,
        maxUsersPerQuery: Optional[int] = 10,
    ) -> dict:
        if not APIFY_API_KEY:
            error = "Please set the APIFY_API_KEY environment variable. See here: https://github.com/surcyf123/smart-scrape/blob/main/docs/env_variables.md. This will be required in the next release."
            bt.logging.warning(error)
            return {"error": error}
        try:
            run_input = {
                "twitterUserIds": [id],
                "maxItems": maxUsersPerQuery,
                "getFollowing": True,
                "getRetweeters": False,
                "getFollowers": False,
                "includeUnavailableUsers": False,
            }
            run_input = {k: v for k, v in run_input.items() if v is not None}

            run = await self.client.actor(self.user_scraper_actor_id).call(
                run_input=run_input
            )

            users: List[dict] = []

            async for item in self.client.dataset(
                run["defaultDatasetId"]
            ).iterate_items():
                users.append(item)

            return {"data": users}
        except Exception as e:
            error_message = f"TwitterScraperActor: Failed to scrape user's followings {id}: {str(e)}"
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            bt.logging.warning("\n".join(tb_str) + error_message)
            return {
                "error": error_message,
            }

    async def get_user_followers(
        self,
        id: str,
        maxUsersPerQuery: Optional[int] = 10,
    ) -> dict:
        if not APIFY_API_KEY:
            error = "Please set the APIFY_API_KEY environment variable. See here: https://github.com/surcyf123/smart-scrape/blob/main/docs/env_variables.md. This will be required in the next release."
            bt.logging.warning(error)
            return {"error": error}
        try:
            run_input = {
                "twitterUserIds": [id],
                "maxItems": maxUsersPerQuery,
                "getFollowing": False,
                "getRetweeters": False,
                "getFollowers": True,
                "includeUnavailableUsers": False,
            }
            run_input = {k: v for k, v in run_input.items() if v is not None}

            run = await self.client.actor(self.user_scraper_actor_id).call(
                run_input=run_input
            )

            users: List[dict] = []

            async for item in self.client.dataset(
                run["defaultDatasetId"]
            ).iterate_items():
                users.append(item)

            return {"data": users}
        except Exception as e:
            error_message = f"TwitterScraperActor: Failed to scrape user's followers {id}: {str(e)}"
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            bt.logging.warning("\n".join(tb_str) + error_message)
            return {
                "error": error_message,
            }
