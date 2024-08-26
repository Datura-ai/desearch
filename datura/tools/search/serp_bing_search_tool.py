import os
import json
import bittensor as bt
from typing import Type
from pydantic import BaseModel, Field
from datura.tools.base import BaseTool
from starlette.types import Send
from .serp_api_wrapper import SerpAPIWrapper


SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")

if not SERPAPI_API_KEY:
    raise ValueError(
        "Please set the SERPAPI_API_KEY environment variable. See here: https://github.com/surcyf123/smart-scrape/blob/main/docs/env_variables.md"
    )


class SerpBingSearchSchema(BaseModel):
    query: str = Field(
        ...,
        description="The search query for Bing search.",
    )


class SerpBingSearchTool(BaseTool):
    name = "Bing Search"

    slug = "serp_bing_search"

    description = (
        "This tool performs Bing searches and extracts relevant snippets and webpages. "
        "It's particularly useful for staying updated with current events and finding quick answers to your queries."
    )

    args_schema: Type[SerpBingSearchSchema] = SerpBingSearchSchema

    tool_id = "88be4eef-6d3c-4eaa-b7a5-a30dda650c14"

    def _run():
        pass

    async def _arun(
        self,
        query: str,
    ):
        """Search Bing and return the results."""

        search = SerpAPIWrapper(
            serpapi_api_key=SERPAPI_API_KEY, params={"engine": "bing"}
        )

        try:
            return await search.arun(query)
        except Exception as err:
            if "Invalid API key" in str(err):
                bt.logging.error(f"SERP API Key is invalid: {err}")
                return "SERP API Key is invalid"

            bt.logging.error(f"Could not perform SERP Bing Search: {err}")
            return "Could not search Bing. Please try again later."

    async def send_event(self, send: Send, response_streamer, data):
        if not data:
            return

        search_results_response_body = {
            "type": "search",
            "content": data,
        }

        response_streamer.more_body = False

        await send(
            {
                "type": "http.response.body",
                "body": json.dumps(search_results_response_body).encode("utf-8"),
                "more_body": True,
            }
        )

        bt.logging.info("Web search results data sent")
