import traceback
import bittensor as bt
from starlette.types import Send
from datura.protocol import (
    DeepResearchSynapse,
)
from datura.tools.tool_manager import ToolManager
from datura.dataset.date_filters import (
    DateFilter,
    DateFilterType,
    get_specified_date_filter,
)
from datetime import datetime
import pytz
import requests
import json


class DeepResearchMiner:
    def __init__(self, miner: any):
        self.miner = miner

    async def deep_research(self, synapse: DeepResearchSynapse, send: Send):
        try:
            response = requests.get(
                "http://127.0.0.1:8008/deep-research",
                json={
                    "prompt": synapse.prompt,
                    "tools": synapse.tools,
                    "date_filter": synapse.date_filter_type,
                    "system_message": synapse.system_message,
                },
            )

            result = response.json()

            # await tool_manager.run()
            await send(
                {
                    "type": "http.response.body",
                    "body": json.dumps(result).encode("utf-8"),
                    "more_body": False,
                }
            )

            bt.logging.info("End of Streaming")

        except Exception as e:
            bt.logging.error(f"error in deep research {e}\n{traceback.format_exc()}")
