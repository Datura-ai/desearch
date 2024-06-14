from abc import ABC
from typing import List
from datura.tools.base import BaseToolkit, BaseTool
from datura.tools.discord.discord_summary import (
    prepare_messages_data_for_summary,
    summarize_discord_data,
)
from datura.tools.discord.discord_search_tool import DiscordSearchTool


class DiscordToolkit(BaseToolkit, ABC):
    name: str = "Discord Toolkit"
    description: str = "Toolkit containing tools for interacting discord."
    slug: str = "discord"
    toolkit_id: str = "fb78b028-f7f4-4d20-b7e8-7dc072e97d9a"

    def get_tools(self) -> List[BaseTool]:
        return [DiscordSearchTool()]

    async def summarize(self, prompt, model, data):
        data = next(iter(data.values()))
        return await summarize_discord_data(
            prompt=prompt,
            model=model,
            filtered_messages=prepare_messages_data_for_summary(data),
        )
