import traceback
import random
from typing import List, Dict, Tuple
import json
import bittensor as bt
from .config import RewardModelType
from .reward import BaseRewardModel, BaseRewardEvent
from datura.protocol import DeepResearchSynapse, ReportItem
from neurons.validators.utils.prompt.deep_research.deep_research_source_links_relevance_prompt import (
    DeepResearchSourceLinksRelevancePrompt,
)
from datura.synapse import collect_responses
from neurons.validators.apify.web_scraper_actor import WebScraperActor

RANDOM_SECTIONS_COUNT = 1


class DeepResearchSourceLinksRelevanceModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return RewardModelType.deep_research_source_links_relevance.value

    def __init__(self, device: str, scoring_type: None):
        super().__init__()
        self.device = device
        self.scoring_type = scoring_type
        self.relevance_prompt = DeepResearchSourceLinksRelevancePrompt()
        self.web_scraper_actor = WebScraperActor()

        self.is_default_normalization = False

    def get_section_links(self, section: ReportItem):
        links = section.links
        return links

    async def check_section_links(self, section: ReportItem, prompt: str):
        try:
            links = self.get_section_links(section)

            if len(links) == 0:
                return 1.0

            results = await self.web_scraper_actor.scrape_metadata(links)

            result_texts = [result["html_text"] for result in results]

            response = await self.relevance_prompt.get_response(
                result_texts.__str__(), prompt
            )

            return self.relevance_prompt.extract_score(response) / 10
        except Exception as e:
            bt.logging.error(
                f"deep_research_source_links check_section_links error: {str(e)}"
            )
            return 0.0

    async def check_response(self, synapse: DeepResearchSynapse) -> float:
        try:
            all_sections = [item for item in synapse.report if len(item.links) > 0]
            for report in synapse.report:
                all_sections.extend(
                    [item for item in report.subsections if len(item.links) > 0]
                )

            sections = random.choices(all_sections, k=RANDOM_SECTIONS_COUNT)

            scores = await collect_responses(
                [
                    self.check_section_links(section, synapse.prompt)
                    for section in sections
                ]
            )

            return sum(scores) / len(scores) if scores else 0.0

        except Exception as e:
            bt.logging.error(
                f"deep_research_source_links check_response error: {str(e)}"
            )
            return 0.0

    async def get_rewards(
        self, responses: List[DeepResearchSynapse], uids: List[int]
    ) -> Tuple[List[BaseRewardEvent], Dict[int, float]]:
        try:

            reward_events = []
            zero_scores = {}
            non_zero_scores = {}
            grouped_val_score_responses = {}

            # Step 2: for each response, compute a final score
            for response, uid_tensor in zip(responses, uids):
                # If uid_tensor is a PyTorch or NumPy scalar, .item() extracts the integer
                uid = uid_tensor.item() if hasattr(uid_tensor, "item") else uid_tensor

                final_score = await self.check_response(response)

                bt.logging.info(
                    f"UID {uid}: deep research source links relevance score => {final_score}"
                )

                # Step 3: create a reward event
                reward_event = BaseRewardEvent()
                reward_event.reward = final_score
                reward_events.append(reward_event)

                # Keep track of final_score for logging
                if final_score == 0:
                    zero_scores[uid] = final_score
                else:
                    non_zero_scores[uid] = final_score

                # Populate grouped_val_score_responses with final_score
                grouped_val_score_responses[uid] = final_score

            # Step 4: Log zero vs. non-zero
            bt.logging.info(
                f"========== Deep Research Source Links Relevance Check Zero Scores ({len(zero_scores)} cases) =========="
            )
            bt.logging.info(json.dumps(zero_scores))
            bt.logging.info(
                f"======== Deep Research Source Links Relevance Check Non-Zero Scores ({len(non_zero_scores)} cases) ========"
            )
            bt.logging.info(json.dumps(non_zero_scores))

            return reward_events, grouped_val_score_responses
        except Exception as e:
            tb_str = traceback.format_exception(type(e), e, e.__traceback__)
            bt.logging.error("\n".join(tb_str))

            # On exception, return zeroed events
            reward_events = []
            for _ in responses:
                revent = BaseRewardEvent()
                revent.reward = 0
                reward_events.append(revent)

            return reward_events, {}
