import unittest
from neurons.validators.reward.deep_research_system_message import (
    DeepResearchSystemMessageRelevanceModel,
)
from tests_data.reports.what_is_blockchain import report_what_is_blockchain
from datura.protocol import DeepResearchSynapse


class DeepResearchSystemMessageRelevanceModelTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.device = "test_device"
        self.scoring_type = None
        self.model = DeepResearchSystemMessageRelevanceModel(
            self.device, self.scoring_type
        )

    async def test_check_response(self):
        report = report_what_is_blockchain
        score = await self.model.check_response(
            DeepResearchSynapse(
                prompt="what is blockchain",
                report=report,
                system_message="Report should mainly consists of less than 12 sections",
            )
        )
        self.assertEqual(score, 1)

        score = await self.model.check_response(
            DeepResearchSynapse(
                prompt="what is blockchain",
                report=report,
                system_message="Report should mainly consists of less than 5 sections",
            )
        )
        self.assertEqual(score, 0.2)


if __name__ == "__main__":
    unittest.main()
