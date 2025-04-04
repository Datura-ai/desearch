import time
import torch
import random
from datura.redis.redis_client import redis_client
import jsonpickle


class OrganicHistoryMixin:
    HISTORY_EXPIRY_TIME = 2 * 3600

    def __init__(self):
        self.organic_history = self._load_history()

    @property
    def redis_key(self):
        return f"{self.__class__.__name__}:organic_history"

    def _load_history(self):
        data = jsonpickle.decode(redis_client.get(self.redis_key) or "{}")
        return {int(uid): values for uid, values in data.items()}

    def _save_history(self, history):
        redis_client.set(
            self.redis_key, jsonpickle.encode(history), ex=self.HISTORY_EXPIRY_TIME
        )

    def _clean_organic_history(self):
        current_time = time.time()
        self.organic_history = {
            uid: [
                value
                for value in values
                if value["start_time"] >= current_time - self.HISTORY_EXPIRY_TIME
            ]
            for uid, values in self.organic_history.items()
        }

        self.organic_history = {
            uid: values
            for uid, values in self.organic_history.items()
            if len(values) > 0
        }

        self._save_history(self.organic_history)

        return self.organic_history

    def _save_organic_response(self, uids, responses, tasks, event, start_time) -> None:
        for uid, response, task, *event_values in zip(
            uids, responses, tasks, *event.values()
        ):
            event = dict(zip(event.keys(), event_values))

            if uid.item() not in self.organic_history:
                self.organic_history[uid.item()] = []

            self.organic_history[uid.item()].append(
                {
                    "response": response,
                    "task": task,
                    "event": event,
                    "start_time": start_time,
                }
            )

        self._save_history(self.organic_history)

    def get_random_organic_responses(self):
        self._clean_organic_history()

        event = {}
        tasks = []
        responses = []
        uids = []

        for uid, item in self.organic_history.items():
            uids.append(torch.tensor([uid]))

            random_index = random.randint(0, len(item) - 1)

            responses.append(item[random_index]["response"])
            tasks.append(item[random_index]["task"])
            for key, value in item[random_index]["event"].items():
                if not key in event:
                    event[key] = []

                event[key].append(value)

        return {
            "event": event,
            "tasks": tasks,
            "responses": responses,
            "uids": torch.tensor(uids),
        }

    def get_latest_organic_responses(self):
        self._clean_organic_history()

        event = {}
        tasks = []
        responses = []
        uids = []

        for uid, item in self.organic_history.items():
            uids.append(torch.tensor([uid]))
            responses.append(item[-1]["response"])
            tasks.append(item[-1]["task"])
            for key, value in item[-1]["event"].items():
                if not key in event:
                    event[key] = []

                event[key].append(value)

        return {
            "event": event,
            "tasks": tasks,
            "responses": responses,
            "uids": torch.tensor(uids),
        }

    def get_uids_with_no_history(self, available_uids):
        self._clean_organic_history()

        uids = [uid for uid in available_uids if uid not in self.organic_history]

        return uids
