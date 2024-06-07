import random
import torch
import copy
import os
import asyncio
import bittensor as bt
import re
import time
from datura.utils import call_openai
from transformers import AutoTokenizer, AutoModelForCausalLM
from neurons.validators.config import add_args, check_config, config
from neurons.validators.reward.subnet_18_reward import StreamPrompting, query_miner
from neurons.validators.utils.prompts import (
    extract_score_and_explanation,
)
from neurons.validators.utils.prompts import ScoringPrompt

from enum import Enum
from transformers import pipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"

EXPECTED_ACCESS_KEY = os.environ.get("EXPECTED_ACCESS_KEY", "hello")
URL_SUBNET_18 = os.environ.get("URL_SUBNET_18")
SUBNET_18_VALIDATOR_UID = os.environ.get("SUBNET_18_VALIDATOR_UID", "0")


class ScoringSource(Enum):
    Subnet18 = 1
    OpenAI = 2
    LocalLLM = 3
    LocalZephyr = 4


class RewardLLM:
    subtensor: "bt.subtensor"
    wallet: "bt.wallet"
    metagraph: "bt.metagraph"
    dendrite: "bt.dendrite"

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def config(cls):
        return config(cls)

    def __init__(self):
        self.config = RewardLLM.config()

        self.tokenizer = None
        self.model = None
        self.device = None
        self.pipe = None
        self.scoring_prompt = ScoringPrompt()

        self.initialize_components()

    def initialize_components(self):
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"Your validator: {self.wallet} is not registered to chain connection: {self.subtensor}. Run btcli register --netuid 18 and try again."
            )
            exit()

    def init_tokenizer(self, device, model_name):
        # https://huggingface.co/VMware/open-llama-7b-open-instruct
        # Fast tokenizer results in incorrect encoding, set the use_fast = False parameter.
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        # Generative default expects most recent token on right-hand side with padding on left.
        # https://github.com/huggingface/transformers/pull/10552
        tokenizer.padding_side = "left"

        # Check if the device is CPU or CUDA and set the precision accordingly
        torch_dtype = torch.float32 if device == "cpu" else torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype
        ).to(device)

        self.tokenizer = tokenizer
        self.model = model
        self.device = device

        return tokenizer, model

    def init_pipe_zephyr(self):
        pipe = pipeline(
            "text-generation",
            model="HuggingFaceH4/zephyr-7b-alpha",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.pipe = pipe
        return pipe

    def clean_text(self, text):
        # Remove newline characters and replace with a space
        text = text.replace("\n", " ")

        # Remove URLs
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )

        # Keep hashtags, alphanumeric characters, and spaces
        # Remove other special characters but ensure to keep structured elements like <Question>, <Answer>, etc., intact
        text = re.sub(r"(?<![\w<>#])[^\w\s#<>]+", "", text)

        return text

    async def get_score_by_subnet_18(self, data):
        def flatten_content(content):
            if isinstance(content, list):
                return " ".join(flatten_content(item) for item in content)
            if isinstance(content, dict):
                return " ".join(f"{key}: {flatten_content(value)}" for key, value in content.items())
            return str(content)

        start_time = time.time()  # Start timing for execution
        try:
            top_miners_to_use = 100
            top_miner_uids = self.metagraph.I.argsort(descending=True)[:top_miners_to_use]
            miner_uid = random.choice(top_miner_uids)

            flattened_data = flatten_content(data)
            messages = [{'role': 'user', 'content': flattened_data}]
            synapse = StreamPrompting(
                messages=messages,
                provider="Claude",
                model="claude-3-opus-20240229",
                uid=miner_uid,
                completion="",
            )
            response = await query_miner(
                self.dendrite,
                self.metagraph.axons[miner_uid],
                synapse,
                60,   # timeout
                True  # streaming
            )
            execution_time = (
                time.time() - start_time
            ) / 60  # Calculate execution time in minutes
            bt.logging.info(
                f"Subnet 18 scoring call execution time: {execution_time:.2f} minutes"
            )
            return response
        except Exception as e:
            bt.logging.warning(f"Error calling Subnet 18 scoring: {e}")
            return None

    async def get_score_by_openai(self, messages):
        try:
            start_time = time.time()  # Start timing for query execution
            query_tasks = []
            for message_dict in messages:  # Iterate over each dictionary in the list
                ((key, message_list),) = message_dict.items()

                async def query_openai(message):
                    try:
                        return await call_openai(
                            messages=message,
                            temperature=0.2,
                            model="gpt-3.5-turbo-16k",
                        )
                    except Exception as e:
                        print(f"Error sending message to OpenAI: {e}")
                        return ""  # Return an empty string to indicate failure

                task = query_openai(message_list)
                query_tasks.append(task)

            query_responses = await asyncio.gather(*query_tasks, return_exceptions=True)

            result = {}
            for response, message_dict in zip(query_responses, messages):
                if isinstance(response, Exception):
                    print(f"Query failed with exception: {response}")
                    response = (
                        ""  # Replace the exception with an empty string in the result
                    )
                ((key, message_list),) = message_dict.items()
                result[key] = response

            execution_time = time.time() - start_time  # Calculate execution time
            print(f"Execution time for OpenAI queries: {execution_time} seconds")
            return result
        except Exception as e:
            print(f"Error processing OpenAI queries: {e}")
            return None

    def get_score_by_llm(self, messages):
        result = {}
        total_start_time = time.time()  # Start timing for total execution
        try:
            for message_dict in messages:  # Iterate over each dictionary in the list
                ((key, message_list),) = message_dict.items()

                with torch.no_grad():
                    # Choose correct scoring prompt for request type.
                    scoring_prompt_text = self.clean_text(
                        message_list[-1]["content"]
                    )  # Determine the scoring prompt based on the provided name or the default scoring type.

                    # Tokenize formatted scoring prompt.
                    encodings_dict = self.tokenizer(
                        scoring_prompt_text,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    input_ids = encodings_dict["input_ids"].to(self.device)

                    # Prompt local reward model.
                    start_time = time.time()
                    generated_tokens = self.model.generate(
                        input_ids, max_new_tokens=500, max_time=5
                    )
                    duration = time.time() - start_time

                    # Decode the new tokens to get the generated text
                    generated_text = self.tokenizer.decode(
                        generated_tokens[0], skip_special_tokens=True
                    )

                    # Extract score from generated text.
                    score_text = extract_score_and_explanation(generated_text)
                    # bt.logging.info(f"Score text: {score_text}")
                    result[key] = score_text

            total_duration = (
                time.time() - total_start_time
            )  # Calculate total execution time
            bt.logging.info(
                f"Total execution time for get_score_by_llm: {total_duration} seconds"
            )
        except Exception as e:
            bt.logging.error(f"Error in get_score_by_llm: {e}")
            return None
        return result

    def get_score_by_zephyer(self, messages):
        result = {}
        total_start_time = time.time()  # Start timing for total execution
        try:
            # Prepare batch
            prompts = []
            keys = []
            for message_dict in messages:  # Iterate over each dictionary in the list
                ((key, message_list),) = message_dict.items()
                prompt = self.pipe.tokenizer.apply_chat_template(
                    message_list, tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt)
                keys.append(key)

            # Process batch
            outputs = self.pipe(
                prompts,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.2,
                top_k=50,
                top_p=0.95,
            )

            # Process outputs
            for key, output in zip(keys, outputs):
                generated_text = output[0]["generated_text"]
                # score_text = extract_score_and_explanation(generated_text)
                score_text = extract_score_and_explanation(generated_text)
                result[key] = score_text

            total_duration = (
                time.time() - total_start_time
            )  # Calculate total execution time
            bt.logging.info(
                f"Total execution time for get_score_by_zephyer: {total_duration} seconds"
            )
        except Exception as e:
            bt.logging.error(f"Error in get_score_by_zephyer: {e}")
            return None
        return result

    def get_score_by_source(self, messages, source: ScoringSource):
        if source == ScoringSource.LocalZephyr:
            return self.get_score_by_zephyer(messages)
        if source == ScoringSource.Subnet18:
            loop = asyncio.get_event_loop_policy().get_event_loop()
            return loop.run_until_complete(self.get_score_by_subnet_18(messages))
        elif source == ScoringSource.OpenAI:
            loop = asyncio.get_event_loop_policy().get_event_loop()
            return loop.run_until_complete(self.get_score_by_openai(messages=messages))
        else:
            return self.get_score_by_llm(messages=messages)

    def llm_processing(self, messages):
        bt.logging.info("[?] >>>>>>>> Starting process of llm reward")
        # Initialize score_responses as an empty dictionary to hold the scoring results
        score_responses = {}

        # Define the order of scoring sources to be used
        scoring_sources = [
            # ScoringSource.LocalZephyr,  # Fallback to Local LLM if Subnet 18 fails or is disabled
            # ScoringSource.OpenAI,  # Final attempt with OpenAI if both Subnet 18 and Local LLM fail
            ScoringSource.Subnet18,  # First attempt with Subnet 18
        ]

        # Attempt to score messages using the defined sources in order
        for source in scoring_sources:
            bt.logging.info(f"[?] >>>>>>>> processing with source:{source}")
            # Attempt to score with the current source
            current_score_responses = self.get_score_by_source(
                messages=messages, source=source
            )
            bt.logging.info("[?] >>>>>>>> score responses")
            bt.logging.info(f"[?] >>>>>>>> {current_score_responses}")
            bt.logging.info("[?] >>>>>>>> score responses")
            if current_score_responses:
                # Update the score_responses with the new scores
                score_responses.update(current_score_responses)

                # Filter messages that still need scoring (i.e., messages that did not receive a score)
                messages = [
                    message
                    for (key, score_text), message in zip(
                        current_score_responses.items(), messages
                    )
                    if score_text is not None
                    and not any(char.isdigit() for char in score_text)
                ]
                # If all messages have been scored, break out of the loop
                if not messages:
                    break
                else:
                    bt.logging.info(
                        f"{source} Attempt for scoring. Remaining messages: {len(messages)}"
                    )
            else:
                bt.logging.info(
                    f"Scoring with {source} failed or returned no results. Attempting next source."
                )

        return score_responses
