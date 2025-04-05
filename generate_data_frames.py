from opendeepsearch import OpenDeepSearchTool
import os
from dotenv import load_dotenv

from datasets import Dataset
import pandas as pd
import json
import time
from datetime import datetime

from smolagents import CodeAgent, MultiStepAgent, GradioUI, LiteLLMModel
from smolagents.agents import ActionStep

from opendeepsearch.advancements.AdvancedPrompts import (
    CodeAgentPrompt
)
from opendeepsearch.advancements.AdvancedAgent import (
    AdvancedAgent
)

# Loading all global variables defined in .env file of working directory
load_dotenv()

num_samples = 5
shuffle = True

# GETTING THE MODEL
search_model_name = "fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct"
code_model_name = "fireworks_ai/accounts/fireworks/models/qwen2p5-coder-32b-instruct"
code_model = LiteLLMModel(
    model_id=code_model_name,
    temperature=0.2,
)

# GETTING THE DATASET
# data_frame = pd.read_csv('evals/datasets/simple_qa_test_set.csv')
data_frame = pd.read_csv('evals/datasets/frames_test_set.csv')

dataset = Dataset.from_pandas(data_frame)
if shuffle:
    dataset = dataset.shuffle()
if not num_samples is None:
    dataset = dataset.select(range(num_samples))
else:
    num_samples = len(dataset)

# SETTING UP ALL THE TOOLS

# Using Serper (default)
search_tool = OpenDeepSearchTool(
    model_name=search_model_name,
    reranker="infinity",
    search_provider='serper',
)
search_tool.setup()


# SETTING UP THE AGENT
agent = AdvancedAgent(
    tools=[search_tool],
    model=code_model,
    additional_authorized_imports=["numpy"],
    prompt_templates=CodeAgentPrompt
)

# RUNNING THE EVALUATION
timestamp = datetime.now().strftime("%H_%M_%S")
filename = f"out/output_trial_{timestamp}.jsonl"

for idx, entry in enumerate(dataset):
    prompt = entry['question']

    print('\033[95m' + f"\nTesting sample number {idx + 1} out of {num_samples}\n" + '\033[0m')

    start_time = time.time()
    answer = agent.run(prompt, max_steps=15)
    end_time = time.time()

    # Remove memory from logs to make them more compact.
    for step in agent.memory.steps:
        if isinstance(step, ActionStep):
            step.agent_memory = None
    
    annotated_output = {
        "idx": idx,
        "model_id": code_model.model_id,
        "agent_action_type": 'MachineEarning',
        "original_question": entry['question'],
        "answer": answer,
        "true_answer": entry["true_answer"],
        "intermediate_steps": str(agent.memory.steps),
        "start_time": start_time,
        "end_time": end_time,
        "token_counts": agent.monitor.get_total_token_counts(),
    }

    print('\033[95m' + f"\nActual final answer is {annotated_output['true_answer']}\n" + '\033[0m')

    # SAVING THE RESULT
    with open(filename, 'a') as f:
        json.dump(annotated_output, f)
        f.write("\n")


# GRADING THE RESULT
print(f"To grade the trial, run the following:\n\npython evals/autograde_df.py {filename}\n")