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
    CodeAgentPrompt,
    ReasoningAgentPrompt,
    code_prompt
)
from opendeepsearch.advancements.AdvancedAgent import (
    AdvancedAgent
)

# Loading all global variables defined in .env file of working directory
load_dotenv()

max_steps = 15
shuffle = False

# GETTING THE MODEL
search_model_name = "fireworks_ai/accounts/fireworks/models/qwen2-vl-72b-instruct"
reasoning_model_name = "fireworks_ai/accounts/fireworks/models/qwen2-vl-72b-instruct"
code_model_name = "fireworks_ai/accounts/fireworks/models/qwen2p5-coder-32b-instruct"
reasoning_model = LiteLLMModel(
    model_id=reasoning_model_name,
    temperature=0.2,
)
code_model = LiteLLMModel(
    model_id=code_model_name,
    temperature=0.2,
)

# GETTING THE DATASET
# data_frame = pd.read_csv('evals/datasets/simple_qa_test_set.csv')
data_frame = pd.read_csv('evals/datasets/frames_test_set.csv')

dataset = Dataset.from_pandas(data_frame)

start = 0
stop = len(dataset)
if (stop or stop==0) and start:
    num_samples = stop - start
else: num_samples = None


if shuffle:
    dataset = dataset.shuffle()
    dataset = dataset.shuffle()
if not num_samples is None:
    dataset = dataset.select(range(start, stop))
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
agent = CodeAgent(
    tools=[search_tool],
    model=code_model,
    additional_authorized_imports=["numpy"],
    prompt_templates=CodeAgentPrompt
)
# agent = AdvancedAgent(
#     tools=[search_tool],
#     code_model=code_model,
#     reasoning_model=reasoning_model,
#     additional_authorized_imports=["numpy"],
#     prompt_templates=ReasoningAgentPrompt,
#     code_prompt_template=code_prompt
# )

# RUNNING THE EVALUATION
filename = f"out/advanced_agent.jsonl"

for idx, entry in enumerate(dataset):
    prompt = entry['question']

    print('\033[95m' + f"Testing sample {idx + 1} out of {num_samples}" + '\033[0m')

    start_time = time.time()
    answer = agent.run(prompt, max_steps=max_steps)
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

    print('\033[95m' + f"Actual answer is: {annotated_output['true_answer']}" + '\033[0m')

    # SAVING THE RESULT
    with open(filename, 'a') as f:
        json.dump(annotated_output, f)
        f.write("\n")


# GRADING THE RESULT
print(f"To grade the trial, run the following:\n\npython evals/autograde_df.py {filename}\n")