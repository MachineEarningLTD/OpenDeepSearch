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

# Loading all global variables defined in .env file of working directory
load_dotenv()

num_samples = 24

# GETTING THE MODEL
search_model_name = "fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct"
code_model_name = "fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct"
code_model = LiteLLMModel(
    model_id=code_model_name,
    temperature=0.2,
)

# GETTING THE DATASET
# data_frame = pd.read_csv('evals/datasets/simple_qa_test_set.csv')
data_frame = pd.read_csv('evals/datasets/frames_test_set.csv')
dataset = Dataset.from_pandas(data_frame).shuffle().select(range(num_samples))


# SETTING UP ALL THE TOOLS

# Using Serper (default)
search_tool = OpenDeepSearchTool(
    model_name=search_model_name,
    reranker="jina",
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

# RUNNING THE EVALUATION
timestamp = datetime.now().strftime("%H_%M_%S")
filename = f"out/output_trial_{timestamp}.jsonl"

for idx, entry in enumerate(dataset):
    prompt = entry['question']

    start_time = time.time()
    answer = agent.run(prompt)
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

    # SAVING THE RESULT
    with open(filename, 'a') as f:
        json.dump(annotated_output, f)
        f.write("\n")


# GRADING THE RESULT
print(f"To grade the trial, run the following:\n\npython evals/autograde_df.py {filename}\n")