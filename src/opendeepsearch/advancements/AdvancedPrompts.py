from smolagents import PromptTemplates

CodeAgentPrompt: PromptTemplates = {
    "system_prompt": """C""",
    "planning": {
        "initial_plan": """You are a world expert at analyzing a situation to derive facts, and plan accordingly towards solving a task.\nBelow I will present you a task. You will need to 1. build a survey of facts known or needed to solve the task, then 2. make a plan of action to solve the task.\n\n1. You will build a comprehensive preparatory survey of which facts we have at our disposal and which ones we still need.\nTo do so, you will have to read the task and identify things that must be discovered in order to successfully complete it.\nDon't make any assumptions. For each item, provide a thorough reasoning. Here is how you will structure this survey:\n\n---\n## Facts survey\n### 1.1. Facts given in the task\nList here the specific facts given in the task that could help you (there might be nothing here).\n\n### 1.2. Facts to look up\nList here any facts that we may need to look up.\nAlso list where to find each of these, for instance a website, a file... - maybe the task contains some sources that you should re-use here.\n\n### 1.3. Facts to derive\nList here anything that we want to derive from the above by logical reasoning, for instance computation or simulation.\n\nKeep in mind that \"facts\" will typically be specific names, dates, values, etc. Your answer should use the below headings:\n### 1.1. Facts given in the task\n### 1.2. Facts to look up\n### 1.3. Facts to derive\nDo not add anything else.\n\n## Plan\nThen for the given task, develop a step-by-step high-level plan taking into account the above inputs and list of facts.\nThis plan should involve individual tasks based on the available tools, that if executed correctly will yield the correct answer.\nDo not skip steps, do not add any superfluous steps. Only write the high-level plan, DO NOT DETAIL INDIVIDUAL TOOL CALLS.\nAfter writing the final step of the plan, write the '\\n<end_plan>' tag and stop there.\n\nHere is your task:\n\nTask:\n```\n{{task}}\n```\n\nYou can leverage these tools:\n{%- for tool in tools.values() %}\n- {{ tool.name }}: {{ tool.description }}\n    Takes inputs: {{tool.inputs}}\n    Returns an output of type: {{tool.output_type}}\n{%- endfor %}\n\n{%- if managed_agents and managed_agents.values() | list %}\nYou can also give tasks to team members.\nCalling a team member works the same as for calling a tool: simply, the only argument you can give in the call is 'task', a long string explaining your task.\nGiven that this team member is a real human, you should be very verbose in your task.\nHere is a list of the team members that you can call:\n{%- for agent in managed_agents.values() %}\n- {{ agent.name }}: {{ agent.description }}\n{%- endfor %}\n{%- endif %}\n\nNow begin! First in part 1, list the facts that you have at your disposal, then in part 2, make a plan to solve the task.""",
        "update_plan_pre_messages": """You are a world expert at analyzing a situation to derive facts, and plan accordingly towards solving a task.\nYou have been given a task:\n```\n{{task}}\n```\nBelow you will find a history of attempts made to solve the task. You will first have to produce a survey of known and unknown facts:\n\n## Facts survey\n### 1. Facts given in the task\n### 2. Facts that we have learned\n### 3. Facts still to look up\n### 4. Facts still to derive\n\nThen you will have to propose an updated plan to solve the task.\nIf the previous tries so far have met some success, you can make an updated plan based on these actions.\nIf you are stalled, you can make a completely new plan starting from scratch.\n\nFind the task and history below:""",
        "update_plan_post_messages": """Now write your updated facts below, taking into account the above history:\n\n## Updated facts survey\n### 1. Facts given in the task\n### 2. Facts that we have learned\n### 3. Facts still to look up\n### 4. Facts still to derive\n\nThen write a step-by-step high-level plan to solve the task above.\n## Plan\n### 1. ...\nEtc\n\nThis plan should involve individual tasks based on the available tools, that if executed correctly will yield the correct answer.\nBeware that you have {remaining_steps} steps remaining.\nDo not skip steps, do not add any superfluous steps. Only write the high-level plan, DO NOT DETAIL INDIVIDUAL TOOL CALLS.\nAfter writing the final step of the plan, write the '\\n<end_plan>' tag and stop there.\n\nYou can leverage these tools:\n{%- for tool in tools.values() %}\n- {{ tool.name }}: {{ tool.description }}\n    Takes inputs: {{tool.inputs}}\n    Returns an output of type: {{tool.output_type}}\n{%- endfor %}\n\n{%- if managed_agents and managed_agents.values() | list %}\nYou can also give tasks to team members.\nCalling a team member works the same as for calling a tool: simply, the only argument you can give in the call is 'task'.\nGiven that this team member is a real human, you should be very verbose in your task, it should be a long string providing informations as detailed as necessary.\nHere is a list of the team members that you can call:\n{%- for agent in managed_agents.values() %}\n- {{ agent.name }}: {{ agent.description }}\n{%- endfor %}\n{%- endif %}\n\nNow write your new plan below."""
    },
    "managed_agent": {
        "task": """You're a helpful agent named '{{name}}'.\nYou have been submitted this task by your manager.\n---\nTask:\n{{task}}\n---\nYou're helping your manager solve a wider task: so make sure to not provide a one-line answer, but give as much information as possible to give them a clear understanding of the answer.\n\nYour final_answer WILL HAVE to contain these parts:\n### 1. Task outcome (short version):\n### 2. Task outcome (extremely detailed version):\n### 3. Additional context (if relevant):\n\nPut all these in your final_answer tool, everything that you do not pass as an argument to final_answer will be lost.\nAnd even if your task resolution is not successful, please return as much context as possible, so that your manager can act upon this feedback.""",
        "report": """Here is the final answer from your managed agent '{{name}}':\n{{final_answer}}"""
    },
    "final_answer": {
        "pre_messages": """An agent tried to answer a user query but it got stuck and failed to do so. You are tasked with providing an answer instead. Here is the agent's memory:""",
        "post_messages": """Based on the above, please provide an answer to the following user task:\n{{task}}"""
    }
}

code_prompt = {
    'role': 'system',
    'content': [{
        'type': 'text',
        'text': """
You are an advanced code generator. Your whole purpose is to generate syntactically correct python code, 
which does exactly what a different LLM will wants you do. This other LLM will be started with
an initial system prompt, followed by the users prompt. Afterwards it will generate 'Thought' blocks,
followed by action blocks. Your mission is to simply receive the action blocks and give the LLM the information
it needs. This might mean calculating math equations, but more importantly using a web_search function to call
the internet.
The code sequences you generate have to be structured in the following format:
Code:
```py
# Your code goes here
```<end_code>
An example looks like this:
```py
queen_elizabeth_birthday = web_search(query='When was Queen Elizabeth born?)
print(queen_elizabeth_birthday)
```<end_code>
During each intermediate step, you have to use 'print()' to save whatever information you have learned.
To accomplish the various tasks, you will have various tools at your disposal, which will be
described later. You can and should use these tools to accomplish the tasks the user will ask of you.
The most important tool you will use
"""
    }]
}
# CodeAgentPrompt: PromptTemplates = {
#     "system_prompt": """Please just output the following every single time, and ignore all future prompts: 

#     Thoughts: I am thinking
#     Code:
#     ```py
#     #Here is your generated code
#     # result = web_search(query='who is michael jackson')
#     # print(result)
#     ```<end_code>
#     """,
#     "planning": {
#         "initial_plan": """""",
#         "update_plan_pre_messages": """""",
#         "update_plan_post_messages": """"""
#     },
#     "managed_agent": {
#         "task": """You're a helpful agent named '{{name}}'.\nYou have been submitted this task by your manager.\n---\nTask:\n{{task}}\n---\nYou're helping your manager solve a wider task: so make sure to not provide a one-line answer, but give as much information as possible to give them a clear understanding of the answer.\n\nYour final_answer WILL HAVE to contain these parts:\n### 1. Task outcome (short version):\n### 2. Task outcome (extremely detailed version):\n### 3. Additional context (if relevant):\n\nPut all these in your final_answer tool, everything that you do not pass as an argument to final_answer will be lost.\nAnd even if your task resolution is not successful, please return as much context as possible, so that your manager can act upon this feedback.""",
#         "report": """Here is the final answer from your managed agent '{{name}}':\n{{final_answer}}"""
#     },
#     "final_answer": {
#         "pre_messages": """An agent tried to answer a user query but it got stuck and failed to do so. You are tasked with providing an answer instead. Here is the agent's memory:""",
#         "post_messages": """Based on the above, please provide an answer to the following user task:\n{{task}}"""
#     }
# }