from smolagents import CodeAgent

from typing import TYPE_CHECKING, Any, Union

from rich.console import Group
from rich.text import Text

from smolagents.local_python_executor import fix_final_answer_code
from smolagents.memory import ActionStep, ToolCall
from smolagents.models import ChatMessage
from smolagents.monitoring import (
    YELLOW_HEX,
    LogLevel,
)
from smolagents.utils import (
    AgentExecutionError,
    AgentGenerationError,
    AgentParsingError,
    parse_code_blobs,
    truncate_content,
)

class AdvancedAgent(CodeAgent):

    def __init__(self, tools, code_model, reasoning_model, prompt_templates = None, code_prompt_template = None, grammar = None, additional_authorized_imports = None, planning_interval = None, executor_type = "local", executor_kwargs = None, max_print_outputs_length = None, **kwargs):
        super().__init__(tools, reasoning_model, prompt_templates, grammar, additional_authorized_imports, planning_interval, executor_type, executor_kwargs, max_print_outputs_length, **kwargs)
        self.code_model = code_model
        self.code_prompt_template = code_prompt_template

    def format_code_prompt(self, reasoning_out_message):
        code_model_prompt = [self.code_prompt_template]
        pass
        return code_model_prompt

    
    def step(self, memory_step: ActionStep) -> Union[None, Any]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns None if the step is not final.
        """
        memory_messages = self.write_memory_to_messages()

        self.input_messages = memory_messages.copy()

        # Add new step in logs
        memory_step.model_input_messages = memory_messages.copy()
        try:
            additional_args = {"grammar": self.grammar} if self.grammar is not None else {}
            chat_message: ChatMessage = self.model(
                self.input_messages,
                stop_sequences=["<end_code>", "Observation:", "Calling tools:"],
                **additional_args,
            )
            memory_step.model_output_message = chat_message
            model_output = chat_message.content

            code_model_prompt = self.format_code_prompt(chat_message)
            code_block = self.code_model(
                code_model_prompt,
                stop_sequences=["<end_code>", "Observation:", "Calling tools:"],
            )

            code_block = code_block.content

            # This adds <end_code> sequence to the history.
            # This will nudge ulterior LLM calls to finish with <end_code>, thus efficiently stopping generation.
            if code_block and code_block.strip().endswith("```"):
                code_block += "<end_code>"

            memory_step.model_output = model_output
        except Exception as e:
            raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e

        self.logger.log_markdown(
            content=model_output,
            title="Output message of the LLM:",
            level=LogLevel.DEBUG,
        )

        # Parse
        try:
            code_action = fix_final_answer_code(parse_code_blobs(code_block))
        except Exception as e:
            error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            raise AgentParsingError(error_msg, self.logger)

        memory_step.tool_calls = [
            ToolCall(
                name="python_interpreter",
                arguments=code_action,
                id=f"call_{len(self.memory.steps)}",
            )
        ]

        # Execute
        self.logger.log_code(title="Executing parsed code:", content=code_action, level=LogLevel.INFO)
        is_final_answer = False
        try:
            output, execution_logs, is_final_answer = self.python_executor(code_action)
            execution_outputs_console = []
            if len(execution_logs) > 0:
                execution_outputs_console += [
                    Text("Execution logs:", style="bold"),
                    Text(execution_logs),
                ]
            observation = "Execution logs:\n" + execution_logs
        except Exception as e:
            if hasattr(self.python_executor, "state") and "_print_outputs" in self.python_executor.state:
                execution_logs = str(self.python_executor.state["_print_outputs"])
                if len(execution_logs) > 0:
                    execution_outputs_console = [
                        Text("Execution logs:", style="bold"),
                        Text(execution_logs),
                    ]
                    memory_step.observations = "Execution logs:\n" + execution_logs
                    self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
            error_msg = str(e)
            if "Import of " in error_msg and " is not allowed" in error_msg:
                self.logger.log(
                    "[bold red]Warning to user: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when initializing your CodeAgent.",
                    level=LogLevel.INFO,
                )
            raise AgentExecutionError(error_msg, self.logger)

        truncated_output = truncate_content(str(output))
        observation += "Last output from code snippet:\n" + truncated_output
        memory_step.observations = observation

        execution_outputs_console += [
            Text(
                f"{('Out - Final answer' if is_final_answer else 'Out')}: {truncated_output}",
                style=(f"bold {YELLOW_HEX}" if is_final_answer else ""),
            ),
        ]
        self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
        memory_step.action_output = output
        return output if is_final_answer else None