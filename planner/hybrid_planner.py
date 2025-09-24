import os
from typing import Dict, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_prompty import create_chat_prompt
from langchain.agents import create_tool_calling_agent, AgentExecutor


from .base import BasePlanner
from .schemas.plan import Plan
from .utils.tracker import Tracker
from .tools.tools import *


class HybridPlanner(BasePlanner):
    llm: BaseChatModel
    tracker: Tracker
    mission_statement: str = ""
    number_of_agents: int = -1
    grid_size: int = -1
    number_of_targets: int = -1
    agent_positions: Dict[int, Tuple[int, int]]

    def __init__(
        self,
        llm: BaseChatModel,
        grid_size,
        observations,
        infos,
    ) -> None:
        self.llm = llm
        self.mission_statement = observations[0]["mission"]
        self.number_of_agents = len(observations.keys()) - 1
        self.number_of_targets = observations["global"]["num_goals"]
        self.agent_positions = {i: (1, 1) for i in range(0, self.number_of_agents)}
        self.grid_size = grid_size
        self.tools = [partition_grid, find_path]

        self.agent = self._init_agent()

    from langchain_core.prompts import ChatPromptTemplate

    def _init_agent(self):
        base_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a multi-agent planner. "
                       "Always use the provided tools (partition_grid, find_path). "
                       "After using tools, output a valid Plan JSON."),
            ("user", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),

        ])

        agent_runnable = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=base_prompt,
        )

        # Wrap in AgentExecutor so `intermediate_steps` is provided + tools are executed
        agent_executor = AgentExecutor(
            agent=agent_runnable,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,  # helpful for hackathons
        )
        return agent_executor



    def initial_plan(self) -> dict:
        # Use prompty only to generate the big mission instruction
        prompty = create_chat_prompt(os.getcwd() + "/prompts/tool2.prompty")
        filled_prompt = prompty.invoke({
            "grid_length": self.grid_size,
            "num_agents": self.number_of_agents,
            "num_targets": self.number_of_targets,
            "mission": self.mission_statement,
        })

        # Feed the prompty text into the agent under "input"
        raw_output = self.agent.invoke({"input": filled_prompt.to_string()})

        # Parse into Plan schema
        model_with_schema = self.llm.with_structured_output(Plan)
        plan = model_with_schema.invoke(raw_output["output"])
        return plan.agents



    def replan(self, observations, rewards, terminations, truncations, infos) -> dict:
        # self.tracker.observe(observations, rewards)
        return {}
