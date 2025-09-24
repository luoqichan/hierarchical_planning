import os
from typing import Dict, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_prompty import create_chat_prompt
from langchain.agents import initialize_agent, AgentType


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
        self.agent = self._init_agent()

    def _init_agent(self): 
        tools = [partition_grid, find_path, validate_plan]

        # create agent that can call tools
        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,  
        )
        return agent

    def initial_plan(self) -> dict:
        model_with_structure = self.llm.with_structured_output(Plan)
        prompt = create_chat_prompt(os.getcwd() + "/prompts/initial_planner.prompty")
        initial_planner = prompt | model_with_structure
        
        initial_plan = initial_planner.invoke(
            {
                "grid_length": self.grid_size,
                "num_agents": self.number_of_agents,
                "num_targets": self.number_of_targets,
                "mission": self.mission_statement,
            }
        )

        raw_output = self.agent.invoke(initial_plan)
        plan = raw_output['input']

        return plan.agents

    def replan(self, observations, rewards, terminations, truncations, infos) -> dict:
        # self.tracker.observe(observations, rewards)
        return {}
