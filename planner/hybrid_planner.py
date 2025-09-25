import os
from typing import Dict, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks import StdOutCallbackHandler

from langchain_prompty import create_chat_prompt
from langchain.agents import create_tool_calling_agent, AgentExecutor

from .base import BasePlanner
from .schemas.plan import Plan, StopAction
from .utils.tracker import Tracker
from .tools.tools import *


class HybridPlanner(BasePlanner):
    llm: BaseChatModel
    tracker: Tracker
    mission_statement: str = ""
    number_of_agents: int = -1
    grid_size: int = -1
    number_of_targets: int = -1
    agent_trajectories: Dict[int, List[Tuple[int, int]]]
    found_targets = set()

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
        self.agent_trajectories = {i: [(1, 1)] for i in range(0, self.number_of_agents)}
        self.tracker = Tracker(grid_size)

        self.tools = [partition_grid_with_mission, assign_agents_to_regions, compute_expected_reward, evaluate_candidate_return]
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
        raw_output = self.agent.invoke({"input": filled_prompt.to_string()},
                                        config={"callbacks": [StdOutCallbackHandler()]})
        
        # Parse into Plan schema
        model_with_schema = self.llm.with_structured_output(Plan)
        plan = model_with_schema.invoke(("user", "Output the following:\n"+raw_output['output']))

        plan = model_with_schema.invoke(("user", "Output the following:\n" + raw_output["output"]))

        # separate rationale call
        rationale_prompt = (
            "Explain in ≤5 bullets why this plan is reasonable, "
            "based only on the mission and tool results. "
            "Do not output JSON, just text."
            f"\n\nMission: {self.mission_statement}\n"
            f"Plan: {plan.model_dump_json(indent=2)}"
        )
        rationale = self.llm.invoke(rationale_prompt)
        print("[rationale]\n", rationale.content if hasattr(rationale, "content") else rationale)

        return plan.agents


    def replan(self, agents, observations, rewards, terminations, truncations, infos):
        del observations["global"]
        for k, v in observations.items():
            location = tuple(int(x) for x in v["location"])
            self.agent_trajectories[k].append(location)
        stuck = False
        for k, v in self.agent_trajectories.items():
            if v[-1] == v[-2]:
                stuck = True
                break

        if (
            any([r == 1 for r in rewards.values()])
            or any(agents.idle(i) for i in range(self.number_of_agents))
            or stuck
        ):
            if any([r == 1 for r in rewards.values()]):
                print("Re-planning due to found target")
            if any(agents.idle(i) for i in range(self.number_of_agents)):
                print("Re-planning due to agent idle")
            elif stuck:
                print("Re-planning due to stuck")

            # Re-plan when a target is found
            found_targets_agents = [k for k, v in rewards.items() if v == 1]
            found_targets_locations = [
                tuple(int(x) for x in observations[i]["location"])
                for i in found_targets_agents
            ]
            self.found_targets.update(found_targets_locations)
            agent_locations = {
                k: tuple(int(x) for x in v["location"]) for k, v in observations.items()
            }
            # Generate a textual plan
            prompt = create_chat_prompt(
                os.getcwd() + "/prompts/tool_replan.prompty"
            )
            replan_text_planner = prompt | self.llm
            text_plan = replan_text_planner.invoke(
                {
                    "grid_length": self.grid_size,
                    "num_agents": self.number_of_agents,
                    "num_targets": self.number_of_targets,
                    "mission": self.mission_statement,
                    "targets_found": str(self.found_targets),
                    "agent_locations": str(agent_locations),
                }
            ).content
            print(text_plan)
            # Convert the textual plan into structured instructions
            new_plan = self.restructure_text_plan(text_plan)
            for k in new_plan:
                new_plan[k].insert(0, StopAction())
            return new_plan, text_plan

        self.tracker.observe(observations, rewards)
        return {}, ""

    def restructure_text_plan(self, text_plan) -> dict:
        # Convert the textual plan into structured instructions
        prompt = create_chat_prompt(os.getcwd() + "/prompts/plan_structurer.prompty")
        model_with_structure = self.llm.with_structured_output(Plan)
        plan_structurer = prompt | model_with_structure
        plan = plan_structurer.invoke(
            {
                "grid_length": self.grid_size,
                "num_agents": self.number_of_agents,
                "num_targets": self.number_of_targets,
                "mission": self.mission_statement,
                "plan": text_plan,
            }
        )
        return plan.agents