from langchain.agents import initialize_agent, AgentType
from models import local_llm as model
from schemas import Plan
from tools import *
from langchain_prompty import create_chat_prompt
import os
from langchain.schema.runnable import RunnableSequence


# keep your model structured
model_with_structure = model.with_structured_output(Plan)

# list of toolsy
tools = [partition_grid, find_path, validate_plan]

# create agent that can call tools
agent = initialize_agent(
    tools=tools,
    llm=model,
    agent=AgentType.OPENAI_FUNCTIONS,  
)

# agent_with_structure = agent | Plan
prompt = create_chat_prompt(os.getcwd() + "/prompts/initial_planner.prompty")
# initial_tool_planner = prompt | agent | model.with_structured_output(Plan)
initial_tool_planner: RunnableSequence = prompt | agent | model.with_structured_output(Plan)

# initial_tool_planner: RunnableSequence = prompt | (agent | model.with_structured_output(Plan))


print(initial_tool_planner)