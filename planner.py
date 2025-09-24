import os

from langchain_prompty import create_chat_prompt

from models import claude_llm as model
from schemas import Plan

model_with_structure = model.with_structured_output(Plan)
prompt = create_chat_prompt(os.getcwd() + "/prompts/initial_planner.prompty")
initial_planner = prompt | model_with_structure
