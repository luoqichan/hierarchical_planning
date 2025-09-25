# planner/tools/tools.py
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel
from langchain.tools import tool
import re, math

# -------------------------------
# partition_grid
# -------------------------------
class PartitionArgs(BaseModel):
    grid_length: int
    num_agents: int

@tool("partition_grid", args_schema=PartitionArgs, return_direct=True)
def partition_grid(grid_length: int, num_agents: int):
    """Split an N x N grid into num_agents rectangular regions."""
    print(f"🔧 Using tool: partition_grid(grid_length={grid_length}, num_agents={num_agents})")
    step = int(grid_length) // int(num_agents)
    regions = []
    for i in range(int(num_agents)):
        x1, y1 = 1, i * step + 1
        x2, y2 = int(grid_length), (i + 1) * step if i < int(num_agents) - 1 else int(grid_length)
        regions.append({"region": [int(x1), int(y1), int(x2), int(y2)]})
    return regions

# -------------------------------
# partition_grid_with_mission
# -------------------------------
class PartitionMissionArgs(BaseModel):
    grid_length: int
    num_agents: int
    mission: str

@tool("partition_grid_with_mission", args_schema=PartitionMissionArgs, return_direct=True)
def partition_grid_with_mission(grid_length: int, num_agents: int, mission: str):
    """
    Partition the grid among agents using mission hints if available.
    """
    print(f"🔧 Tool called: partition_grid_with_mission(grid_length={grid_length}, num_agents={num_agents}, mission={mission})")
    pattern = r"\((\d+),\s*(\d+)\)\s*to\s*\((\d+),\s*(\d+)\)"
    matches = re.findall(pattern, mission)
    regions = []
    for m in matches:
        x1, y1, x2, y2 = map(int, m)
        regions.append({"region": (x1, y1, x2, y2)})

    if not regions:
        step = grid_length // num_agents
        for i in range(num_agents):
            x1, y1 = 1, i * step + 1
            x2, y2 = grid_length, (i + 1) * step if i < num_agents - 1 else grid_length
            regions.append({"region": (x1, y1, x2, y2)})
        return regions

    # merge if more than agents
    if len(regions) > num_agents:
        merged = []
        group_size = len(regions) // num_agents
        for i in range(num_agents):
            group = regions[i*group_size : (i+1)*group_size]
            xs = [r["region"][0] for r in group] + [r["region"][2] for r in group]
            ys = [r["region"][1] for r in group] + [r["region"][3] for r in group]
            merged.append({"region": (min(xs), min(ys), max(xs), max(ys))})
        regions = merged

    # pad if fewer
    while len(regions) < num_agents:
        regions.append({"region": (1, 1, grid_length, grid_length)})
    return regions

# -------------------------------
# compute_expected_reward
# -------------------------------
class ExpectedRewardArgs(BaseModel):
    found_targets: int
    total_targets: int
    steps_taken: int
    Tmax: int
    gamma: float = 0.99
    M: int = 1

@tool("compute_expected_reward", args_schema=ExpectedRewardArgs, return_direct=True)
def compute_expected_reward(found_targets: int, total_targets: int, steps_taken: int, Tmax: int, gamma: float = 0.99, M: int = 1):
    """Compute expected cumulative reward."""
    print(f"🔧 Tool called: compute_expected_reward(found={found_targets}, total={total_targets}, steps={steps_taken}, Tmax={Tmax})")
    base_reward = (2*found_targets - total_targets) / total_targets
    Rt = sum([(gamma**t) * base_reward for t in range(steps_taken)])
    B = (2 - steps_taken/Tmax) * (1/(1-gamma)) if found_targets == total_targets else 0
    return Rt + B

# -------------------------------
# greedy_baseline_plan
# -------------------------------
class GreedyArgs(BaseModel):
    grid_length: int
    num_agents: int
    mission: str

def _parse_mission_regions(mission: str):
    pattern = r"\((\d+),\s*(\d+)\)\s*to\s*\((\d+),\s*(\d+)\)"
    return [(int(a), int(b), int(c), int(d)) for (a,b,c,d) in re.findall(pattern, mission)]

@tool("greedy_baseline_plan", args_schema=GreedyArgs, return_direct=True)
def greedy_baseline_plan(grid_length: int, num_agents: int, mission: str):
    """Generate a greedy baseline plan using mission hints."""
    regions = _parse_mission_regions(mission)
    agents_plan = {}
    if regions:
        assigned = 0
        for region in regions:
            if assigned >= num_agents: break
            x1,y1,x2,y2 = region
            agents_plan[assigned] = [
                {"type": "move", "cur_x": 1, "cur_y": 1, "tar_x": x1, "tar_y": y1},
                {"type": "search", "cur_x": x1, "cur_y": y1, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
            ]
            assigned += 1
    if not regions or len(agents_plan) < num_agents:
        step = grid_length // num_agents
        for j in range(len(agents_plan), num_agents):
            y1 = j*step+1
            y2 = (j+1)*step if j < num_agents-1 else grid_length
            agents_plan[j] = [
                {"type": "move", "cur_x": 1, "cur_y": 1, "tar_x": 1, "tar_y": y1},
                {"type": "search", "cur_x": 1, "cur_y": y1, "x1": 1, "y1": y1, "x2": grid_length, "y2": y2}
            ]
    return {"agents": agents_plan}

# -------------------------------
# compute_true_reward
# -------------------------------
class TrueRewardArgs(BaseModel):
    found_targets: int
    total_targets: int
    steps_taken: int
    Tmax: int
    gamma: float
    M: int

@tool("compute_true_reward", args_schema=TrueRewardArgs, return_direct=True)
def compute_true_reward(found_targets: int, total_targets: int, steps_taken: int, Tmax: int, gamma: float, M: int):
    """Compute the true expected cumulative reward."""
    if found_targets >= total_targets:
        B = (2 - steps_taken / Tmax) * (1 / (1 - gamma))
    else:
        B = 0
    base_reward = ((found_targets / total_targets) * 1) + ((1 - found_targets / total_targets) * -1)
    discounted = sum([gamma**t * base_reward for t in range(steps_taken)])
    return discounted + B

# -------------------------------
# evaluate_candidate_return
# -------------------------------
class CandidateReturnArgs(BaseModel):
    travel_steps: Dict[int,int]
    search_cells: Dict[int,int]
    remaining_targets: int
    Tmax: int
    gamma: float
    M: int
    guaranteed_target_agents: Optional[List[int]] = None

def _geom_sum(gamma: float, T: int) -> float:
    return (1.0 - (gamma ** T)) / (1.0 - gamma) if T > 0 else 0.0

@tool("evaluate_candidate_return", args_schema=CandidateReturnArgs, return_direct=True)
def evaluate_candidate_return(travel_steps: Dict[int,int], search_cells: Dict[int,int], remaining_targets: int, Tmax: int, gamma: float, M: int, guaranteed_target_agents: Optional[List[int]] = None) -> float:
    """Evaluate expected discounted return for a candidate plan."""
    if remaining_targets <= 0:
        return 0.0
    t_hits = {}
    for aid, travel in travel_steps.items():
        s = max(0, int(search_cells.get(aid, 0)))
        t_hits[aid] = float(int(travel)) + (s + 1.0)/2.0
    candidates = []
    if guaranteed_target_agents:
        candidates = [t_hits[aid] for aid in guaranteed_target_agents if aid in t_hits]
    if not candidates:
        candidates = list(t_hits.values())
    if not candidates:
        return -_geom_sum(gamma, Tmax)
    t_first = min(candidates)
    T_finish = int(math.ceil(t_first))
    prefix = -_geom_sum(gamma, T_finish)
    hit_step = (gamma**T_finish) * ((2.0 - M)/M)
    B = (2.0 - (T_finish/float(Tmax))) * (1.0/(1.0-gamma))
    return prefix + hit_step + B

# -------------------------------
# compare_with_baseline
# -------------------------------
class CompareArgs(BaseModel):
    grid_length: int
    num_agents: int
    num_targets: int
    mission: str
    llm_plan: dict
    Tmax: Optional[int] = None
    gamma: float = 0.99

@tool("compare_with_baseline", args_schema=CompareArgs, return_direct=True)
def compare_with_baseline(grid_length: int, num_agents: int, num_targets: int, mission: str, llm_plan: dict, Tmax: int = None, gamma: float = 0.99):
    """Compare LLM plan against greedy baseline."""
    if Tmax is None: Tmax = grid_length * grid_length
    baseline_plan = greedy_baseline_plan(grid_length=grid_length, num_agents=num_agents, mission=mission)
    def estimate_steps(plan):
        steps = 0
        for acts in plan["agents"].values():
            for a in acts:
                if a["type"]=="move":
                    steps += abs(a["cur_x"]-a["tar_x"]) + abs(a["cur_y"]-a["tar_y"])
                elif a["type"]=="search":
                    steps += (a["x2"]-a["x1"]+1)*(a["y2"]-a["y1"]+1)
        return steps
    baseline_steps = estimate_steps(baseline_plan)
    llm_steps = estimate_steps(llm_plan)
    baseline_reward = compute_expected_reward(found_targets=0, total_targets=num_targets, steps_taken=baseline_steps, Tmax=Tmax, gamma=gamma, M=num_agents)
    llm_reward = compute_expected_reward(found_targets=0, total_targets=num_targets, steps_taken=llm_steps, Tmax=Tmax, gamma=gamma, M=num_agents)
    return {"baseline_plan": baseline_plan, "baseline_reward": baseline_reward, "llm_plan": llm_plan, "llm_reward": llm_reward, "winner": "baseline" if baseline_reward>llm_reward else "llm"}
