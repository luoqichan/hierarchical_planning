# planner/tools/tools.py
from langchain.tools import tool

@tool("partition_grid")
def partition_grid(grid_length: int, num_agents: int):
    """Split an N x N grid into num_agents rectangular regions."""
    print(f"🔧 Using tool: partition_grid(grid_length={grid_length}, num_agents={num_agents})")
    # always use built-in ints
    step = int(grid_length) // int(num_agents)
    regions = []
    for i in range(int(num_agents)):
        x1, y1 = 1, i * step + 1
        x2, y2 = int(grid_length), (i + 1) * step if i < int(num_agents) - 1 else int(grid_length)
        regions.append({"region": [int(x1), int(y1), int(x2), int(y2)]})  # list, not tuple
    return regions  # list[dict]

import re
from typing import List, Dict, Tuple

@tool("partition_grid_with_mission")
def partition_grid_with_mission(
    grid_length: int,
    num_agents: int,
    mission: str
) -> List[Dict[str, Tuple[int,int,int,int]]]:
    """
    Partition the grid among agents using mission hints if available.

    - Extract subregions mentioned in the mission (e.g. "from (x1,y1) to (x2,y2)").
    - If #subregions <= num_agents: assign one agent per subregion, rest get fallback coverage.
    - If #subregions > num_agents: merge/simplify subregions until #agents matches.
    - If no subregions found: fall back to uniform partition of the grid.
    
    Returns a list of region dicts like:
    [{"region": (x1, y1, x2, y2)}, ...]
    """
    print(f"🔧 Tool called: partition_grid_with_mission(grid_length={grid_length}, num_agents={num_agents}, mission={mission})")

    # regex to extract coordinates (x1, y1, x2, y2)
    pattern = r"\((\d+),\s*(\d+)\)\s*to\s*\((\d+),\s*(\d+)\)"
    matches = re.findall(pattern, mission)

    regions = []
    for m in matches:
        x1, y1, x2, y2 = map(int, m)
        regions.append({"region": (x1, y1, x2, y2)})

    if not regions:
        # fallback uniform partition
        step = grid_length // num_agents
        for i in range(num_agents):
            x1, y1 = 1, i * step + 1
            x2, y2 = grid_length, (i + 1) * step if i < num_agents - 1 else grid_length
            regions.append({"region": (x1, y1, x2, y2)})
        return regions

    # If more subregions than agents → merge
    if len(regions) > num_agents:
        merged = []
        group_size = len(regions) // num_agents
        for i in range(num_agents):
            group = regions[i*group_size : (i+1)*group_size]
            if not group:
                continue
            # merge bounding box
            xs = [r["region"][0] for r in group] + [r["region"][2] for r in group]
            ys = [r["region"][1] for r in group] + [r["region"][3] for r in group]
            merged.append({"region": (min(xs), min(ys), max(xs), max(ys))})
        regions = merged

    # If fewer subregions than agents → assign extras to cover the rest of the grid
    if len(regions) < num_agents:
        covered = []
        for r in regions:
            covered.append(r["region"])
        # naive: assign leftover agents to uniform partitions of uncovered space
        # (for now, just reuse the full grid for extras)
        while len(regions) < num_agents:
            regions.append({"region": (1, 1, grid_length, grid_length)})

    return regions


from typing import List, Dict, Tuple

@tool("assign_agents_to_regions")
def assign_agents_to_regions(
    agent_positions: Dict[int, Tuple[int, int]],
    regions: List[Dict[str, Tuple[int, int, int, int]]]
) -> Dict[int, Dict[str, Tuple[int, int, int, int]]]:
    """
    Assign each agent to one region, preferring closest regions by Manhattan distance.

    Inputs:
    - agent_positions: {agent_id: (x, y)}
    - regions: [{"region": (x1,y1,x2,y2)}, ...]

    Returns:
    - {agent_id: {"region": (x1,y1,x2,y2)}}
    """
    print(f"🔧 Tool called: assign_agents_to_regions(agent_positions={agent_positions}, regions={regions})")

    assignments = {}
    available = regions.copy()

    for aid, (ax, ay) in agent_positions.items():
        if not available:
            break
        # pick nearest region by distance to its center
        dists = []
        for r in available:
            x1, y1, x2, y2 = r["region"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            dist = abs(ax - cx) + abs(ay - cy)
            dists.append((dist, r))
        dists.sort(key=lambda x: x[0])
        best_region = dists[0][1]
        assignments[aid] = best_region
        available.remove(best_region)

    # if more agents than regions → assign extras to the last region
    if available == [] and len(assignments) < len(agent_positions):
        last_region = list(assignments.values())[-1]
        for aid in agent_positions.keys():
            if aid not in assignments:
                assignments[aid] = last_region

    return assignments


@tool("compute_expected_reward")
def compute_expected_reward(
    found_targets: int,
    total_targets: int,
    steps_taken: int,
    Tmax: int,
    gamma: float = 0.99,
    M: int = 1
):
    """
    Compute the expected cumulative reward for a trajectory,
    given number of found targets, total targets, steps taken, Tmax, and gamma.
    """
    print(f"🔧 Tool called: compute_expected_reward(found={found_targets}, total={total_targets}, steps={steps_taken}, Tmax={Tmax})")

    # Per-timestep reward (simplified for expected calculation):
    # if an agent finds a target: +1, else -1
    # collective reward is mean across M agents
    # Here we simulate: assume rewards_t ~ (found/total) success ratio
    base_reward = (2*found_targets - total_targets) / total_targets

    # discounted return
    Rt = sum([ (gamma**t) * base_reward for t in range(steps_taken) ])

    # Bonus term
    if found_targets == total_targets:
        B = (2 - steps_taken/Tmax) * (1/(1-gamma))
    else:
        B = 0

    return Rt + B


# tools/reward_tools.py
from typing import Dict, List, Optional
from langchain.tools import tool
import math

def _geom_sum(gamma: float, T: int) -> float:
    """Sum_{t=0}^{T-1} gamma^t = (1 - gamma^T)/(1 - gamma)"""
    return (1.0 - (gamma ** T)) / (1.0 - gamma) if T > 0 else 0.0

@tool("evaluate_candidate_return")
def evaluate_candidate_return(
    travel_steps: Dict[int, int],
    search_cells: Dict[int, int],
    remaining_targets: int,
    Tmax: int,
    gamma: float,
    M: int,
    # Optional: if you know which agent is assigned to a guaranteed target region, name them here.
    # Otherwise we assume exactly `remaining_targets` of the agents are searching target-bearing regions,
    # and take the min expected discovery time among those.
    guaranteed_target_agents: Optional[List[int]] = None
) -> float:
    """
    Evaluate expected discounted return for ONE remaining target using your reward function.

    Assumptions (simple, robust for hackathon):
    - One target remains to be found (set `remaining_targets=1`). If >1, call this tool separately per target or extend logic.
    - Agent i needs `travel_steps[i]` orthogonal steps to reach its start corner.
    - Agent i's search covers `search_cells[i]` cells via snake traversal (1 cell/step).
    - Expected discovery time for agent i scanning a target-bearing region is:
        t_i = travel_steps[i] + (search_cells[i] + 1)/2
      (uniform target location model in that region).
    - If multiple agents scan the same target-bearing region (or disjoint parts of it),
      the expected time-to-first-hit is min_i t_i (we approximate using the min of expectations).
    - Per-timestep team reward r_t = mean over agents: +1 for the agent that finds at its hit step, -1 for all others.
    - Episode ends immediately when the (last) remaining target is found; then bonus B applies:
        B = (2 - T_finish / Tmax) * 1/(1 - gamma)

    Inputs:
    - travel_steps: {agent_id: int}
    - search_cells: {agent_id: int}
    - remaining_targets: int (use 1 for this tool)
    - Tmax: int (use N*N or your chosen cap)
    - gamma: float (e.g., 0.99)
    - M: int (number of agents)
    - guaranteed_target_agents: optional list of agent_ids believed to be scanning the true target region

    Returns:
    - Expected discounted return (float)
    """
    # Guard
    if remaining_targets <= 0:
        # Nothing left to find → return only the (degenerate) bonus if you consider episode ended earlier.
        # We return 0 here; let the caller handle end-of-episode.
        return 0.0

    # Build per-agent expected hit time
    t_hits = {}
    for aid, travel in travel_steps.items():
        s = max(0, int(search_cells.get(aid, 0)))
        t_hits[aid] = float(int(travel)) + (s + 1.0) / 2.0  # E[hit] in steps

    # Choose which agents are truly "searching target-bearing region"
    if guaranteed_target_agents:
        candidates = [t_hits[aid] for aid in guaranteed_target_agents if aid in t_hits]
        if not candidates:
            # fall back if ids don't match
            candidates = list(t_hits.values())
    else:
        # Assume all provided agents are contenders (caller should pass only searching agents)
        candidates = list(t_hits.values())

    if not candidates:
        # No one searching? Episode cannot finish → negative tail, but we cap at Tmax:
        # r_t = -1 for all agents forever; discounted sum ≈ -_geom_sum(gamma, Tmax)
        neg = -_geom_sum(gamma, Tmax)
        return neg

    # Expected first discovery time (approx): min of expected times (simple but effective)
    t_first = min(candidates)
    T_finish = int(math.ceil(t_first))

    # Per-step team reward:
    # For t < T_finish: all agents did NOT find target → each gets -1 → mean r_t = -1
    # At t = T_finish: exactly one agent finds (in expectation) → r_t = ((+1) + (M-1)*(-1))/M = (2-M)/M
    # Episode ends at T_finish (since 1 target remains). Then add bonus B based on T_finish.
    prefix = -_geom_sum(gamma, T_finish)                    # sum_{t=0}^{T_finish-1} gamma^t * (-1)
    hit_step = (gamma ** T_finish) * ((2.0 - M) / M)        # reward at the hit moment
    B = (2.0 - (T_finish / float(Tmax))) * (1.0 / (1.0 - gamma))

    return prefix + hit_step + B


def _parse_mission_regions(mission: str):
    """
    Extract rectangular regions from mission text.
    Expects patterns like: "from (x1, y1) to (x2, y2)".
    Returns list of (x1, y1, x2, y2).
    """
    pattern = r"\((\d+),\s*(\d+)\)\s*to\s*\((\d+),\s*(\d+)\)"
    matches = re.findall(pattern, mission)
    regions = []
    for (x1, y1, x2, y2) in matches:
        regions.append((int(x1), int(y1), int(x2), int(y2)))
    return regions


@tool("greedy_baseline_plan")
def greedy_baseline_plan(
    grid_length: int, 
    num_agents: int, 
    mission: str
) -> Dict[int, List[dict]]:
    """
    Generate a baseline plan guided by mission hints.

    - Parse mission to find subspaces likely containing targets.
    - If enough regions exist, assign one agent per region (greedy).
    - If fewer regions than agents, remaining agents cover leftover strips.
    - Each agent moves orthogonally from (1,1) to the top-left corner of its region,
      then issues a `search` covering the entire rectangle.
    - If no regions are found in the mission, fall back to naive partition (horizontal strips).
    """
    regions = _parse_mission_regions(mission)
    agents_plan = {}

    # Case 1: Mission has usable regions
    if regions:
        assigned = 0
        for i, region in enumerate(regions):
            if assigned >= num_agents:
                break
            x1, y1, x2, y2 = region

            move_action = {
                "type": "move",
                "cur_x": 1,
                "cur_y": 1,
                "tar_x": x1,
                "tar_y": y1
            }
            search_action = {
                "type": "search",
                "cur_x": x1,
                "cur_y": y1,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }
            agents_plan[assigned] = [move_action, search_action]
            assigned += 1

        # Fill leftover agents with fallback coverage
        if assigned < num_agents:
            step = grid_length // num_agents
            for j in range(assigned, num_agents):
                y1 = j * step + 1
                y2 = (j + 1) * step if j < num_agents - 1 else grid_length
                move_action = {
                    "type": "move",
                    "cur_x": 1,
                    "cur_y": 1,
                    "tar_x": 1,
                    "tar_y": y1
                }
                search_action = {
                    "type": "search",
                    "cur_x": 1,
                    "cur_y": y1,
                    "x1": 1,
                    "y1": y1,
                    "x2": grid_length,
                    "y2": y2
                }
                agents_plan[j] = [move_action, search_action]

    # Case 2: No regions in mission → fallback naive strip partition
    else:
        step = grid_length // num_agents
        for i in range(num_agents):
            y1 = i * step + 1
            y2 = (i + 1) * step if i < num_agents - 1 else grid_length
            move_action = {
                "type": "move",
                "cur_x": 1,
                "cur_y": 1,
                "tar_x": 1,
                "tar_y": y1
            }
            search_action = {
                "type": "search",
                "cur_x": 1,
                "cur_y": y1,
                "x1": 1,
                "y1": y1,
                "x2": grid_length,
                "y2": y2
            }
            agents_plan[i] = [move_action, search_action]

    return {"agents": agents_plan}
# tools/reward_tools.py
from langchain.tools import tool
import math

@tool("compute_true_reward", return_direct=True)
def compute_true_reward(
    found_targets: int,
    total_targets: int,
    steps_taken: int,
    Tmax: int,
    gamma: float,
    M: int,
) -> float:
    """
    Compute the true expected cumulative reward for a trajectory.
    Uses the formal definition from the environment spec.

    R(τ) = Σ_t γ^t r_t + B
      - r_t = (1/M) Σ_m r^m_t, where r^m_t = +1 if agent found a target, else -1
      - Here we approximate: each step until a new target is found gives -1 for all agents.
      - Bonus B applies if all targets are found.
    """
    # Approximate reward accumulation
    if found_targets >= total_targets:
        B = (2 - steps_taken / Tmax) * (1 / (1 - gamma))
    else:
        B = 0

    # Simplified per-step reward trajectory
    # Each step = -1 if not found yet, +1 if found
    base_reward = ((found_targets / total_targets) * 1) + (
        (1 - found_targets / total_targets) * -1
    )
    discounted = sum([gamma**t * base_reward for t in range(steps_taken)])

    return discounted + B



@tool("compare_with_baseline", return_direct=True)
def compare_with_baseline(
    grid_length: int,
    num_agents: int,
    num_targets: int,
    mission: str,
    llm_plan: dict,
    Tmax: int = None,
    gamma: float = 0.99,
) -> dict:
    """
    Compare the LLM's proposed plan against the greedy baseline.
    Returns both plans and their evaluated rewards.
    """
    if Tmax is None:
        Tmax = grid_length * grid_length  # default cap

    # Generate greedy baseline plan
    baseline_plan = greedy_baseline_plan(grid_length=grid_length, num_agents=num_agents, mission=mission)

    # Approximate steps taken as "number of actions" per agent (very rough)
    def estimate_steps(plan):
        steps = 0
        for acts in plan["agents"].values():
            for a in acts:
                if a["type"] == "move":
                    steps += abs(a["cur_x"] - a["tar_x"]) + abs(a["cur_y"] - a["tar_y"])
                elif a["type"] == "search":
                    steps += (a["x2"] - a["x1"] + 1) * (a["y2"] - a["y1"] + 1)
        return steps

    baseline_steps = estimate_steps(baseline_plan)
    llm_steps = estimate_steps(llm_plan)

    # Compute rewards
    baseline_reward = compute_true_reward.invoke({
        "found_targets": 0,  # assume no targets at start
        "total_targets": num_targets,
        "steps_taken": baseline_steps,
        "Tmax": Tmax,
        "gamma": gamma,
        "M": num_agents,
    })

    llm_reward = compute_true_reward.invoke({
        "found_targets": 0,
        "total_targets": num_targets,
        "steps_taken": llm_steps,
        "Tmax": Tmax,
        "gamma": gamma,
        "M": num_agents,
    })

    return {
        "baseline_plan": baseline_plan,
        "baseline_reward": baseline_reward,
        "llm_plan": llm_plan,
        "llm_reward": llm_reward,
        "winner": "baseline" if baseline_reward > llm_reward else "llm"
    }
