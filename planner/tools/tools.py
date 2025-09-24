from langchain.tools import tool

@tool
def partition_grid(grid_length: int, num_agents: int):
    """Split grid into subregions for each agent."""
    step = grid_length // num_agents
    regions = []
    for i in range(num_agents):
        x1, y1 = 1, i*step+1
        x2, y2 = grid_length, (i+1)*step if i < num_agents-1 else grid_length
        regions.append({"region": (x1,y1,x2,y2)})
    return regions

@tool
def find_path(cur_x: int, cur_y: int, tar_x: int, tar_y: int):
    """Compute Manhattan path from start to target."""
    path = []
    x, y = cur_x, cur_y
    while (x,y) != (tar_x, tar_y):
        if x < tar_x: x += 1
        elif x > tar_x: x -= 1
        elif y < tar_y: y += 1
        elif y > tar_y: y -= 1
        path.append((x,y))
    return path

@tool
def validate_plan(plan: dict, grid_length: int):
    """Check bounds of plan regions."""
    for agent, actions in plan["agents"].items():
        for act in actions:
            if act["type"] == "search":
                x1,y1,x2,y2 = act["x1"], act["y1"], act["x2"], act["y2"]
                if not (1 <= x1 <= x2 <= grid_length and 1 <= y1 <= y2 <= grid_length):
                    return {"valid": False}
    return {"valid": True}
