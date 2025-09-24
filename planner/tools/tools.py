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

@tool("find_path")
def find_path(cur_x: int, cur_y: int, tar_x: int, tar_y: int):
    """Compute Manhattan path from start to target within grid."""
    print(f"🔧 Using tool: find_path(cur_x={cur_x}, cur_y={cur_y}, tar_x={tar_x}, tar_y={tar_y})")
    x, y = int(cur_x), int(cur_y)
    tx, ty = int(tar_x), int(tar_y)
    path: list[list[int]] = []
    while (x, y) != (tx, ty):
        if x < tx: x += 1
        elif x > tx: x -= 1
        elif y < ty: y += 1
        elif y > ty: y -= 1
        path.append([int(x), int(y)])  # list of ints
    return path  # list[list[int]]
