from collections import defaultdict, deque

# 3-hop coloring
def get_neighbor_charts(chart_grid, i, j, hop=5):
    height, width = len(chart_grid), len(chart_grid[0])
    queue = deque([(i, j, 0)])
    visited = set([(i, j)])
    neighbors = set()
    while queue:
        x, y, d = queue.popleft()
        if d > hop:
            break
        if (x, y) != (i, j):
            neighbors.add(chart_grid[x][y])
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < height and 0 <= ny < width and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny, d + 1))
    return neighbors

def color_charts(input_data):
    # Parse input
    height, width = input_data[0], input_data[1]
    grid = [input_data[i:i+width] for i in range(2, len(input_data), width)]
    assert len(input_data) == 2 + height * width, f"{len(input_data)}, {height}, {width}"

    # Identify and label charts
    chart_id = 0
    chart_grid = [[None for _ in range(width)] for _ in range(height)]
    chart_sizes = defaultdict(int)

    visited = set()
    def dfs(start_i, start_j, value):
        stack = [(start_i, start_j)]
        while stack:
            i, j = stack.pop()
            if 0 <= i < height and 0 <= j < width and grid[i][j] == value and (i, j) not in visited:
                visited.add((i, j))
                chart_grid[i][j] = chart_id
                chart_sizes[chart_id] += 1
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    stack.append((i + di, j + dj))

    for i in range(height):
        for j in range(width):
            if chart_grid[i][j] is None:
                dfs(i, j, grid[i][j])
                chart_id += 1

    # Sort charts by size
    sorted_charts = sorted(chart_sizes.items(), key=lambda x: (-x[1], x[0]))

    color_assignment = {}
    for chart, chart_size in sorted_charts:
        used_colors = set()
        for i in range(height):
            for j in range(width):
                if chart_grid[i][j] == chart:
                    #used_colors.update(color_assignment.get(n) for n in get_neighbor_charts(chart_grid, i, j) if n in color_assignment)
                    used_colors.update(color_assignment.get(n, -1) for n in get_neighbor_charts(chart_grid, i, j))
                    break
            if used_colors:
                break
        color = min(set(range(len(sorted_charts))) - used_colors)
        # print(f'chart {chart} ({chart_size}) assign color {color} for used_colors {used_colors}')
        color_assignment[chart] = color

    # Create output grid
    output_grid = [[color_assignment[chart_grid[i][j]] for j in range(width)] for i in range(height)]
    return [item for sublist in output_grid for item in sublist]
