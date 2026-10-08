import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

area_size = 100
T = 150

n_targets = 5
v_max = 8.0
time_win = 50
targets = [{'id': 1, 'p0': np.array([-22.94671667, 19.10413504]), 'v': np.array([0.61020226, -0.90597546]), 't_window': (6.12036967, 56.12036967)},
           {'id': 2, 'p0': np.array([31.37261874, 24.51002961]), 'v': np.array([-0.59455568, 0.50307043]), 't_window': (95.78321681, 145.78321681)},
           {'id': 3, 'p0': np.array([-7.24447602, 8.42896432]), 'v': np.array([0.85131776, 0.55594759]), 't_window': (98.88862739, 148.88862739)},
           {'id': 4, 'p0': np.array([48.20291086, 20.89554138]), 'v': np.array([-0.60936521, -0.58374415]), 't_window': (31.34207882, 81.34207882)},
           {'id': 5, 'p0': np.array([41.15369974, -41.37443202]), 'v': np.array([-0.65703701, 0.60956868]), 't_window': (82.85403649, 132.85403649)}]

depot = {'id': 0, 'p0': np.array([0, 0]), 'v': np.array([0, 0]), 't_window': (0, 0)}
depot_copy = {'id': n_targets + 1, 'p0': np.array([0, 0]), 'v': np.array([0, 0]), 't_window': (0, T)}

model = gp.Model("MT-TSP_GCS")

nodes = [depot] + targets + [depot_copy]
n_nodes = len(nodes)
node_ids = [n['id'] for n in nodes]

edges = []
for i in range(n_nodes):
    for j in range(n_nodes):
        if i != j and nodes[i]['id'] != depot_copy['id'] and nodes[j]['id'] != depot['id']:
            edges.append((i, j))  # The direct edge from depot to depot_copy can be omitted without affecting the formulation.

# ======================
# Variables:

b = {}
alpha = {}
alpha_prime = {}
l_tilde = {}

for e in edges:
    i, j = e
    b[e] = model.addVar(vtype=GRB.BINARY, name=f"b_{i}_{j}")

    alpha[e] = {
        'x': model.addVar(lb=-GRB.INFINITY, name=f"alpha_{i}_{j}_x"),
        'y': model.addVar(lb=-GRB.INFINITY, name=f"alpha_{i}_{j}_y"),
        't': model.addVar(lb=0, name=f"alpha_{i}_{j}_t")
    }

    alpha_prime[e] = {
        'x': model.addVar(lb=-GRB.INFINITY, name=f"alpha_prime_{i}_{j}_x"),
        'y': model.addVar(lb=-GRB.INFINITY, name=f"alpha_prime_{i}_{j}_y"),
        't': model.addVar(lb=0, name=f"alpha_prime_{i}_{j}_t")
    }

    l_tilde[e] = model.addVar(lb=0, name=f"l_tilde_{i}_{j}")

model.setObjective(gp.quicksum(l_tilde[e] for e in edges), GRB.MINIMIZE)

# ======================
# Constraints:

model.addConstr(gp.quicksum(b[e] for e in edges if e[0] == 0) == 1, "depot_out")

model.addConstr(gp.quicksum(b[e] for e in edges if e[1] == n_nodes - 1) == 1, "depot_copy_in")

for k in range(1, n_nodes - 1):
    model.addConstr(gp.quicksum(b[e] for e in edges if e[1] == k) == 1, f"target_{k}_in")
    model.addConstr(gp.quicksum(b[e] for e in edges if e[0] == k) == gp.quicksum(b[e] for e in edges if e[1] == k), f"target_{k}_out")

for k in range(1, n_nodes - 1):
    incoming = [e for e in edges if e[1] == k]
    outgoing = [e for e in edges if e[0] == k]

    model.addConstr(gp.quicksum(alpha_prime[e]['t'] for e in incoming) ==
                    gp.quicksum(alpha[e]['t'] for e in outgoing),
                    f"time_continuity_{k}")

for e in edges:
    model.addConstr(l_tilde[e] ** 2 >= (alpha_prime[e]['x'] - alpha[e]['x']) ** 2 + (alpha_prime[e]['y'] - alpha[e]['y']) ** 2,
                    f"soc_{e[0]}_{e[1]}")

    model.addConstr(l_tilde[e] <= v_max * (alpha_prime[e]['t'] - alpha[e]['t']), f"speed_{e[0]}_{e[1]}")

for e in edges:
    i, j = e
    node_i = nodes[i]
    node_j = nodes[j]

    model.addConstr(alpha[e]['t'] >= b[e] * node_i['t_window'][0], f"time_lb_{i}_{j}")
    model.addConstr(alpha[e]['t'] <= b[e] * node_i['t_window'][1], f"time_ub_{i}_{j}")
    model.addConstr(alpha_prime[e]['t'] >= b[e] * node_j['t_window'][0], f"time_prime_lb_{i}_{j}")
    model.addConstr(alpha_prime[e]['t'] <= b[e] * node_j['t_window'][1], f"time_prime_ub_{i}_{j}")

    model.addConstr(alpha[e]['x'] - node_i['v'][0] * alpha[e]['t'] ==
                    b[e] * (node_i['p0'][0] - node_i['v'][0] * node_i['t_window'][0]),
                    f"perspective_x_{i}_{j}")

    model.addConstr(alpha[e]['y'] - node_i['v'][1] * alpha[e]['t'] ==
                    b[e] * (node_i['p0'][1] - node_i['v'][1] * node_i['t_window'][0]),
                    f"perspective_y_{i}_{j}")

    model.addConstr(alpha_prime[e]['x'] - node_j['v'][0] * alpha_prime[e]['t'] ==
                    b[e] * (node_j['p0'][0] - node_j['v'][0] * node_j['t_window'][0]),
                    f"perspective_prime_x_{i}_{j}")

    model.addConstr(alpha_prime[e]['y'] - node_j['v'][1] * alpha_prime[e]['t'] ==
                    b[e] * (node_j['p0'][1] - node_j['v'][1] * node_j['t_window'][0]),
                    f"perspective_prime_y_{i}_{j}")

# Solve
model.Params.TimeLimit = 300

model.optimize()

print("\n=== Positive l_tilde[e] values ===")
for e in edges:
    val = l_tilde[e].X
    if val > 1e-6:
        i, j = e
        print(f"Edge {e} | from Node {nodes[i]['id']} to Node {nodes[j]['id']} | l_tilde = {val:.4f}")

print("\n=== Selected edges (b[e] = 1) ===")
for e in edges:
    if b[e].X > 0.5:
        i, j = e
        print(f"Edge {e} | from Node {nodes[i]['id']} to Node {nodes[j]['id']} | b = {b[e].X}")

print("\n=== Non-zero alpha[e] values ===")
for e in edges:
    alpha_vals = {dim: alpha[e][dim].X for dim in ['x', 'y', 't']}
    if any(abs(val) > 1e-6 for val in alpha_vals.values()):
        i, j = e
        print(f"Edge {e} | from Node {nodes[i]['id']} to Node {nodes[j]['id']}")
        for dim, val in alpha_vals.items():
            if abs(val) > 1e-6:
                print(f"   alpha[{dim}] = {val:.6f}")

print("\n=== Non-zero alpha_prime[e] values ===")
for e in edges:
    alpha_prime_vals = {dim: alpha_prime[e][dim].X for dim in ['x', 'y', 't']}
    if any(abs(val) > 1e-6 for val in alpha_prime_vals.values()):
        i, j = e
        print(f"Edge {e} | from Node {nodes[i]['id']} to Node {nodes[j]['id']}")
        for dim, val in alpha_prime_vals.items():
            if abs(val) > 1e-6:
                print(f"   alpha_prime[{dim}] = {val:.6f}")

if model.status == GRB.OPTIMAL:
    print(f"model.Runtime: {model.Runtime}")

    selected_edges = [e for e in edges if b[e].X > 0.5]

    path = []
    current = 0
    path.append(current)

    while current != n_nodes - 1:
        for e in selected_edges:
            if e[0] == current:
                next_node = e[1]
                path.append(next_node)
                current = next_node
                break

    print("GCS path:", path)

    path_info = []
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        e = (u, v)
        if b[e].X > 0.5:
            t_u = alpha[e]['t'].X / b[e].X
            t_v = alpha_prime[e]['t'].X / b[e].X
            p_u = (alpha[e]['x'].X / b[e].X, alpha[e]['y'].X / b[e].X)
            p_v = (alpha_prime[e]['x'].X / b[e].X, alpha_prime[e]['y'].X / b[e].X)
            path_info.append({
                'from': u,
                'to': v,
                't_start': t_u,
                't_end': t_v,
                'p_start': p_u,
                'p_end': p_v
            })

    visualization_dir = "Visualization"
    os.makedirs(visualization_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.xlim(-area_size / 2, area_size / 2)
    plt.ylim(-area_size / 2, area_size / 2)

    for node in nodes:
        if node['id'] == 0:
            plt.plot(node['p0'][0], node['p0'][1], 'ks', markersize=10, label='Depot')
        elif node['id'] == n_nodes - 1:
            plt.plot(node['p0'][0], node['p0'][1], 'ko', markersize=10, label='Depot Copy')
        else:
            t_start, t_end = node['t_window']
            p_start = node['p0'] + node['v'] * (t_start - node['t_window'][0])
            p_end = node['p0'] + node['v'] * (t_end - node['t_window'][0])
            plt.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], '--', alpha=0.3)
            plt.text(p_start[0], p_start[1], f"Target {node['id']}")

    for segment in path_info:
        plt.plot([segment['p_start'][0], segment['p_end'][0]],
                 [segment['p_start'][1], segment['p_end'][1]],
                 'r-', linewidth=2)
        plt.annotate('', xy=segment['p_end'], xytext=segment['p_start'],
                     arrowprops=dict(arrowstyle='->', color='red', linewidth=2))

    plt.title(f"MT-TSP: GCS Based Formulation ($n_{{tar}}$={n_targets})")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True)
    plt.legend()

    static_fname = f"MT-TSP_GCS_n_tar_{n_targets}_v_{v_max}_t_win_{time_win}_obj_{model.ObjVal:.2f}.jpg"
    static_path = os.path.join(visualization_dir, static_fname)
    plt.savefig(static_path, dpi=300)
    plt.show()
    print(f"Static figure saved: {static_path}")

    fig, ax = plt.subplots(figsize=(10, 8))

    agent_dot, = ax.plot([], [], 'bo', markersize=10, label='Agent')
    path_line, = ax.plot([], [], 'r-', linewidth=2)
    time_text = ax.text(0.98, 0.95, '', transform=ax.transAxes,
                        ha='right',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    target_lines = []
    target_dots = []
    target_colors = plt.cm.tab10.colors

    for i, node in enumerate(nodes):
        if node['id'] == 0:
            ax.plot(node['p0'][0], node['p0'][1], 'ks', markersize=10, label='Depot')
        elif node['id'] == n_nodes - 1:
            ax.plot(node['p0'][0], node['p0'][1], 'ko', markersize=10, label='Depot Copy')
        else:
            color = target_colors[i % len(target_colors)]
            t_start, t_end = node['t_window']
            p_start = node['p0'] + node['v'] * (t_start - node['t_window'][0])
            p_end = node['p0'] + node['v'] * (t_end - node['t_window'][0])
            line, = ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], '--',
                            color=color, alpha=0.3)
            dot, = ax.plot([], [], 'o', color=color, markersize=8,
                           label=f"Target {node['id']}")
            target_lines.append(line)
            target_dots.append(dot)
            ax.text(p_start[0], p_start[1], f"Target {node['id']}")

    ax.set_xlim(-area_size / 2, area_size / 2)
    ax.set_ylim(-area_size / 2, area_size / 2)
    ax.set_title(f"MT-TSP: GCS Based Formulation ($n_{{tar}}$={n_targets})")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True)
    ax.legend()

    total_time = 0
    for e in selected_edges:
        if e[1] == n_nodes - 1:
            total_time = alpha_prime[e]['t'].X / b[e].X
            break
    print("total_time:", total_time)

    frames = 100
    time_points = np.linspace(0, total_time, frames)

    agent_positions = []
    target_positions = [[] for _ in range(len(nodes))]
    path_x, path_y = [], []

    for t in time_points:
        current_pos = None
        for segment in path_info:
            if segment['t_start'] <= t <= segment['t_end']:
                alpha_interp = (t - segment['t_start']) / (segment['t_end'] - segment['t_start'])
                x = segment['p_start'][0] + alpha_interp * (segment['p_end'][0] - segment['p_start'][0])
                y = segment['p_start'][1] + alpha_interp * (segment['p_end'][1] - segment['p_start'][1])
                current_pos = (x, y)
                break
        if current_pos:
            path_x.append(current_pos[0])
            path_y.append(current_pos[1])
            agent_positions.append(current_pos)
        else:
            agent_positions.append((0, 0))

        for i, node in enumerate(nodes[1:-1]):
            t_start, t_end = node['t_window']
            if t_start <= t <= t_end:
                p = node['p0'] + node['v'] * (t - t_start)
                target_positions[i].append((p[0], p[1]))
            else:
                target_positions[i].append((None, None))

    def update(frame):
        t = time_points[frame]
        pos = agent_positions[frame]

        agent_dot.set_data([pos[0]], [pos[1]])

        path_line.set_data(path_x[:frame + 1], path_y[:frame + 1])

        for i, dot in enumerate(target_dots):
            x, y = target_positions[i][frame]
            if x is not None and y is not None:
                dot.set_data([x], [y])
                dot.set_alpha(0.7)
            else:
                dot.set_data([], [])

        time_text.set_text(f'Time: {min(t, total_time):.1f}/{total_time:.1f}')

        return [agent_dot, path_line, time_text] + target_dots

    ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

    gif_name = f"MT-TSP_GCS_n_tar_{n_targets}_v_{v_max}_t_win_{time_win}_obj_{model.ObjVal:.2f}.gif"
    gif_path = os.path.join(visualization_dir, gif_name)

    ani.save(gif_path, writer='pillow', fps=10)

    print(f"Animation saved: {gif_path}")

else:
    print("No optimal solution found")