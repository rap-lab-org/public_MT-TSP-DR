import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import math

area_size = 100
T = 150
R = math.ceil(math.sqrt(area_size ** 2 + area_size ** 2))

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
nodes = [depot] + targets + [depot_copy]
n_nodes = len(nodes)

edges = []
for i in range(n_nodes):
    for j in range(n_nodes):
        if i!=j and nodes[i]['id']!=depot_copy['id'] and nodes[j]['id']!=depot['id']:
            edges.append((i,j))  # The direct edge from depot to depot_copy can be omitted without affecting the formulation.

model = gp.Model("MT-TSP_bigM")

# ======================
# Variables:

t = {i: model.addVar(lb=nodes[i]['t_window'][0],
                     ub=nodes[i]['t_window'][1], name=f"t_{i}")
     for i in range(n_nodes)}
x_c = {i: model.addVar(lb=-GRB.INFINITY, name=f"x_c_{i}") for i in range(n_nodes)}
y_c = {i: model.addVar(lb=-GRB.INFINITY, name=f"y_c_{i}") for i in range(n_nodes)}

b = {e: model.addVar(vtype=GRB.BINARY, name=f"b_{e[0]}_{e[1]}") for e in edges}

l = {e: model.addVar(lb=0, name=f"l_{e[0]}_{e[1]}") for e in edges}
l_tilde = {e: model.addVar(lb=0, name=f"l_tilde_{e[0]}_{e[1]}") for e in edges}

model.setObjective(gp.quicksum(l_tilde[e] for e in edges), GRB.MINIMIZE)

# ======================
# Constraints:

model.addConstr(gp.quicksum(b[e] for e in edges if e[0]==0)==1, "depot_out")
model.addConstr(gp.quicksum(b[e] for e in edges if e[1]==n_nodes-1)==1, "depot_in")

for k in range(1,n_nodes-1):
    model.addConstr(gp.quicksum(b[e] for e in edges if e[1]==k)==1, f"in_{k}")
    model.addConstr(gp.quicksum(b[e] for e in edges if e[0] == k) ==
                    gp.quicksum(b[e] for e in edges if e[1] == k), f"cons_{k}")

for i in range(n_nodes):
    model.addConstr(x_c[i] == nodes[i]['p0'][0] + nodes[i]['v'][0] * (t[i] - nodes[i]['t_window'][0]))
    model.addConstr(y_c[i] == nodes[i]['p0'][1] + nodes[i]['v'][1] * (t[i] - nodes[i]['t_window'][0]))

for e in edges:
    i,j = e

    model.addConstr((x_c[j] - x_c[i]) * (x_c[j] - x_c[i]) + (y_c[j] - y_c[i]) * (y_c[j] - y_c[i]) <= l[e] * l[e], f"soc_{e[0]}_{e[1]}")
    model.addConstr(l[e] == l_tilde[e] + R * (1 - b[e]), f"l_tilde_{e[0]}_{e[1]}")
    model.addConstr(l_tilde[e] <= v_max * (t[j] - t[i] + T * (1 - b[e])), f"speed_{i}_{j}")

# Solve
model.Params.TimeLimit = 300
model.optimize()

print("\n=== Positive l_tilde values ===")
for e in edges:
    val = l_tilde[e].X
    if val > 1e-6:
        print(f"Edge {e}: l_tilde = {val:.4f}")

print("\n=== Selected edges (b[e] = 1) ===")
for e in edges:
    if b[e].X > 0.5:
        i, j = e
        print(f"Edge {e} | from Node {nodes[i]['id']} to Node {nodes[j]['id']} | b = {b[e].X}")

if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
    def varX(name):
        v = model.getVarByName(name)
        if v is None:
            return None
        try:
            return v.X
        except gp.GurobiError:
            return None

    selected = [e for e in edges if varX(f"b_{e[0]}_{e[1]}") is not None and varX(f"b_{e[0]}_{e[1]}") > 0.5]

    path = [0]
    current = 0
    visited_edges = set()
    while current != n_nodes - 1:
        found = False
        for (i, j) in selected:
            if i == current and (i, j) not in visited_edges:
                path.append(j)
                visited_edges.add((i, j))
                current = j
                found = True
                break
        if not found:
            print("Warning: path reconstruction failed at node", current)
            break

    print("Big-M path:", path)

    path_info = []
    for idx in range(len(path) - 1):
        u = path[idx]
        v = path[idx + 1]
        bval = varX(f"b_{u}_{v}")
        if bval is None or bval <= 0.5:
            continue
        t_u = varX(f"t_{u}")
        t_v = varX(f"t_{v}")
        x_u = varX(f"x_c_{u}")
        y_u = varX(f"y_c_{u}")
        x_v = varX(f"x_c_{v}")
        y_v = varX(f"y_c_{v}")

        if x_u is None or y_u is None:
            if t_u is not None:
                tstart_u = nodes[u]['t_window'][0]
                x_u_calc = nodes[u]['p0'][0] + nodes[u]['v'][0] * (t_u - tstart_u)
                y_u_calc = nodes[u]['p0'][1] + nodes[u]['v'][1] * (t_u - tstart_u)
                x_u, y_u = x_u_calc, y_u_calc
            else:
                x_u, y_u = nodes[u]['p0'][0], nodes[u]['p0'][1]
        if x_v is None or y_v is None:
            if t_v is not None:
                tstart_v = nodes[v]['t_window'][0]
                x_v_calc = nodes[v]['p0'][0] + nodes[v]['v'][0] * (t_v - tstart_v)
                y_v_calc = nodes[v]['p0'][1] + nodes[v]['v'][1] * (t_v - tstart_v)
                x_v, y_v = x_v_calc, y_v_calc
            else:
                x_v, y_v = nodes[v]['p0'][0], nodes[v]['p0'][1]

        if t_u is None: t_u = nodes[u]['t_window'][0]
        if t_v is None: t_v = nodes[v]['t_window'][0]

        path_info.append({
            'from': u,
            'to': v,
            't_start': float(t_u),
            't_end': float(t_v),
            'p_start': (float(x_u), float(y_u)),
            'p_end': (float(x_v), float(y_v))
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

    for seg in path_info:
        plt.plot([seg['p_start'][0], seg['p_end'][0]],
                 [seg['p_start'][1], seg['p_end'][1]], 'r-', linewidth=2)
        plt.annotate('', xy=seg['p_end'], xytext=seg['p_start'],
                     arrowprops=dict(arrowstyle='->', color='red', linewidth=2))

    plt.title(f"MT-TSP: Big-M Based Formulation ($n_{{tar}}$={n_targets})")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True)
    plt.legend()
    static_fname = f"MT-TSP_Big-M_n_tar_{n_targets}_v_{v_max}_t_win_{time_win}_obj_{model.ObjVal:.2f}.jpg"
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
            line, = ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], '--', color=color, alpha=0.3)
            dot, = ax.plot([], [], 'o', color=color, markersize=8, label=f"Target {node['id']}")
            target_lines.append(line)
            target_dots.append(dot)
            ax.text(p_start[0], p_start[1], f"Target {node['id']}")

    ax.set_xlim(-area_size / 2, area_size / 2)
    ax.set_ylim(-area_size / 2, area_size / 2)
    ax.set_title(f"MT-TSP: Big-M Based Formulation ($n_{{tar}}$={n_targets})")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True)
    ax.legend()

    total_time = 0.0
    for seg in path_info:
        if seg['to'] == n_nodes - 1:
            total_time = max(total_time, seg['t_end'])
    if total_time == 0.0:
        all_t = [seg['t_end'] for seg in path_info] + [seg['t_start'] for seg in path_info]
        if all_t:
            total_time = max(all_t)
        else:
            total_time = T

    frames = 100
    time_points = np.linspace(0, total_time, frames)

    agent_positions = []
    path_x, path_y = [], []

    target_positions = [[] for _ in range(len(nodes) - 2)]

    for tnow in time_points:
        cur_pos = None
        for seg in path_info:
            ts, te = seg['t_start'], seg['t_end']
            if te == ts:
                if abs(tnow - ts) < 1e-6:
                    cur_pos = seg['p_end']
            else:
                if ts <= tnow <= te:
                    alpha = (tnow - ts) / (te - ts)
                    x = seg['p_start'][0] + alpha * (seg['p_end'][0] - seg['p_start'][0])
                    y = seg['p_start'][1] + alpha * (seg['p_end'][1] - seg['p_start'][1])
                    cur_pos = (x, y)
                    break
        if cur_pos is None:
            if path_info:
                first_start = path_info[0]['t_start']
                if tnow < first_start:
                    cur_pos = (nodes[0]['p0'][0], nodes[0]['p0'][1])
                else:
                    last = None
                    for seg in path_info:
                        if seg['t_end'] <= tnow:
                            last = seg
                    if last is not None:
                        cur_pos = last['p_end']
                    else:
                        cur_pos = (nodes[0]['p0'][0], nodes[0]['p0'][1])
        agent_positions.append(cur_pos)

        for idx, node in enumerate(nodes[1:-1]):
            t_start, t_end = node['t_window']
            if t_start <= tnow <= t_end:
                p = node['p0'] + node['v'] * (tnow - t_start)
                target_positions[idx].append((float(p[0]), float(p[1])))
            else:
                target_positions[idx].append((None, None))

    cum_x, cum_y = [], []
    for frame_idx, tnow in enumerate(time_points):
        pos = agent_positions[frame_idx]
        if frame_idx == 0:
            cum_x = [pos[0]]
            cum_y = [pos[1]]
        else:
            if pos != (cum_x[-1], cum_y[-1]):
                cum_x.append(pos[0])
                cum_y.append(pos[1])

    def update(frame):
        tnow = time_points[frame]
        pos = agent_positions[frame]

        agent_dot.set_data([pos[0]], [pos[1]])

        xs = [p[0] for p in agent_positions[:frame + 1]]
        ys = [p[1] for p in agent_positions[:frame + 1]]
        path_line.set_data(xs, ys)

        for idx, dot in enumerate(target_dots):
            x, y = target_positions[idx][frame]
            if x is not None and y is not None:
                dot.set_data([x], [y])
                dot.set_alpha(0.7)
            else:
                dot.set_data([], [])

        time_text.set_text(f'Time: {min(tnow, total_time):.1f}/{total_time:.1f}')
        return [agent_dot, path_line, time_text] + target_dots

    ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

    gif_name = f"MT-TSP_Big-M_n_tar_{n_targets}_v_{v_max}_t_win_{time_win}_obj_{model.ObjVal:.2f}.gif"
    gif_path = os.path.join(visualization_dir, gif_name)

    ani.save(gif_path, writer='pillow', fps=10)
    print(f"Animation saved: {gif_path}")
    print(f"model.Runtime: {model.Runtime}")

else:
    print("No solution found")