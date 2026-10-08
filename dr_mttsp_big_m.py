import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import os

area_size = 100
ws_z = 40
T = 150
k_r = 0.5
z0 = 20.0

n_targets = 5
v_max = 8.0
time_win = 50
min_duration = 5
targets = [{'id': 1, 'p0': np.array([-4.07511080, -16.62913889, 0.0]), 'v': np.array([-0.57143341, 0.82544424, 0.0]), 't_window': (72.19987723, 122.19987723), 'z_window': (0, 15)},
           {'id': 2, 'p0': np.array([-45.33343368, 47.37555188, 0.0]), 'v': np.array([0.61638567, -0.54530322, 0.0]), 't_window': (38.24619913, 88.24619913), 'z_window': (0, 29)},
           {'id': 3, 'p0': np.array([9.24145689, -45.35495873, 0.0]), 'v': np.array([-0.80377243, 0.58526206, 0.0]), 't_window': (94.88855373, 144.88855373), 'z_window': (0, 23)},
           {'id': 4, 'p0': np.array([30.83973481, -19.53862308, 0.0]), 'v': np.array([-0.54883606, 0.84211651, 0.0]), 't_window': (12.20382348, 62.20382348), 'z_window': (0, 17)},
           {'id': 5, 'p0': np.array([38.72127426, -2.77850748, 0.0]), 'v': np.array([-0.55979712, -0.85662239, 0.0]), 't_window': (56.12771976, 106.12771976), 'z_window': (0, 23)}]

depot = {'id': 0, 'p0': np.array([0, 0, z0]), 'v': np.array([0, 0, 0]), 't_window': (0, 0), 'z_window': (z0, z0)}
depot_copy = {'id': n_targets + 1, 'p0': np.array([0, 0, z0]), 'v': np.array([0, 0, 0]), 't_window': (0, T), 'z_window': (z0, z0)}

model = gp.Model("DR-MT-TSP_bigM")
nodes = [depot] + targets + [depot_copy]
n_nodes = len(nodes)

edges = []
for i in range(n_nodes):
    for j in range(n_nodes):
        if i != j and nodes[i]['id'] != depot_copy['id'] and nodes[j]['id'] != depot['id']:
            edges.append((i, j))  # The direct edge from depot to depot_copy can be omitted without affecting the formulation.

# ======================
# Variables:
t_s = {i: model.addVar(lb=nodes[i]['t_window'][0],
                       ub=nodes[i]['t_window'][1], name=f"t_s_{i}")
       for i in range(1, n_nodes)}
t_f = {i: model.addVar(lb=nodes[i]['t_window'][0],
                       ub=nodes[i]['t_window'][1], name=f"t_f_{i}")
       for i in range(0, n_nodes - 1)}
x_c_s = {i: model.addVar(lb=-area_size/2, ub=area_size/2, name=f"x_c_s_{i}")
         for i in range(1, n_nodes)}
y_c_s = {i: model.addVar(lb=-area_size/2, ub=area_size/2, name=f"y_c_s_{i}")
         for i in range(1, n_nodes)}
x_c_f = {i: model.addVar(lb=-area_size/2, ub=area_size/2, name=f"x_c_f_{i}")
         for i in range(0, n_nodes - 1)}
y_c_f = {i: model.addVar(lb=-area_size/2, ub=area_size/2, name=f"y_c_f_{i}")
         for i in range(0, n_nodes - 1)}
z_c_s = {i: model.addVar(lb=nodes[i]['z_window'][0], ub=nodes[i]['z_window'][1], name=f"z_c_s_{i}")
         for i in range(1, n_nodes)}
z_c_f = {i: model.addVar(lb=nodes[i]['z_window'][0], ub=nodes[i]['z_window'][1], name=f"z_c_f_{i}")
         for i in range(0, n_nodes - 1)}

d_x_s = {i: model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"dx_s_{i}")
         for i in range(1, n_nodes)}
d_y_s = {i: model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"dy_s_{i}")
         for i in range(1, n_nodes)}
d_x_f = {i: model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"dx_f_{i}")
         for i in range(0, n_nodes - 1)}
d_y_f = {i: model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"dy_f_{i}")
         for i in range(0, n_nodes - 1)}

b, l, l_tilde = {}, {}, {}
for e in edges:
    i, j = e
    b[e] = model.addVar(vtype=GRB.BINARY, name=f"b_{i}_{j}")
    l[e] = model.addVar(lb=0.0, name=f"l_{i}_{j}")
    l_tilde[e] = model.addVar(lb=0.0, name=f"l_tilde_{i}_{j}")

x_diff = {}
y_diff = {}
z_diff = {}
t_diff = {}
l_diff = {}
for k in range(1, n_nodes - 1):
    x_diff[k] = model.addVar(lb=0.0, name=f"x_diff_{k}")
    y_diff[k] = model.addVar(lb=0.0, name=f"y_diff_{k}")
    z_diff[k] = model.addVar(lb=0.0, ub=ws_z, name=f"z_diff_{k}")
    t_diff[k] = model.addVar(lb=min_duration, ub=T, name=f"t_diff_{k}")
    l_diff[k] = model.addVar(lb=0.0, name=f"l_diff_{k}")

model.setObjective(gp.quicksum(l_tilde[e] for e in edges), GRB.MINIMIZE)
# Uncomment to use the objective in Eq. (41), Sec. 5.1.5, which includes the monitoring distance:
# model.setObjective(gp.quicksum(l_tilde[e] for e in edges) + gp.quicksum(l_diff[k] for k in range(1, n_nodes - 1)), GRB.MINIMIZE)

# ------------------------------
# Constraints:
model.addConstr(gp.quicksum(b[e] for e in edges if e[0] == 0) == 1)  # depart depot once
model.addConstr(gp.quicksum(b[e] for e in edges if e[1] == n_nodes - 1) == 1)  # arrive depot_copy once
for k in range(1, n_nodes - 1):
    model.addConstr(gp.quicksum(b[e] for e in edges if e[1] == k) == 1)
    model.addConstr(gp.quicksum(b[e] for e in edges if e[0] == k) == gp.quicksum(b[e] for e in edges if e[1] == k))

R_c = math.ceil(math.sqrt(area_size**2 + area_size**2 + ws_z**2))

for i in range(n_nodes):
    base_i_x = nodes[i]['p0'][0] - nodes[i]['v'][0] * nodes[i]['t_window'][0]
    base_i_y = nodes[i]['p0'][1] - nodes[i]['v'][1] * nodes[i]['t_window'][0]
    if i != 0:
        model.addConstr(x_c_s[i] == nodes[i]['v'][0] * t_s[i] + base_i_x + d_x_s[i])
        model.addConstr(y_c_s[i] == nodes[i]['v'][1] * t_s[i] + base_i_y + d_y_s[i])
        model.addQConstr((k_r * z_c_s[i]) * (k_r * z_c_s[i]) >= d_x_s[i] * d_x_s[i] + d_y_s[i] * d_y_s[i],
                         name=f"soc_dr_s_{i}")  # This constraint can be omitted for depot_copy.
    if i != n_nodes - 1:
        model.addConstr(x_c_f[i] == nodes[i]['v'][0] * t_f[i] + base_i_x + d_x_f[i])
        model.addConstr(y_c_f[i] == nodes[i]['v'][1] * t_f[i] + base_i_y + d_y_f[i])
        model.addQConstr((k_r * z_c_f[i]) * (k_r * z_c_f[i]) >= d_x_f[i] * d_x_f[i] + d_y_f[i] * d_y_f[i],
                         name=f"soc_dr_f_{i}")  # This constraint can be omitted for depot.
    if i != 0 and i != n_nodes - 1:
        model.addConstr(t_diff[i] == t_f[i] - t_s[i])
        model.addConstr(x_c_f[i] - x_c_s[i] <= x_diff[i])
        model.addConstr(x_c_f[i] - x_c_s[i] >= - x_diff[i])
        model.addConstr(y_c_f[i] - y_c_s[i] <= y_diff[i])
        model.addConstr(y_c_f[i] - y_c_s[i] >= - y_diff[i])
        model.addConstr(z_c_f[i] - z_c_s[i] <= z_diff[i])
        model.addConstr(z_c_f[i] - z_c_s[i] >= - z_diff[i])
        model.addQConstr(l_diff[i] * l_diff[i] >= x_diff[i] * x_diff[i] + y_diff[i] * y_diff[i] + z_diff[i] * z_diff[i],
                         name=f"soc_diff_{i}")
        model.addConstr(l_diff[i] <= v_max * t_diff[i])

    if nodes[i]['id'] == 0:
        model.addConstr(d_x_f[i] == 0)
        model.addConstr(d_y_f[i] == 0)
    if nodes[i]['id'] == n_nodes - 1:
        model.addConstr(d_x_s[i] == 0)
        model.addConstr(d_y_s[i] == 0)

for e in edges:
    i, j = e
    model.addQConstr(l[e] * l[e] >= (x_c_s[j] - x_c_f[i]) * (x_c_s[j] - x_c_f[i]) + (y_c_s[j] - y_c_f[i]) * (y_c_s[j] - y_c_f[i]) + (z_c_s[j] - z_c_f[i]) * (z_c_s[j] - z_c_f[i]),
                     name=f"soc_euclid_{e[0]}_{e[1]}")
    model.addConstr(l_tilde[e] <= v_max * (t_s[j] - t_f[i] + T * (1 - b[e])))
    model.addConstr(l[e] == l_tilde[e] + R_c * (1 - b[e]))

if min_duration == 0:
    u = {}

    for i in range(1, n_nodes - 1):
        u[i] = model.addVar(lb=1, ub=n_nodes - 2, vtype=GRB.INTEGER, name=f"u_{i}")

    for e in edges:
        i, j = e
        if 1 <= i <= n_targets and 1 <= j <= n_targets:
            model.addConstr(u[i] - u[j] + n_targets * b[e] <= n_targets - 1, name=f"MTZ_{i}_{j}")

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

def reconstruct_path(b_sol, start_node, end_node):
    next_node = {}
    for (i, j), val in b_sol.items():
        if val > 0.5:
            next_node[i] = j

    path = [start_node]
    current = start_node
    while current != end_node:
        if current not in next_node:
            print(f"Warning: path reconstruction failed at node {current}: no outgoing edge found.")
            break
        nxt = next_node[current]
        path.append(nxt)
        current = nxt
    return path

b_sol_bigM = {}
for e in edges:
    b_sol_bigM[e] = b[e].X

path_bigM = reconstruct_path(b_sol_bigM, start_node=0, end_node=n_nodes-1)
print("Big-M path:", path_bigM)

if model.status == GRB.OPTIMAL:
    print(f"model.Runtime: {model.Runtime}")

    selected_edges = [e for e in edges if b[e].X > 0.5]
    path = [0]
    cur = 0
    while cur != n_nodes - 1:
        progressed = False
        for e in selected_edges:
            if e[0] == cur:
                cur = e[1]
                path.append(cur)
                progressed = True
                break
        if not progressed:
            print("Warning: path traversal failed at node", cur)
            break

    path_info = []
    for node_idx in path:
        if node_idx == 0 or node_idx == n_nodes - 1:
            continue

        t_s_val = float(t_s[node_idx].X)
        t_f_val = float(t_f[node_idx].X)
        x_s_val = float(x_c_s[node_idx].X)
        y_s_val = float(y_c_s[node_idx].X)
        z_s_val = float(z_c_s[node_idx].X)
        x_f_val = float(x_c_f[node_idx].X)
        y_f_val = float(y_c_f[node_idx].X)
        z_f_val = float(z_c_f[node_idx].X)

        incoming_edge = None
        for e in selected_edges:
            if e[1] == node_idx:
                incoming_edge = e
                break

        outgoing_edge = None
        for e in selected_edges:
            if e[0] == node_idx:
                outgoing_edge = e
                break

        if incoming_edge:
            i, j = incoming_edge
            t_arrival = t_s_val
            p_arrival = np.array([x_s_val, y_s_val, z_s_val], dtype=float)
            path_info.append({
                'type': 'travel',
                'from': i, 'to': j,
                't_start': float(t_f[i].X) if i > 0 else 0.0,
                't_end': t_arrival,
                'p_start': np.array([float(x_c_f[i].X), float(y_c_f[i].X), float(z_c_f[i].X)]) if i > 0 else np.array(
                    [depot['p0'][0], depot['p0'][1], depot['p0'][2]]),
                'p_end': p_arrival
            })

        path_info.append({
            'type': 'monitor',
            'target': node_idx,
            't_start': t_s_val,
            't_end': t_f_val,
            'p_start': np.array([x_s_val, y_s_val, z_s_val]),
            'p_end': np.array([x_f_val, y_f_val, z_f_val])
        })

    if len(path) >= 2:
        last_target = path[-2]
        depot_copy_node = path[-1]
        e = (last_target, depot_copy_node)
        if e in edges and b[e].X > 0.5:
            t_start = float(t_f[last_target].X)
            t_end = float(t_s[depot_copy_node].X)
            p_start = np.array([
                float(x_c_f[last_target].X),
                float(y_c_f[last_target].X),
                float(z_c_f[last_target].X)
            ])
            p_end = np.array([depot['p0'][0], depot['p0'][1], depot['p0'][2]])

            path_info.append({
                'type': 'travel',
                'from': last_target,
                'to': depot_copy_node,
                't_start': t_start,
                't_end': t_end,
                'p_start': p_start,
                'p_end': p_end
            })

    path_info = sorted(path_info, key=lambda s: s['t_start'])

    total_time = max(seg['t_end'] for seg in path_info) if path_info else 0.0

    path_points, path_times, path_types = [], [], []
    for seg in path_info:
        if len(path_points) == 0 or not np.array_equal(path_points[-1], seg['p_start']):
            path_points.append(seg['p_start'])
            path_times.append(seg['t_start'])
            path_types.append(seg['type'])
        path_points.append(seg['p_end'])
        path_times.append(seg['t_end'])
        path_types.append(seg['type'])

    eps = 1e-4
    for i in range(1, len(path_times)):
        if path_times[i] <= path_times[i - 1]:
            path_times[i] = path_times[i - 1] + eps

    target_monitor_periods = [[] for _ in range(n_targets)]
    target_monitor_positions = [[] for _ in range(n_targets)]

    for seg in path_info:
        if seg['type'] == 'monitor':
            target_idx = seg['target'] - 1
            target_monitor_periods[target_idx].append((seg['t_start'], seg['t_end']))
            node = targets[target_idx]
            t_s_target = node['t_window'][0]
            p_start_monitor = node['p0'] + node['v'] * (seg['t_start'] - t_s_target)
            p_end_monitor = node['p0'] + node['v'] * (seg['t_end'] - t_s_target)
            target_monitor_positions[target_idx].append({
                't_start': seg['t_start'],
                't_end': seg['t_end'],
                'p_start': np.array([p_start_monitor[0], p_start_monitor[1], 0]),
                'p_end': np.array([p_end_monitor[0], p_end_monitor[1], 0])
            })

    frames = max(80, int(min(300, max(40, total_time * 4)))) if total_time > 0 else 120
    time_points = np.linspace(0.0, total_time, frames)

    agent_positions = []
    path_x, path_y, path_z = [], [], []

    for t_curr in time_points:
        pos = path_points[-1].copy() if len(path_points) > 0 else np.array([0, 0, 0])
        for k in range(len(path_times) - 1):
            t0, t1 = path_times[k], path_times[k + 1]
            if t0 <= t_curr <= t1:
                alpha = (t_curr - t0) / (t1 - t0 + 1e-12)
                pos = path_points[k] + alpha * (path_points[k + 1] - path_points[k])
                break
        agent_positions.append(pos)
        path_x.append(pos[0])
        path_y.append(pos[1])
        path_z.append(pos[2])

    cmap = plt.cm.tab10
    colors = [cmap(i % 10) for i in range(n_targets)]

    target_positions = [[] for _ in range(n_targets)]
    target_colors = [[] for _ in range(n_targets)]

    for fi, t_curr in enumerate(time_points):
        for idx, node in enumerate(targets):
            t_s_target, t_e_target = node['t_window']
            if t_s_target <= t_curr <= t_e_target:
                p = node['p0'] + node['v'] * (t_curr - t_s_target)
                target_positions[idx].append(np.array([p[0], p[1], 0.0]))

                is_monitoring = False
                for monitor_start, monitor_end in target_monitor_periods[idx]:
                    if monitor_start <= t_curr <= monitor_end:
                        is_monitoring = True
                        break
                target_colors[idx].append('green' if is_monitoring else colors[idx])
            else:
                target_positions[idx].append(None)
                target_colors[idx].append(colors[idx])

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-area_size / 2, area_size / 2)
    ax.set_ylim(-area_size / 2, area_size / 2)
    ax.set_zlim(0, ws_z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"DR-MT-TSP: Big-M Based Formulation ($n_{{tar}}$={n_targets})")

    legend_el = [
        plt.Line2D([0], [0], color='b', marker='o', linestyle='None', label='Robot'),
        plt.Line2D([0], [0], color='r', linestyle='-', label='Robot Path'),
        plt.Line2D([0], [0], color='k', marker='s', linestyle='None', label='Depot'),
        plt.Line2D([0], [0], color='gray', linestyle='-', label='Height Window'),
        plt.Line2D([0], [0], color='green', marker='o', linestyle='None', label='Target Being Monitored')
    ]
    for idx in range(n_targets):
        legend_el.append(
            plt.Line2D([0], [0], color=colors[idx], marker='o', linestyle='None', label=f'Target {idx + 1}'))
    ax.legend(handles=legend_el, loc='upper left', bbox_to_anchor=(0, 1))

    agent_dot, = ax.plot([], [], [], 'bo', markersize=10)

    path_line, = ax.plot([], [], [], 'r-', linewidth=2, alpha=0.8)

    time_text = ax.text2D(0.98, 0.95, '', transform=ax.transAxes, fontsize=12,
                          ha='right',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.plot([depot['p0'][0]], [depot['p0'][1]], [depot['p0'][2]], 'ks', markersize=10)

    target_dots = []
    target_bars = []
    target_trajectories = []
    target_monitor_trajectories = []

    for idx, node in enumerate(targets):
        t_s_target, t_e_target = node['t_window']
        p_s = node['p0']
        p_e = node['p0'] + node['v'] * (t_e_target - t_s_target)

        traj, = ax.plot([p_s[0], p_e[0]], [p_s[1], p_e[1]], [0, 0], '--',
                        color=colors[idx], alpha=0.4, linewidth=1)
        target_trajectories.append(traj)

        monitor_traj, = ax.plot([], [], [], 'g-', linewidth=3, alpha=0.7)
        target_monitor_trajectories.append(monitor_traj)

        dot, = ax.plot([], [], [], 'o', color=colors[idx], markersize=8)
        target_dots.append(dot)

        bar_line, = ax.plot([], [], [], '-', color=colors[idx], linewidth=3, alpha=0.6)
        target_bars.append(bar_line)

    cone_surf = [None]


    def plot_cone(ax, x0, y0, z0, kr, num_segments=40):
        if z0 <= 0:
            return None
        theta = np.linspace(0, 2 * np.pi, num_segments)
        Z = np.linspace(0, z0, 2)
        T, ZZ = np.meshgrid(theta, Z)
        R = kr * (z0 - ZZ)
        X = x0 + R * np.cos(T)
        Y = y0 + R * np.sin(T)
        return ax.plot_surface(X, Y, ZZ, alpha=0.2, color='blue', linewidth=0, antialiased=True)


    permanent_monitor_trajectories = [[] for _ in range(n_targets)]


    def update(frame):
        t_curr = time_points[frame]
        pos = agent_positions[frame]
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

        agent_dot.set_data([x], [y])
        agent_dot.set_3d_properties([z])

        path_line.set_data(path_x[:frame + 1], path_y[:frame + 1])
        path_line.set_3d_properties(path_z[:frame + 1])

        if cone_surf[0] is not None:
            try:
                cone_surf[0].remove()
            except Exception:
                ax.collections.clear()
            cone_surf[0] = None
        cone_surf[0] = plot_cone(ax, x, y, z, k_r)

        for idx in range(n_targets):
            dot = target_dots[idx]
            bar = target_bars[idx]
            monitor_traj = target_monitor_trajectories[idx]

            pcur = target_positions[idx][frame] if frame < len(target_positions[idx]) else None

            if pcur is not None:
                dot.set_data([pcur[0]], [pcur[1]])
                dot.set_3d_properties([pcur[2]])

                current_color = target_colors[idx][frame] if frame < len(target_colors[idx]) else colors[idx]
                dot.set_color(current_color)
                dot.set_markersize(10 if current_color == 'green' else 8)

                z_low, z_high = targets[idx]['z_window']
                bar.set_data([pcur[0], pcur[0]], [pcur[1], pcur[1]])
                bar.set_3d_properties([z_low, z_high])
                bar.set_alpha(0.6)

                for monitor_info in target_monitor_positions[idx]:
                    monitor_start, monitor_end = monitor_info['t_start'], monitor_info['t_end']
                    if monitor_start <= t_curr <= monitor_end:
                        p_monitor_start = monitor_info['p_start']
                        p_monitor_end = monitor_info['p_end']

                        progress = (t_curr - monitor_start) / (monitor_end - monitor_start + 1e-12)
                        p_current_monitor = p_monitor_start + progress * (p_monitor_end - p_monitor_start)

                        monitor_x = [p_monitor_start[0], p_current_monitor[0]]
                        monitor_y = [p_monitor_start[1], p_current_monitor[1]]
                        monitor_z = [0, 0]
                        monitor_traj.set_data(monitor_x, monitor_y)
                        monitor_traj.set_3d_properties(monitor_z)

                    elif t_curr > monitor_end:
                        segment_key = (monitor_start, monitor_end)
                        if segment_key not in [seg['key'] for seg in permanent_monitor_trajectories[idx]]:
                            p_monitor_start = monitor_info['p_start']
                            p_monitor_end = monitor_info['p_end']
                            permanent_traj, = ax.plot(
                                [p_monitor_start[0], p_monitor_end[0]],
                                [p_monitor_start[1], p_monitor_end[1]],
                                [0, 0], 'g-', linewidth=3, alpha=0.8
                            )
                            permanent_monitor_trajectories[idx].append({
                                'key': segment_key,
                                'trajectory': permanent_traj
                            })

                        monitor_traj.set_data([], [])
                        monitor_traj.set_3d_properties([])

            else:
                dot.set_data([], [])
                dot.set_3d_properties([])
                bar.set_data([], [])
                bar.set_3d_properties([])
                monitor_traj.set_data([], [])
                monitor_traj.set_3d_properties([])

        time_text.set_text(f"Time: {t_curr:.1f}/{total_time:.1f}")

        artists = [agent_dot, path_line, time_text] + target_dots + target_bars + target_monitor_trajectories
        if cone_surf[0] is not None:
            artists.append(cone_surf[0])

        return artists


    ani = FuncAnimation(fig, update, frames=frames, interval=int(1000 / 15), blit=False, repeat=True)

    save_fps = 15
    visualization_dir = "Visualization"
    os.makedirs(visualization_dir, exist_ok=True)

    gif_name = f"DR-MT-TSP_Big-M_n_tar_{n_targets}_v_{v_max}_t_win_{time_win}_min_dura_{min_duration}_obj_{model.ObjVal:.2f}.gif"
    gif_path = os.path.join(visualization_dir, gif_name)

    ani.save(gif_path, writer='pillow', fps=save_fps)
    print(f"Animation saved: {gif_path}")

    plt.tight_layout()
    plt.show()

else:
    print("No optimal solution found")