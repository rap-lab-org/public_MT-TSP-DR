import math
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

area_size = 100
T = 150
r_c = 5

n_targets = 5
v_max = 8.0
time_win = 50
min_duration = 5
targets = [{'id': 1, 'p0': np.array([10.22485879, 11.54655510, 0.0]), 'v': np.array([-0.66927942, 0.73667038, 0.0]), 't_window': (24.56381877, 74.56381877), },
           {'id': 2, 'p0': np.array([-1.86163076, 5.07354358, 0.0]), 'v': np.array([-0.52936680, 0.72578388, 0.0]), 't_window': (69.11411500, 119.11411500), },
           {'id': 3, 'p0': np.array([7.61490359, -1.13988795, 0.0]), 'v': np.array([0.58013150, -0.83125306, 0.0]), 't_window': (11.53451063, 61.53451063), },
           {'id': 4, 'p0': np.array([14.07406132, 33.21421635, 0.0]), 'v': np.array([0.55521283, -0.75650834, 0.0]), 't_window': (50.10013982, 100.10013982), },
           {'id': 5, 'p0': np.array([-44.51981106, 14.29017596, 0.0]), 'v': np.array([0.64956440, -0.69126916, 0.0]), 't_window': (88.55355074, 138.55355074), }]

depot = {'id': 0, 'p0': np.array([0, 0]), 'v': np.array([0, 0]), 't_window': (0, 0)}
depot_copy = {'id': n_targets + 1, 'p0': np.array([0, 0]), 'v': np.array([0, 0]), 't_window': (0, T)}

model = gp.Model("CE-MT-TSP_bigM")
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
t_diff = {}
l_diff = {}

for k in range(1, n_nodes - 1):
    x_diff[k] = model.addVar(lb=0.0, name=f"x_diff_{k}")
    y_diff[k] = model.addVar(lb=0.0, name=f"y_diff_{k}")
    t_diff[k] = model.addVar(lb=min_duration, ub=T, name=f"t_diff_{k}")
    l_diff[k] = model.addVar(lb=0.0, name=f"l_diff_{k}")

model.setObjective(gp.quicksum(l_tilde[e] for e in edges), GRB.MINIMIZE)
# Uncomment to use the objective in Eq. (41), Sec. 5.1.5, which includes the monitoring distance:
# model.setObjective(gp.quicksum(l_tilde[e] for e in edges) + gp.quicksum(l_diff[k] for k in range(1, n_nodes - 1)), GRB.MINIMIZE)

# ======================
# Constraints
# ======================
model.addConstr(gp.quicksum(b[e] for e in edges if e[0] == 0) == 1)  # depart depot once
model.addConstr(gp.quicksum(b[e] for e in edges if e[1] == n_nodes - 1) == 1)  # arrive depot_copy once

for k in range(1, n_nodes - 1):
    model.addConstr(gp.quicksum(b[e] for e in edges if e[1] == k) == 1)
    model.addConstr(gp.quicksum(b[e] for e in edges if e[0] == k) ==
                    gp.quicksum(b[e] for e in edges if e[1] == k))

for i in range(n_nodes):
    base_i_x = nodes[i]['p0'][0] - nodes[i]['v'][0] * nodes[i]['t_window'][0]
    base_i_y = nodes[i]['p0'][1] - nodes[i]['v'][1] * nodes[i]['t_window'][0]
    if i != 0:
        model.addConstr(x_c_s[i] == nodes[i]['v'][0] * t_s[i] + base_i_x + d_x_s[i])
        model.addConstr(y_c_s[i] == nodes[i]['v'][1] * t_s[i] + base_i_y + d_y_s[i])
        model.addQConstr(r_c ** 2 >= d_x_s[i] * d_x_s[i] + d_y_s[i] * d_y_s[i])  # This constraint can be omitted for depot_copy.

    if i != n_nodes - 1:
        model.addConstr(x_c_f[i] == nodes[i]['v'][0] * t_f[i] + base_i_x + d_x_f[i])
        model.addConstr(y_c_f[i] == nodes[i]['v'][1] * t_f[i] + base_i_y + d_y_f[i])
        model.addQConstr(r_c ** 2 >= d_x_f[i] * d_x_f[i] + d_y_f[i] * d_y_f[i])  # This constraint can be omitted for depot.

    if i != 0 and i != n_nodes - 1:
        model.addConstr(t_diff[i] == t_f[i] - t_s[i])
        model.addConstr(x_c_f[i] - x_c_s[i] <= x_diff[i])
        model.addConstr(x_c_f[i] - x_c_s[i] >= - x_diff[i])
        model.addConstr(y_c_f[i] - y_c_s[i] <= y_diff[i])
        model.addConstr(y_c_f[i] - y_c_s[i] >= - y_diff[i])
        model.addQConstr(l_diff[i] * l_diff[i] >= x_diff[i] * x_diff[i] + y_diff[i] * y_diff[i],
                         name=f"soc_diff_{i}")
        model.addConstr(l_diff[i] <= v_max * t_diff[i])

    if nodes[i]['id'] == 0:
        model.addConstr(d_x_f[i] == 0)
        model.addConstr(d_y_f[i] == 0)
    if nodes[i]['id'] == n_nodes - 1:
        model.addConstr(d_x_s[i] == 0)
        model.addConstr(d_y_s[i] == 0)

R = math.ceil(math.sqrt(area_size ** 2 + area_size ** 2))
for e in edges:
    i, j = e
    model.addQConstr(
        l[e] * l[e] >= (x_c_s[j] - x_c_f[i]) * (x_c_s[j] - x_c_f[i]) + (y_c_s[j] - y_c_f[i]) * (y_c_s[j] - y_c_f[i]),
        name=f"soc_euclid_{e[0]}_{e[1]}")
    model.addConstr(l_tilde[e] <= v_max * (t_s[j] - t_f[i] + T * (1 - b[e])))
    model.addConstr(l[e] == l_tilde[e] + R * (1 - b[e]))

if min_duration == 0:
    u = {}

    for i in range(1, n_nodes - 1):
        u[i] = model.addVar(lb=1, ub=n_nodes - 2, vtype=GRB.INTEGER, name=f"u_{i}")

    for e in edges:
        i, j = e
        if 1 <= i <= n_targets and 1 <= j <= n_targets:
            model.addConstr(u[i] - u[j] + n_targets * b[e] <= n_targets - 1, name=f"MTZ_{i}_{j}")

# ======================
# Solve
# ======================
model.Params.TimeLimit = 300
model.optimize()

# ======================
# Visualization
# ======================
if model.status == GRB.OPTIMAL:
    selected_edges = [e for e in edges if b[e].X > 0.5]
    path = [0]
    cur = 0
    while cur != n_nodes - 1:
        found = False
        for e in selected_edges:
            if e[0] == cur:
                cur = e[1]
                path.append(cur)
                found = True
                break
        if not found:
            print("Warning: cannot continue path from", cur)
            break

    path_points = []
    path_times = []

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if u in t_f and v in t_s:
            p_u = np.array([x_c_f[u].X, y_c_f[u].X])
            p_v = np.array([x_c_s[v].X, y_c_s[v].X])
            t_u = t_f[u].X
            t_v = t_s[v].X
            path_points.extend([p_u, p_v])
            path_times.extend([t_u, t_v])

    if len(path_times) == 0:
        path_points = [np.array([0, 0])]
        path_times = [0.0]
    eps = 1e-4
    for i in range(1, len(path_times)):
        if path_times[i] <= path_times[i - 1]:
            path_times[i] = path_times[i - 1] + eps

    total_time = path_times[-1]

    frames = max(80, int(min(300, max(40, total_time * 4))))
    time_points = np.linspace(0.0, total_time, frames)

    agent_positions = []
    for t_curr in time_points:
        pos = path_points[-1].copy()
        for k in range(len(path_times) - 1):
            t0, t1 = path_times[k], path_times[k + 1]
            if t0 <= t_curr <= t1:
                alpha = (t_curr - t0) / (t1 - t0 + 1e-12)
                pos = path_points[k] + alpha * (path_points[k + 1] - path_points[k])
                break
        agent_positions.append(pos)

    target_positions = [[] for _ in range(n_targets)]
    for fi, t_curr in enumerate(time_points):
        for idx, node in enumerate(targets):
            t_s_i, t_e_i = node['t_window']
            if t_s_i <= t_curr <= t_e_i:
                p = node['p0'] + node['v'] * (t_curr - t_s_i)
                target_positions[idx].append(np.array([float(p[0]), float(p[1])]))
            else:
                target_positions[idx].append(None)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-area_size / 2, area_size / 2)
    ax.set_ylim(-area_size / 2, area_size / 2)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f"CE-MT-TSP: Big-M Based Formulation ($n_{{tar}}$={n_targets})")

    colors = plt.cm.tab10(np.linspace(0, 1, n_targets))

    legend_elements = [
        plt.Line2D([0], [0], color='b', marker='o', linestyle='None', label='Robot'),
        plt.Line2D([0], [0], color='k', marker='s', linestyle='None', label='Depot'),
        plt.Line2D([0], [0], color='r', linestyle='-', linewidth=2, label='Robot Path')
    ]
    for idx in range(n_targets):
        legend_elements.append(
            plt.Line2D([0], [0], color=colors[idx], marker='o', linestyle='None', label=f'Target {idx + 1}')
        )
    ax.legend(handles=legend_elements, loc='upper left')

    agent_dot, = ax.plot([], [], 'bo', markersize=8)
    agent_circle = plt.Circle((0, 0), r_c, color='b', alpha=0.15)
    ax.add_patch(agent_circle)
    path_line, = ax.plot([], [], 'r-', linewidth=2)
    time_text = ax.text(0.98, 0.95, '', transform=ax.transAxes,
                        ha='right',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.plot(depot['p0'][0], depot['p0'][1], 'ks', markersize=10)

    target_dots = []
    target_green_trajs = []
    for idx, node in enumerate(targets):
        t_s_i, t_e_i = node['t_window']
        p_start = node['p0']
        p_end = node['p0'] + node['v'] * (t_e_i - t_s_i)
        ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], '--', color='gray', alpha=0.4)
        dot, = ax.plot([], [], 'o', color=colors[idx], markersize=6)
        target_dots.append(dot)
        green_traj, = ax.plot([], [], '-', color='green', linewidth=2, alpha=0.8)
        target_green_trajs.append(green_traj)

    green_x = [[] for _ in range(n_targets)]
    green_y = [[] for _ in range(n_targets)]
    inside_prev = [False] * n_targets

    def update(frame):
        t_curr = time_points[frame]
        pos = agent_positions[frame]
        x, y = pos[0], pos[1]

        agent_dot.set_data([x], [y])
        agent_circle.center = (x, y)
        path_line.set_data([p[0] for p in agent_positions[:frame + 1]],
                           [p[1] for p in agent_positions[:frame + 1]])
        time_text.set_text(f"Time: {t_curr:.1f}/{total_time:.1f}")

        for idx in range(n_targets):
            p = target_positions[idx][frame]
            if p is not None:
                dist = np.linalg.norm(p - pos)
                target_dots[idx].set_data([p[0]], [p[1]])

                if dist <= r_c:
                    green_x[idx].append(p[0])
                    green_y[idx].append(p[1])
                    target_dots[idx].set_color('green')
                    inside_prev[idx] = True
                else:
                    target_dots[idx].set_color(colors[idx])
                    if inside_prev[idx]:
                        inside_prev[idx] = False

                target_green_trajs[idx].set_data(green_x[idx], green_y[idx])
            else:
                target_dots[idx].set_data([], [])

        return [agent_dot, agent_circle, path_line, time_text] + target_dots + target_green_trajs

    ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=True)

    visualization_dir = "Visualization"
    os.makedirs(visualization_dir, exist_ok=True)

    gif_name = f"CE-MT-TSP_Big-M_n_tar_{n_targets}_v_{v_max}_t_win_{time_win}_min_dura_{min_duration}_obj_{model.ObjVal:.2f}.gif"
    gif_path = os.path.join(visualization_dir, gif_name)

    ani.save(gif_path, writer='pillow', fps=15)
    print(f"Animation saved: {gif_path}")
    plt.close()

else:
    print("No optimal solution found")