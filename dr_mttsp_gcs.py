import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

model = gp.Model("DR-MT-TSP_GCS")
nodes = [depot] + targets + [depot_copy]  # node definition
n_nodes = len(nodes)

edges = []
for i in range(n_nodes):
    for j in range(n_nodes):
        if i != j and nodes[i]['id'] != depot_copy['id'] and nodes[j]['id'] != depot['id']:
            edges.append((i, j))  # The direct edge from depot to depot_copy can be omitted without affecting the formulation.

# ======================
# Variables:

b, alpha, alpha_prime, beta, beta_prime, l_tilde = {}, {}, {}, {}, {}, {}
for e in edges:
    i, j = e
    b[e] = model.addVar(vtype=GRB.BINARY, name=f"b_{i}_{j}")  #binary variable for edge selection

    bounds = {
        'x': (-area_size/2, area_size/2),
        'y': (-area_size/2, area_size/2),
        'z': (0, GRB.INFINITY),
        't': (0, GRB.INFINITY)
    }

    bounds_prime = {
        'x': (-area_size/2, area_size/2),
        'y': (-area_size/2, area_size/2),
        'z': (0, GRB.INFINITY),
        't': (0, GRB.INFINITY)
    }

    alpha[e] = {
        dim: model.addVar(
            lb=bounds[dim][0],
            ub=bounds[dim][1],
            name=f"alpha_{i}_{j}_{dim}"
        )
        for dim in ['x', 'y', 'z', 't']
    }

    alpha_prime[e] = {
        dim: model.addVar(
            lb=bounds_prime[dim][0],
            ub=bounds_prime[dim][1],
            name=f"alpha_prime_{i}_{j}_{dim}"
        )
        for dim in ['x', 'y', 'z', 't']
    }

    beta[e] = {dim: model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"beta_{i}_{j}_{dim}") for dim in ['x', 'y']}
    beta_prime[e] = {dim: model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"beta_prime_{i}_{j}_{dim}") for dim in
                     ['x', 'y']}

    l_tilde[e] = model.addVar(lb=0, name=f"l_tilde_{i}_{j}")  # Inter-target traveling distance variable

x_diff = {}
y_diff = {}
z_diff = {}
t_diff = {}
l_diff = {}
for k in range(1, n_nodes - 1):
    x_diff[k] = model.addVar(lb=0.0, name=f"x_diff_{k}")  # position difference in x direction
    y_diff[k] = model.addVar(lb=0.0, name=f"y_diff_{k}")  # position difference in y direction
    z_diff[k] = model.addVar(lb=0.0, ub=ws_z, name=f"z_diff_{k}")  # position difference in z direction
    t_diff[k] = model.addVar(lb=min_duration, ub=T, name=f"t_diff_{k}")  # time difference
    l_diff[k] = model.addVar(lb=0.0, name=f"l_diff_{k}")  # distance traveled during monitoring target k

# ======================
# Objective function:

model.setObjective(gp.quicksum(l_tilde[e] for e in edges), GRB.MINIMIZE)
# Uncomment to use the objective in Eq. (41), Sec. 5.1.5, which includes the monitoring distance:
# model.setObjective(gp.quicksum(l_tilde[e] for e in edges) + gp.quicksum(l_diff[k] for k in range(1, n_nodes - 1)), GRB.MINIMIZE)

# ======================
# Constraints:

model.addConstr(gp.quicksum(b[e] for e in edges if e[0] == 0) == 1)  # one output edge for depot s
model.addConstr(gp.quicksum(b[e] for e in edges if e[1] == n_nodes - 1) == 1)  # one input edge for depot copy s'

for k in range(1, n_nodes - 1):
    incoming = [e for e in edges if e[1] == k]
    outgoing = [e for e in edges if e[0] == k]
    model.addConstr(gp.quicksum(b[e] for e in incoming) == 1)
    model.addConstr(gp.quicksum(b[e] for e in incoming) ==
                    gp.quicksum(b[e] for e in outgoing))
    model.addConstr(gp.quicksum(alpha_prime[e]['t'] for e in incoming) ==
                    gp.quicksum(alpha[e]['t'] for e in outgoing) - t_diff[k])
    model.addConstr(gp.quicksum(alpha_prime[e]['x'] for e in incoming) <=
                    gp.quicksum(alpha[e]['x'] for e in outgoing) + x_diff[k])
    model.addConstr(gp.quicksum(alpha_prime[e]['x'] for e in incoming) >=
                    gp.quicksum(alpha[e]['x'] for e in outgoing) - x_diff[k])
    model.addConstr(gp.quicksum(alpha_prime[e]['y'] for e in incoming) <=
                    gp.quicksum(alpha[e]['y'] for e in outgoing) + y_diff[k])
    model.addConstr(gp.quicksum(alpha_prime[e]['y'] for e in incoming) >=
                    gp.quicksum(alpha[e]['y'] for e in outgoing) - y_diff[k])
    model.addConstr(gp.quicksum(alpha_prime[e]['z'] for e in incoming) <=
                    gp.quicksum(alpha[e]['z'] for e in outgoing) + z_diff[k])
    model.addConstr(gp.quicksum(alpha_prime[e]['z'] for e in incoming) >=
                    gp.quicksum(alpha[e]['z'] for e in outgoing) - z_diff[k])
    model.addQConstr(l_diff[k] ** 2 >= x_diff[k] ** 2 + y_diff[k] ** 2 + z_diff[k] ** 2)
    model.addConstr(l_diff[k] <= v_max * t_diff[k])

for e in edges:
    model.addQConstr(l_tilde[e] ** 2 >=
                    (alpha_prime[e]['x'] - alpha[e]['x']) ** 2 +
                    (alpha_prime[e]['y'] - alpha[e]['y']) ** 2 +
                    (alpha_prime[e]['z'] - alpha[e]['z']) ** 2)  # Inter-target traveling distance
    model.addConstr(l_tilde[e] <= v_max * (alpha_prime[e]['t'] - alpha[e]['t']))  # Speed limit

for e in edges:
    i, j = e
    ni, nj = nodes[i], nodes[j]
    model.addConstr(alpha[e]['t'] >= b[e] * ni['t_window'][0])
    model.addConstr(alpha[e]['t'] <= b[e] * ni['t_window'][1])
    model.addConstr(alpha_prime[e]['t'] >= b[e] * nj['t_window'][0])
    model.addConstr(alpha_prime[e]['t'] <= b[e] * nj['t_window'][1])

    base_i_x = ni['p0'][0] - ni['v'][0] * ni['t_window'][0]
    base_i_y = ni['p0'][1] - ni['v'][1] * ni['t_window'][0]
    model.addConstr(alpha[e]['x'] - ni['v'][0] * alpha[e]['t'] == b[e] * base_i_x + beta[e]['x'])
    model.addConstr(alpha[e]['y'] - ni['v'][1] * alpha[e]['t'] == b[e] * base_i_y + beta[e]['y'])

    base_j_x = nj['p0'][0] - nj['v'][0] * nj['t_window'][0]
    base_j_y = nj['p0'][1] - nj['v'][1] * nj['t_window'][0]
    model.addConstr(alpha_prime[e]['x'] - nj['v'][0] * alpha_prime[e]['t'] == b[e] * base_j_x + beta_prime[e]['x'])
    model.addConstr(alpha_prime[e]['y'] - nj['v'][1] * alpha_prime[e]['t'] == b[e] * base_j_y + beta_prime[e]['y'])

    z_lower_i, z_upper_i = ni['z_window']
    model.addConstr(alpha[e]['z'] >= b[e] * z_lower_i)
    model.addConstr(alpha[e]['z'] <= b[e] * z_upper_i)
    z_lower_j, z_upper_j = nj['z_window']
    model.addConstr(alpha_prime[e]['z'] >= b[e] * z_lower_j)
    model.addConstr(alpha_prime[e]['z'] <= b[e] * z_upper_j)

    model.addQConstr((k_r * alpha[e]['z']) ** 2 >= beta[e]['x'] ** 2 + beta[e]['y'] ** 2)
    model.addQConstr((k_r * alpha_prime[e]['z']) ** 2 >= beta_prime[e]['x'] ** 2 + beta_prime[e]['y'] ** 2)

    if ni['id'] == 0:
        model.addConstr(beta[e]['x'] == 0)
        model.addConstr(beta[e]['y'] == 0)

    if nj['id'] == n_targets + 1:
        model.addConstr(beta_prime[e]['x'] == 0)
        model.addConstr(beta_prime[e]['y'] == 0)

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
    alpha_vals = {dim: alpha[e][dim].X for dim in ['x', 'y', 'z', 't']}
    if any(abs(val) > 1e-6 for val in alpha_vals.values()):
        i, j = e
        print(f"Edge {e} | from Node {nodes[i]['id']} to Node {nodes[j]['id']}")
        for dim, val in alpha_vals.items():
            if abs(val) > 1e-6:
                print(f"   alpha[{dim}] = {val:.6f}")

print("\n=== Non-zero alpha_prime[e] values ===")
for e in edges:
    alpha_prime_vals = {dim: alpha_prime[e][dim].X for dim in ['x', 'y', 'z', 't']}
    if any(abs(val) > 1e-6 for val in alpha_prime_vals.values()):
        i, j = e
        print(f"Edge {e} | from Node {nodes[i]['id']} to Node {nodes[j]['id']}")
        for dim, val in alpha_prime_vals.items():
            if abs(val) > 1e-6:
                print(f"   alpha_prime[{dim}] = {val:.6f}")

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

b_sol_GCS = {}
for e in edges:
    b_sol_GCS[e] = b[e].X

path_GCS = reconstruct_path(b_sol_GCS, start_node=0, end_node=n_nodes-1)
print("GCS path:", path_GCS)

# ======================
# Visualization for GCS method with monitoring periods
# ======================
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
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        e = (u, v)
        if b[e].X <= 0.5:
            continue

        coeff = b[e].X if abs(b[e].X) > 1e-9 else 1.0

        t_start = float(alpha[e]['t'].X / coeff)
        t_end = float(alpha_prime[e]['t'].X / coeff)
        p_start = np.array([alpha[e]['x'].X, alpha[e]['y'].X, alpha[e]['z'].X]) / coeff
        p_end = np.array([alpha_prime[e]['x'].X, alpha_prime[e]['y'].X, alpha_prime[e]['z'].X]) / coeff

        path_info.append({
            'type': 'travel',
            'from': u, 'to': v,
            't_start': t_start,
            't_end': t_end,
            'p_start': p_start.astype(float),
            'p_end': p_end.astype(float)
        })

        if 1 <= v <= n_nodes - 2:
            target_idx = v - 1
            node = targets[target_idx]

            monitor_t_start = t_end

            outgoing_edge = None
            for e_out in selected_edges:
                if e_out[0] == v:
                    outgoing_edge = e_out
                    break

            if outgoing_edge:
                coeff_out = b[outgoing_edge].X if abs(b[outgoing_edge].X) > 1e-9 else 1.0
                monitor_t_end = float(alpha[outgoing_edge]['t'].X / coeff_out)

                monitor_p_start = p_end

                monitor_p_end = np.array([
                    alpha[outgoing_edge]['x'].X / coeff_out,
                    alpha[outgoing_edge]['y'].X / coeff_out,
                    alpha[outgoing_edge]['z'].X / coeff_out
                ])

                path_info.append({
                    'type': 'monitor',
                    'target': v,
                    't_start': monitor_t_start,
                    't_end': monitor_t_end,
                    'p_start': monitor_p_start.astype(float),
                    'p_end': monitor_p_end.astype(float)
                })

    path_info = sorted(path_info, key=lambda s: s['t_start'])

    total_time = path_info[-1]['t_end'] if path_info else 0.0

    path_points = []
    path_times = []
    path_types = []

    for seg in path_info:
        if seg['type'] == 'travel':
            if len(path_points) == 0 or not np.array_equal(path_points[-1], seg['p_start']):
                path_points.append(seg['p_start'])
                path_times.append(seg['t_start'])
                path_types.append('travel')
            path_points.append(seg['p_end'])
            path_times.append(seg['t_end'])
            path_types.append('travel')
        else:  # monitor segment
            if len(path_points) == 0 or not np.array_equal(path_points[-1], seg['p_start']):
                path_points.append(seg['p_start'])
                path_times.append(seg['t_start'])
                path_types.append('monitor')
            path_points.append(seg['p_end'])
            path_times.append(seg['t_end'])
            path_types.append('monitor')

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
                alpha_interp = (t_curr - t0) / (t1 - t0 + 1e-12)
                pos = path_points[k] + alpha_interp * (path_points[k + 1] - path_points[k])
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
    ax.set_title(f"DR-MT-TSP: GCS Based Formulation ($n_{{tar}}$={n_targets})")

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
                    p_monitor_start = monitor_info['p_start']
                    p_monitor_end = monitor_info['p_end']

                    if monitor_start <= t_curr <= monitor_end:
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

    gif_name = f"DR-MT-TSP_GCS_n_tar_{n_targets}_v_{v_max}_t_win_{time_win}_min_dura_{min_duration}_obj_{model.ObjVal:.2f}.gif"
    gif_path = os.path.join(visualization_dir, gif_name)

    ani.save(gif_path, writer='pillow', fps=save_fps)
    print(f"Animation saved: {gif_path}")

    plt.tight_layout()
    plt.show()

else:
    print("No optimal solution found")