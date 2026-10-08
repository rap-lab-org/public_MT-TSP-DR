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

model = gp.Model("CE-MT-TSP_GCS")
nodes = [depot] + targets + [depot_copy]
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
    b[e] = model.addVar(vtype=GRB.BINARY, name=f"b_{i}_{j}")
    bounds = {
        'x': (-area_size/2, area_size/2),
        'y': (-area_size/2, area_size/2),
        't': (0, GRB.INFINITY)
    }
    bounds_prime = {
        'x': (-area_size/2, area_size/2),
        'y': (-area_size/2, area_size/2),
        't': (0, GRB.INFINITY)
    }
    alpha[e] = {
        dim: model.addVar(
            lb=bounds[dim][0],
            ub=bounds[dim][1],
            name=f"alpha_{i}_{j}_{dim}"
        )
        for dim in ['x', 'y', 't']
    }
    alpha_prime[e] = {
        dim: model.addVar(
            lb=bounds_prime[dim][0],
            ub=bounds_prime[dim][1],
            name=f"alpha_prime_{i}_{j}_{dim}"
        )
        for dim in ['x', 'y', 't']
    }
    beta[e] = {dim: model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"beta_{i}_{j}_{dim}") for dim in ['x', 'y']}
    beta_prime[e] = {dim: model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"beta_prime_{i}_{j}_{dim}") for dim in
                     ['x', 'y']}
    l_tilde[e] = model.addVar(lb=0, name=f"l_tilde_{i}_{j}")

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
model.addConstr(gp.quicksum(b[e] for e in edges if e[0] == 0) == 1)
model.addConstr(gp.quicksum(b[e] for e in edges if e[1] == n_nodes - 1) == 1)
for k in range(1, n_nodes - 1):
    model.addConstr(gp.quicksum(b[e] for e in edges if e[1] == k) == 1)
    model.addConstr(gp.quicksum(b[e] for e in edges if e[0] == k) == gp.quicksum(b[e] for e in edges if e[1] == k))

for k in range(1, n_nodes - 1):
    incoming = [e for e in edges if e[1] == k]
    outgoing = [e for e in edges if e[0] == k]
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
    model.addQConstr(l_diff[k] ** 2 >= x_diff[k] ** 2 + y_diff[k] ** 2)
    model.addConstr(l_diff[k] <= v_max * t_diff[k])

for e in edges:
    model.addQConstr(l_tilde[e] ** 2 >=
                    (alpha_prime[e]['x'] - alpha[e]['x']) ** 2 +
                    (alpha_prime[e]['y'] - alpha[e]['y']) ** 2)
    model.addConstr(l_tilde[e] <= v_max * (alpha_prime[e]['t'] - alpha[e]['t']))

for e in edges:
    i, j = e
    ni, nj = nodes[i], nodes[j]
    model.addConstr(alpha[e]['t'] >= b[e] * ni['t_window'][0])
    model.addConstr(alpha[e]['t'] <= b[e] * ni['t_window'][1])
    model.addConstr(alpha_prime[e]['t'] >= b[e] * nj['t_window'][0])
    model.addConstr(alpha_prime[e]['t'] <= b[e] * nj['t_window'][1])

    model.addConstr(alpha[e]['x'] - ni['v'][0] * alpha[e]['t'] ==
                    b[e] * (ni['p0'][0] - ni['v'][0] * ni['t_window'][0]) + beta[e]['x'])
    model.addConstr(alpha[e]['y'] - ni['v'][1] * alpha[e]['t'] ==
                    b[e] * (ni['p0'][1] - ni['v'][1] * ni['t_window'][0]) + beta[e]['y'])
    model.addConstr(alpha_prime[e]['x'] - nj['v'][0] * alpha_prime[e]['t'] ==
                    b[e] * (nj['p0'][0] - nj['v'][0] * nj['t_window'][0]) + beta_prime[e]['x'])
    model.addConstr(alpha_prime[e]['y'] - nj['v'][1] * alpha_prime[e]['t'] ==
                    b[e] * (nj['p0'][1] - nj['v'][1] * nj['t_window'][0]) + beta_prime[e]['y'])

    model.addQConstr((b[e] * r_c) ** 2 >= beta[e]['x'] ** 2 + beta[e]['y'] ** 2)
    if ni['id'] == 0:
        model.addConstr(beta[e]['x'] == 0)
        model.addConstr(beta[e]['y'] == 0)

    model.addQConstr((b[e] * r_c) ** 2 >= beta_prime[e]['x'] ** 2 + beta_prime[e]['y'] ** 2)
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
        p_start = np.array([alpha[e]['x'].X, alpha[e]['y'].X]) / coeff
        p_end = np.array([alpha_prime[e]['x'].X, alpha_prime[e]['y'].X]) / coeff

        path_info.append({
            'type': 'travel',
            'from': u, 'to': v,
            't_start': t_start, 't_end': t_end,
            'p_start': p_start.astype(float), 'p_end': p_end.astype(float)
        })

        if 1 <= v <= n_nodes - 2:
            outgoing_edge = None
            for e_out in selected_edges:
                if e_out[0] == v:
                    outgoing_edge = e_out
                    break
            if outgoing_edge is not None:
                coeff_out = b[outgoing_edge].X if abs(b[outgoing_edge].X) > 1e-9 else 1.0
                monitor_t_start = t_end
                monitor_t_end = float(alpha[outgoing_edge]['t'].X / coeff_out)

                monitor_p_start = p_end
                monitor_p_end = np.array([alpha[outgoing_edge]['x'].X, alpha[outgoing_edge]['y'].X]) / coeff_out

                path_info.append({
                    'type': 'monitor',
                    'target': v,
                    't_start': monitor_t_start,
                    't_end': monitor_t_end,
                    'p_start': monitor_p_start.astype(float),
                    'p_end': monitor_p_end.astype(float)
                })

    path_info = sorted(path_info, key=lambda s: s['t_start']) if path_info else []

    if not path_info:
        print("No path_info available for visualization.")
    else:
        total_time = float(max(seg['t_end'] for seg in path_info))

        path_points = []
        path_times = []
        for seg in path_info:
            if len(path_points) == 0 or not np.allclose(path_points[-1], seg['p_start']):
                path_points.append(np.array(seg['p_start'], dtype=float))
                path_times.append(float(seg['t_start']))
            path_points.append(np.array(seg['p_end'], dtype=float))
            path_times.append(float(seg['t_end']))

        eps = 1e-6
        for i in range(1, len(path_times)):
            if path_times[i] <= path_times[i - 1]:
                path_times[i] = path_times[i - 1] + eps

        target_monitor_periods = [[] for _ in range(n_targets)]
        target_monitor_positions = [[] for _ in range(n_targets)]
        for seg in path_info:
            if seg['type'] == 'monitor':
                t0, t1 = seg['t_start'], seg['t_end']
                target_idx = seg['target'] - 1
                target_monitor_periods[target_idx].append((t0, t1))
                node = targets[target_idx]
                t_s_target = node['t_window'][0]
                p_start_real = node['p0'] + node['v'] * (t0 - t_s_target)
                p_end_real = node['p0'] + node['v'] * (t1 - t_s_target)
                target_monitor_positions[target_idx].append({
                    't_start': t0, 't_end': t1,
                    'p_start': np.array([float(p_start_real[0]), float(p_start_real[1])]),
                    'p_end': np.array([float(p_end_real[0]), float(p_end_real[1])])
                })

        frames = max(80, int(min(400, max(40, total_time * 4))))
        time_points = np.linspace(0.0, total_time, frames)

        agent_positions = []
        path_x, path_y = [], []
        for t_curr in time_points:
            pos = path_points[-1].copy()
            for k in range(len(path_times) - 1):
                t0, t1 = path_times[k], path_times[k + 1]
                if t0 <= t_curr <= t1:
                    alpha_interp = (t_curr - t0) / (t1 - t0 + 1e-12)
                    pos = path_points[k] + alpha_interp * (path_points[k + 1] - path_points[k])
                    break
            agent_positions.append(pos)
            path_x.append(float(pos[0]))
            path_y.append(float(pos[1]))

        target_positions = [[] for _ in range(n_targets)]
        for fi, t_curr in enumerate(time_points):
            for idx, node in enumerate(targets):
                t_s, t_e = node['t_window']
                if t_s <= t_curr <= t_e:
                    p = node['p0'] + node['v'] * (t_curr - t_s)
                    target_positions[idx].append(np.array([float(p[0]), float(p[1])]))
                else:
                    target_positions[idx].append(None)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(-area_size / 2, area_size / 2)
        ax.set_ylim(-area_size / 2, area_size / 2)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f"CE-MT-TSP: GCS Based Formulation ($n_{{tar}}$={n_targets})")

        cmap = plt.cm.tab10
        colors = [cmap(i % 10) for i in range(n_targets)]

        # legend: Robot path (red), Robot (blue), Depot (black square), targets (colored dots)
        legend_el = [
            plt.Line2D([0], [0], color='r', linestyle='-', label='Robot Path'),
            plt.Line2D([0], [0], color='b', marker='o', linestyle='None', label='Robot'),
            plt.Line2D([0], [0], color='k', marker='s', linestyle='None', label='Depot'),
        ]
        for idx in range(n_targets):
            legend_el.append(plt.Line2D([0], [0], color=colors[idx], marker='o', linestyle='None',
                                        label=f"Target {idx + 1}"))
        ax.legend(handles=legend_el, loc='upper left')

        robot_dot, = ax.plot([], [], 'bo', markersize=8)
        robot_path_line, = ax.plot([], [], 'r-', linewidth=2)
        agent_circle = plt.Circle((0, 0), r_c, color='b', alpha=0.12, zorder=0)
        ax.add_patch(agent_circle)

        ax.plot([depot['p0'][0]], [depot['p0'][1]], 'ks', markersize=10)

        target_dots = []
        for idx, node in enumerate(targets):
            t_s, t_e = node['t_window']
            p_s = node['p0']
            p_e = node['p0'] + node['v'] * (t_e - t_s)
            ax.plot([p_s[0], p_e[0]], [p_s[1], p_e[1]], '--', color='gray', alpha=0.4, linewidth=1)
            dot, = ax.plot([], [], 'o', color=colors[idx], markersize=6)
            target_dots.append(dot)

        monitor_trajs_temp = []
        permanent_monitor_trajectories = [[] for _ in range(n_targets)]
        for idx in range(n_targets):
            tline, = ax.plot([], [], '-', color='green', linewidth=3, alpha=0.85)
            monitor_trajs_temp.append(tline)

        time_text = ax.text(0.98, 0.95, '', transform=ax.transAxes,
                            ha='right',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        def is_in_monitor(idx, t_curr):
            for (a, b) in target_monitor_periods[idx]:
                if a <= t_curr <= b:
                    return True
            return False

        def get_active_monitor_info(idx, t_curr):
            for info in target_monitor_positions[idx]:
                if info['t_start'] <= t_curr <= info['t_end']:
                    return info
            return None

        def update(frame):
            t_curr = time_points[frame]
            pos = agent_positions[frame]
            x, y = float(pos[0]), float(pos[1])

            robot_dot.set_data([x], [y])
            robot_path_line.set_data(path_x[:frame + 1], path_y[:frame + 1])
            agent_circle.center = (x, y)
            time_text.set_text(f"Time: {t_curr:.1f}/{total_time:.1f}")

            for idx in range(n_targets):
                pcur = target_positions[idx][frame] if frame < len(target_positions[idx]) else None

                if pcur is not None:
                    in_monitor = is_in_monitor(idx, t_curr)
                    if in_monitor:
                        target_dots[idx].set_color('green')
                        target_dots[idx].set_markersize(9)
                    else:
                        target_dots[idx].set_color(colors[idx])
                        target_dots[idx].set_markersize(6)
                    target_dots[idx].set_data([pcur[0]], [pcur[1]])
                else:
                    target_dots[idx].set_data([], [])

                for monitor_info in target_monitor_positions[idx]:
                    m_start = monitor_info['t_start']
                    m_end = monitor_info['t_end']
                    p_m_s = monitor_info['p_start']
                    p_m_e = monitor_info['p_end']
                    key = (m_start, m_end)

                    if m_start <= t_curr <= m_end:
                        progress = (t_curr - m_start) / (m_end - m_start + 1e-12)
                        p_current = p_m_s + progress * (p_m_e - p_m_s)
                        mon_idx = target_monitor_positions[idx].index(monitor_info)
                        monitor_trajs_temp[idx].set_data([p_m_s[0], p_current[0]], [p_m_s[1], p_current[1]])
                        monitor_trajs_temp[idx].set_alpha(0.95)
                        break
                    elif t_curr > m_end:
                        if key not in [seg['key'] for seg in permanent_monitor_trajectories[idx]]:
                            perm_line, = ax.plot([p_m_s[0], p_m_e[0]], [p_m_s[1], p_m_e[1]], 'g-', linewidth=3, alpha=0.9)
                            permanent_monitor_trajectories[idx].append({'key': key, 'line': perm_line})
                        monitor_trajs_temp[idx].set_data([], [])
                    else:
                        monitor_trajs_temp[idx].set_data([], [])

            artists = [robot_dot, robot_path_line, agent_circle, time_text] + target_dots + monitor_trajs_temp
            for lst in permanent_monitor_trajectories:
                for rec in lst:
                    artists.append(rec['line'])
            return artists

        ani = FuncAnimation(fig, update, frames=frames, interval=int(1000 / 15), blit=False, repeat=False)

        visualization_dir = "Visualization"
        os.makedirs(visualization_dir, exist_ok=True)

        gif_name = f"CE-MT-TSP_GCS_n_tar_{n_targets}_v_{v_max}_t_win_{time_win}_min_dura_{min_duration}_{r_c}_obj_{model.ObjVal:.2f}.gif"
        gif_path = os.path.join(visualization_dir, gif_name)

        ani.save(gif_path, writer='pillow', fps=15)
        print(f"Animation saved: {gif_path}")

        plt.tight_layout()
        plt.show()

else:
    print("No optimal solution found")