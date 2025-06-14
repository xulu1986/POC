import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

np.random.seed(15)
N_CITIES = 60
cities = np.random.rand(N_CITIES, 2)

Dx = cities[:, 0][:, None] - cities[:, 0][None, :]
Dy = cities[:, 1][:, None] - cities[:, 1][None, :]
DIST = np.hypot(Dx, Dy)

POP = 250
GENS = 400
ELITE_FRAC = 0.05
MUT_P = 0.4

def route_length(route):
    return DIST[route, np.roll(route, -1)].sum()


def initialize_pop():
    base = np.arange(N_CITIES)
    return np.array([np.random.permutation(base) for _ in range(POP)])

def ox(p1, p2):
    a, b = sorted(np.random.choice(N_CITIES, 2, replace=False))
    child = -np.ones(N_CITIES, dtype=int)
    child[a:b] = p1[a:b]
    fill = [c for c in p2 if c not in child]
    child[child == -1] = fill
    return child

def mutate(route):
    if np.random.rand() < MUT_P:
        i, j = sorted(np.random.choice(N_CITIES, 2, replace=False))
        route[i:j] = route[i:j][::-1]

best_routes = []
best_lengths = []

pop = initialize_pop()
for g in range(GENS):
    lengths = np.array([route_length(r) for r in pop])
    idx_best = int(np.argmin(lengths))
    best_routes.append(pop[idx_best].copy())
    best_lengths.append(lengths[idx_best])

    elite_n = max(1, int(POP*ELITE_FRAC))
    elite_idx = lengths.argsort()[:elite_n]
    elite = pop[elite_idx]

    children = []
    while len(children) < POP:
        p1, p2 = elite[np.random.randint(elite_n, size=2)]
        child = ox(p1, p2)
        mutate(child)
        children.append(child)
    pop = np.array(children)

fig, (ax_map, ax_plot) = plt.subplots(1, 2, figsize=(9, 4))
fig.suptitle('Genetic Algorithm – Tightening the Salesman Route')

ax_map.scatter(cities[:, 0], cities[:, 1], s=25, c='black')
route_line, = ax_map.plot([], [], 'ro-', lw=1.8)
ax_map.set_xticks([]); ax_map.set_yticks([])

line_len, = ax_plot.plot([], [], color='lime')
ax_plot.set_xlim(0, GENS); ax_plot.set_xlabel('Generation')
ax_plot.set_ylabel('Route length')
ax_plot.set_ylim(min(best_lengths)-0.1, max(best_lengths)+0.1)

text = ax_map.text(0.02, 0.96, '', transform=ax_map.transAxes, va='top',
                   bbox=dict(boxstyle='round', fc='white', alpha=0.8))


def update(frame):
    route = best_routes[frame]
    xs = np.append(cities[route][:, 0], cities[route[0]][0])
    ys = np.append(cities[route][:, 1], cities[route[0]][1])
    route_line.set_data(xs, ys)

    line_len.set_data(range(frame+1), best_lengths[:frame+1])
    text.set_text(f'Gen {frame+1}/{GENS}\nLen = {best_lengths[frame]:.3f}')
    return route_line, line_len, text

ani = animation.FuncAnimation(fig, update, frames=GENS, interval=60,
                              blit=True, repeat=False)

ani.save('ga_tsp_route.gif', writer='pillow', fps=15)
