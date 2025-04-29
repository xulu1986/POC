import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

SIZE          = 12
MAX_STEPS     = 300
INIT_LEN      = 3

DIR_VEC       = np.array([[0,-1],[1,0],[0,1],[-1,0]])

INPUTS        = 5
OUTPUTS       = 3
WEIGHTS_LEN   = INPUTS*OUTPUTS + OUTPUTS        # flattened (W,b)

POP_SIZE      = 120
GENERATIONS   = 200
ELITE_FRAC    = 0.2
MUT_P         = 0.1

rng = np.random.default_rng(2)

def reset_game():
    mid = SIZE//2
    body = [(mid, mid+i) for i in range(INIT_LEN)][::-1]  # head at body[0]
    dir_idx = 0  # up
    food = spawn_food(body)
    return body, dir_idx, food


def spawn_food(body):
    empty = [(x,y) for x in range(SIZE) for y in range(SIZE) if (x,y) not in body]
    return empty[rng.integers(len(empty))]


def step(body, dir_idx, food, turn):
    dir_idx = (dir_idx + turn) % 4
    head = np.add(body[0], DIR_VEC[dir_idx])
    head = tuple(head)

    if not (0 <= head[0] < SIZE and 0 <= head[1] < SIZE) or head in body:
        return body, dir_idx, food, -1, True
    body.insert(0, head)

    reward = 0
    if head == food:
        reward = 1
        food = spawn_food(body)
    else:
        body.pop()

    return body, dir_idx, food, reward, False

def policy_outputs(weights, inp):
    W = weights[:INPUTS*OUTPUTS].reshape(OUTPUTS, INPUTS)
    b = weights[INPUTS*OUTPUTS:]
    return W @ inp + b

def get_inputs(body, dir_idx, food):
    head = np.array(body[0])
    vec_f = np.array(food) - head
    theta = np.arctan2(vec_f[1], vec_f[0] + 1e-9)
    sin_t, cos_t = np.sin(theta), np.cos(theta)

    def blocked(turn):
        d = (dir_idx + turn) % 4
        nxt = head + DIR_VEC[d]
        return int(not (0 <= nxt[0] < SIZE and 0 <= nxt[1] < SIZE) or tuple(nxt) in body)

    return np.array([sin_t, cos_t, blocked(0), blocked(-1), blocked(1)])

def fitness(weights):
    body, dir_idx, food = reset_game()
    score, steps_since_food = 0, 0
    for step_idx in range(MAX_STEPS):
        inp = get_inputs(body, dir_idx, food)
        logits = policy_outputs(weights, inp)
        turn = np.argmax(logits) - 1  # -1 left,0 straight,1 right
        body, dir_idx, food, reward, dead = step(body, dir_idx, food, turn)
        score += reward
        steps_since_food = 0 if reward==1 else steps_since_food+1
        if dead or steps_since_food > SIZE*SIZE:
            break
    return score + len(body)*0.1

def init_pop():
    return rng.normal(scale=1.0, size=(POP_SIZE, WEIGHTS_LEN))

def crossover(p1, p2):
    cut = rng.integers(1, WEIGHTS_LEN)
    child = np.concatenate([p1[:cut], p2[cut:]])
    return child

def mutate(child):
    mask = rng.random(WEIGHTS_LEN) < MUT_P
    child[mask] += rng.normal(scale=0.5, size=mask.sum())

history_best_frames = []

pop = init_pop()
for g in range(GENERATIONS):
    scores = np.array([fitness(ind) for ind in pop])
    best_idx = int(scores.argmax())
    best = pop[best_idx].copy()

    frames = []
    body, dir_idx, food = reset_game()
    grid = np.zeros((SIZE, SIZE), int)
    for _ in range(60):
        grid[:] = 0
        for (x,y) in body:
            grid[y,x]=1
        grid[food[1], food[0]] = 2
        frames.append(grid.copy())
        inp = get_inputs(body, dir_idx, food)
        turn = np.argmax(policy_outputs(best, inp)) - 1
        body, dir_idx, food, reward, dead = step(body, dir_idx, food, turn)
        if dead:
            break
    history_best_frames.append(frames)

    elite_n = max(2, int(POP_SIZE*ELITE_FRAC))
    elite_idx = scores.argsort()[::-1][:elite_n]
    elite = pop[elite_idx]

    children = []
    while len(children) < POP_SIZE:
        p1, p2 = elite[rng.integers(elite_n, size=2)]
        child = crossover(p1, p2)
        mutate(child)
        children.append(child)
    pop = np.array(children)

cmap = plt.get_cmap('nipy_spectral')
fig, ax = plt.subplots(figsize=(4,4))
img = ax.imshow(history_best_frames[0][0], cmap=cmap, vmin=0, vmax=2)
ax.set_title('Evolving Neon Snake-Bot')
ax.axis('off')
text = ax.text(0.02,0.97,'',transform=ax.transAxes,va='top',color='white')

def update(frame):
    gen = frame // 60
    step_idx = frame % 60
    frames = history_best_frames[min(gen, len(history_best_frames)-1)]
    if step_idx >= len(frames):
        frame_img = frames[-1]
    else:
        frame_img = frames[step_idx]
    img.set_data(frame_img)
    text.set_text(f'Generation {gen+1}/{GENERATIONS}')
    return img, text

ani = animation.FuncAnimation(fig, update, frames=GENERATIONS*60, interval=50,
                              blit=True, repeat=False)

ani.save('neon_snakebot.gif', writer='pillow', fps=20)