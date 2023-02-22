
from flask import Flask, render_template
import numpy as np

app = Flask(__name__)

def q_learning(grid, start, end, alpha=0.01, gamma=0.6, eps=0.1, num_episodes=10000):
    Q = np.zeros((grid.shape[0], grid.shape[1], 4))
    for episode in range(num_episodes):
        state = start
        done = False
        while not done:
            if np.random.uniform() < eps:
                action = np.random.choice(4)
            else:
                action = np.argmax(Q[state[0], state[1], :])
            next_state, reward, done = step(state, action, grid)
            td_error = reward + gamma * np.max(Q[next_state[0], next_state[1], :]) - Q[state[0], state[1], action]
            Q[state[0], state[1], action] += alpha * td_error
            state = next_state
    policy = np.zeros((grid.shape[0], grid.shape[1], 4))
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if grid[row, col] != -1:
                best_action = np.argmax(Q[row, col, :])
                policy[row, col, best_action] = 1
    return policy

def step(state, action, grid):
    if action == 0:  # up
        next_state = (max(state[0] - 1, 0), state[1])
    elif action == 1:  # right
        next_state = (state[0], min(state[1] + 1, grid.shape[1] - 1))
    elif action == 2:  # down
        next_state = (min(state[0] + 1, grid.shape[0] - 1), state[1])
    else:  # left
        next_state = (state[0], max(state[1] - 1, 0))
    reward = grid[next_state[0], next_state[1]]
    done = (reward != 0)
    return next_state, reward, done

def render_grid_world(grid, policy=None):
    html = '<table>'
    for row in range(grid.shape[0]):
        html += '<tr>'
        for col in range(grid.shape[1]):
            if grid[row, col] == 0:
                html += '<td>.</td>'
            elif grid[row, col] == -1:
                html += '<td>X</td>'
            elif grid[row, col] == 1:
                html += '<td>G</td>'
            if policy is not None:
                action = np.argmax(policy[row, col, :])
                if action == 0:
                    html += '<td>^</td>'
                elif action == 1:
                    html += '<td>&gt;</td>'
                elif action == 2:
                    html += '<td>v</td>'
                else:
                    html += '<td>&lt;</td>'
        html += '</tr>'
    html += '</table>'
    return html

@app.route('/')
def index():
    grid = np.array([
        [0, 0, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    start = (0, 0)
    end = (3, 3)
    policy = q_learning(grid, start, end)
    grid_html = render_grid_world(grid)
    policy_html = render_grid_world(grid, policy)
  
    # render the template and pass the grid and policy to 
    return render_template('index.html', grid_html=grid_html, policy_html=policy_html)
if __name__ == '__main__':
    app.run(debug='on',port=5002)