from rl.env import Env
from rl.agent import Agent
from rl.transition import Transition
from rl.monitor import Monitor


def simulate(env: Env, agent: Agent, monitor: Monitor, episodes: int) -> None:
    for episode in range(episodes):
        agent.episode_start()
        done = False
        obs = env.reset()
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            transition = Transition(obs, action, reward, next_obs, done)
            agent.store_transition(transition)
            monitor.store_transition(env, transition)
            obs = next_obs
        agent.episode_end()
        monitor.episode_end(episode, episodes)
    env.close()
    monitor.simulation_end()
