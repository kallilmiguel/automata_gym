from gym.envs.registration import register

register(
    id='automata-v0',
    entry_point='automata.envs:automataEnv',
)
