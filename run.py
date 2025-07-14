from sim import SIM_ENV

env = SIM_ENV(render=True)

for i in range(300):
    if env.step():
        break


