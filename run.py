from sim import SIM_ENV

env = SIM_ENV(render=True)

for i in range(3000):
    if env.step():
        env.env.end(ending_time=i*0.1, suffix='.gif')
        break


