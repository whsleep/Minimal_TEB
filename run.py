import irsim
from sim import SIM_ENV

# env = irsim.make('robot_world.yaml')
env = SIM_ENV(render=True)

for i in range(300):

    env.step()
    # env.step()
    # env.render(0.05)
    
    # if env.done():
    #     break

