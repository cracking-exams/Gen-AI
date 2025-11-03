import random

class Agent:
    def __init__(self, name):
        self.name = name
        self.energy = 10
        self.food_collected = 0

    def sense(self, environment):
        # See if food is at current location
        return environment[self.energy % len(environment)]

    def decide(self, observation):
        if observation == "food":
            return "collect"
        else:
            return "move"

    def act(self, action):
        if action == "collect":
            self.food_collected += 1
        self.energy -= 1 # lose 1 energy per step


environment = ["nothing", "food", "nothing", "food", "nothing", "nothing", "nothing", "food"]

agent = Agent("A")
for _ in range(8):
    observation = agent.sense(environment)
    action = agent.decide(observation)
    agent.act(action)

print(f"{agent.name} collected {agent.food_collected} food. Energy left: {agent.energy}")
