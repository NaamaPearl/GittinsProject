# Distributed Q-Learning with Gittins Prioritization
We consider a distributed reinforcement learning framework where multiple agents interact with the environment in parallel, while sharing experience, in order to ﬁnd the optimal policy. At each time step, only a sub set of the agents is selected to interact with the environment. We explore several mechanisms for selecting which agents to prioritize based on the reward and the TD-error, and analyze their effect on the learning process. When the model is known, the optimal prioritization policy is the Gittins index. We propose an algorithm for learning the Gittins index from demonstrations and show that it yields an &epsilon;−optimal Gittins policy. Simulations in tabular MDPs show that prioritization signiﬁcantly improves the sample complexity.

## Code
Use `main.py` as an entry point the the code. Generate new MDPs, and save run results:
```
$ python main.py
```
use `-p` to plot results, and `-l` to load previously generated MDPs.

## Parameters
### Simulation Paramneters
Defualt values for all needed simulation parameters are defined in `Framework\config.py`. 

Override defaults by explicitly assigning other values in `main.py`. For example, overriding the default value for simulation steps number, is done as follows:
```
cfg.SimulationParameters(steps=1000)
```

### MDP Parameters
Parmaters for MDP types are stored in `MDPModel\MDPConfig.py`, and can be changed as listed above. Note that some default values here are derived from other values, so it'd be wise to check the file before overriding.
