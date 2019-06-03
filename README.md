# Distributed Q-Learning with Gittins Prioritization

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
