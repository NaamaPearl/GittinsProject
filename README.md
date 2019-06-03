# Distributed Q-Learning with Gittins Prioritization

Use `main.py` as an entry point the the code. Generate new MDPs, and save run results:
```
$ python main.py
```
use `-p` to plot results, and `-l` to use previously generated MDPs.

Defualt values for all needed simulation parameters are defined in `Framework\config.py`. 
Override defaults by explicitly assign other values in `main.py`:
```
cfg.SimulationParameters(steps=1000)
```

Parmaters for MDP types are stored in `MDPModel\MDPConfig.py`, and can be changed as listed above
