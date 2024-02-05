## RL path planning for UAV

To deploy the trained RL algorithm for velocity based path planning, simply run ```run_agent.py``` script. There are two main functions in the script - 

* ```setup_agent```
* ```test```

```setup_agent``` function loads the desired RL algorithm and returns the policy object. Whereas, ```test``` function returns the UAV velocity, given the current pointcloud data, position of the UAV and previous velocity of the UAV as input. 