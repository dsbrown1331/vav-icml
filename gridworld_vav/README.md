# value_alignment_verification grid world experiments

Set up the conda environment. Really just needs numpy python3 and scipy.

`conda env create -f environment.yml`

To run the experiment in Appendix F.

`python experiments/basic_value_alignment/gaussian_reward_value_alignment_experiment_runner_diffmethods_arp.py`

To plot the results 

`python data_analysis/arp_gaussian_basic_runner_analysis.py`


To run the Island gridworld case study

`python sandbox/island_navigation_debug.py`

To run the Lavaland gridworld case study:

`python sandbox/lava_land_debug.py`
