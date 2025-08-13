# Bachelor-Project

# Installing prerequisites:
1. 
install Python 3.10

2. 
run command "pip install -U taichi" 
        or  "py -m pip install -U taichi"

3. 
run command "pip install optuna"

4. Install plotly
run command "pip install plotly"

# Running program:

run command "python main.py arg1 arg2 arg3 arg4"

# Examples
"python main.py valley 0 1 1"
"python main.py valley 1 10 1"
"python main.py box 0 1 1"
"python main.py box 1 50 0"

# Arguments:

arg1 (map): 
Map type. Must be "valley" or "box".

arg2 (training): 
"1" means it runs the training optuna function. 
"0" means it runs the regular simulation.

arg3 (trials) (Only relevant when training, other set value to 1): 
Number of trials to run. Must be at least 1

arg4 (drawing) (Only relevant when training, otherwise set value to 1):
"1" means that simulation should still be drawn. (Only every 3rd or 4th picture is drawn to speed up execution)
"0" means no rendering = faster

# Visualizations
Visualization will pop up in browser after all training trials are done