import taichi as ti
import numpy as np
import time
import random
import optuna
import optuna.visualization as vis
import sys

# read command line arguments (Used to run specific funtion at file end)
map = sys.argv[1]
training = int(sys.argv[2])
trials = int(sys.argv[3])
drawing = int(sys.argv[3])

# Initializes Taichi - Use GPU
ti.init(arch=ti.gpu)

# Definition of GUI dimensions
width = 1800
height = 600

# Box dimensions for the valley (Left and right side)
boxHeight = 0.4
boxWidth = 0.4

# Particle radius and number of particle types
rad = 3
types = 2

# Colors for particle types
colors = [0xff5050, 0x80ff80, 0x8080ff, 0xffAA00]

# Parameters for general physics
amount = 500                        # Number of main particles
foodAmount = 30                     # Number of food particles when in valley
coll_range = 30                     # Range for guranteed repulsion force in the unit for 'rad'
force_mult = 60                     # Multiplier for stength of particle interaction forces

# Standards for physics (Can be overwritten for training phase)
glob_gravity = 0.1              
glob_max_speed = 0.0035             # Maximum speed pr. tick
glob_coll_force = 300               # Repulsion force when in range of coll_range

# Fields to hold data on all particles
positions = ti.Vector.field(2, dtype=float, shape=(types, amount))
velocities = ti.Vector.field(2, dtype=float, shape=(types, amount))
alive = ti.field(bool, shape=(types, amount))

# Field for definition of forces
forces = ti.field(float, shape=(types, types))

# Taichi GUI setup
gui = ti.GUI("Particle Life", res=(width, height), background_color=0x000000)


# Initializes environment for valley map
def initValley():
    # Initialize all main particles
    for j in range(amount):
        positions[0, j] = [random.random()*boxWidth, (random.random()*0.5) + boxHeight + 0.05]
        velocities[0, j] = [0.0, 0.0]
        alive[0, j] = True
    
    # Initialize all food particles
    for k in range(foodAmount):
        positions[1, k] = [random.random()*boxWidth + (1-boxWidth), random.random()*0.5 + boxHeight+0.05]
        velocities[1, k] = [0.0, 0.0]
        alive[1, k] = True
    
    # Set forces for particle interaction (Positive = attraction)
    forces[0, 0] = 0.15
    forces[0, 1] = 0.9
    forces[1, 0] = -0.02
    forces[1, 1] = 1


# Initializes environment for box map
def initBox():
    # Initialize all main particles
    for j in range(amount):
        positions[0, j] = [random.random(), (random.random())]
        velocities[0, j] = [0.0, 0.0]
        alive[0, j] = True
        
    # Set force for particle attraction
    forces[0, 0] = 0.5


# Function for updating all particle velocities
@ti.kernel
def update_vel(tickCount: int, max_speed: float, coll_force: int, gravity: float):
    
    # Loop though all particles
    for i, j in ti.ndrange(types, amount):
        
        # Skip dead particles
        if alive[i, j] == False:
            continue
        
        force_acc = ti.Vector([0.0, 0.0])
        pos1 = positions[i, j]

        # Loop though all particles
        for iOther, jOther in ti.ndrange(types, amount):
            
            # Particles does not interact with food before cooldown
            if i == 0 and iOther == 1 and tickCount < 200:
                continue
            
            # Skip dead particles
            if alive[iOther, jOther] == False:
                continue
            
            # Continiue if not the same particle
            if i != iOther or j != jOther:
                dir = positions[iOther, jOther] - pos1  # Vector from current to other particle 
                dist_sqr = dir.norm_sqr() + 1e-10
                dist = ti.sqrt(dist_sqr)                # Distance between particles
                
                # Food is eaten if main particle type is within range
                if i == 0 and iOther == 1 and dist <= (40 * rad / width):
                    alive[iOther, jOther] = False
                    positions[iOther, jOther] = [-1, 1]
                    continue
                
                result_force = 0.0
                
                # Same type is repelled if within range
                if dist < (coll_range) * rad / width and i == iOther:
                    result_force = -coll_force
                    
                # Otherwise apply the defined force
                else:
                    result_force = force_mult * forces[i, iOther]
                
                # Accumulate the force in direction of other particle
                endForce = result_force * dir.normalized()
                force_acc += endForce
        
        # If force_acc is NAN then set as [0, 0] to avoid errors
        if ti.math.isnan(force_acc.x) or ti.math.isnan(force_acc.y):
            force_acc = ti.Vector([0.0, 0.0])
        
        # Apply force to accumulate velocity
        velocities[i, j] += force_acc / width
        
        # Keep velocity within max_speed
        speed = velocities[i, j].norm()
        if speed > max_speed:
            velocities[i, j] = velocities[i, j].normalized() * max_speed
        
        # Apply gravity force
        velocities[i, j] += ti.Vector([0.0, -gravity*0.015])
        
        # Food has additional gravity to right stay in place
        if i == 1:
            velocities[i, j] += ti.Vector([gravity*0.015, 0.0])
   

# Function to move particles on valley map
@ti.kernel
def move_valley():
    
    # Loop through all particles
    for i, j in ti.ndrange(types, amount):
        
        # Skip dead particles
        if alive[i, j] == False:
                continue
        
        # Old position and initial new position
        oldPos = positions[i, j]
        newPos = positions[i, j] + velocities[i, j]
        vel = velocities[i, j]
        
        # Particle dies when bottom of valley is touched
        if newPos.y < 0.01:
            alive[i, j] = False
            positions[i, j] = [-1, 1]
            continue

        # The following keeps particles withing the map:
        
        # When in the valley:
        if oldPos.x >= boxWidth and oldPos.x <= 1-boxWidth and oldPos.y < boxHeight:
            positions[i, j] = ti.Vector([
                max(boxWidth, min(newPos.x, 1-boxWidth)), 
                max(0.0, min(newPos.y, 1.0))
            ])
            
        # When above one of the boxes
        elif oldPos.x < boxWidth or oldPos.x > 1-boxWidth:
            positions[i, j] = ti.Vector([
                max(0.0, min(newPos.x, 1.0)), 
                max(boxHeight, min(newPos.y, 1.0))
            ])
            
            # Bounce on box
            if vel.y < 0 and oldPos.y > boxHeight and positions[i, j].y == boxHeight:
                velocities[i, j] = ti.Vector([
                    vel.x,
                    -1*vel.y
                ])
        
        # Otherwise keep withing screen
        else:
            positions[i, j] = ti.Vector([
                max(0.0, min(newPos.x, 1.0)), 
                max(0.0, min(newPos.y, 1.0))
            ])
  
# Function to move particles on box map        
@ti.kernel
def move_box():
    
    # Loop through all particles
    for j in ti.ndrange(amount):
        
        # Old position and initial new position
        oldPos = positions[0, j]
        newPos = positions[0, j] + velocities[0, j]
        vel = velocities[0, j]
        
        # Keep withing screen
        positions[0, j] = ti.Vector([
                max(0.0, min(newPos.x, 1.0)), 
                max(0.0, min(newPos.y, 1.0))
        ])
        
        # Bounce on floor
        if vel.y < 0 and oldPos.y > 0 and positions[0, j].y == 0.0:
            velocities[0, j] = ti.Vector([
                    vel.x,
                    -1*vel.y
                ])


# Function for rendering / drawing
def render(hasValley):
    
    # Numpy with all particles
    np_pos = positions.to_numpy().reshape(-1, 2)
    
    # Draw the two boxes if on valley map
    if hasValley:
        gui.rect([0, 0], [boxWidth, boxHeight], 1, color=0xAAAAAA)
        gui.rect([1-boxWidth, 0], [1, boxHeight], 1, color=0xAAAAAA)

    # Draw all particles
    for i in range(types):
        gui.circles(np_pos[i * amount: (i + 1) * amount], radius=rad, color=colors[i])
    
    # Show the GUI
    gui.show()


# Function checks if a particle touches roof of a box
@ti.kernel
def checkTouchGround() -> bool:
    hasTouched = False
    
    # Tests for all particle y-position
    for j in range(amount):
        if positions[0, j].y <= boxHeight + 0.01:
            hasTouched = True
    return hasTouched


# Run function for valley map
def run_valley():
    
    # Initialize
    initValley()
    
    # Get physics variables
    gravity = glob_gravity
    max_speed = glob_max_speed
    coll_force = glob_coll_force
    
    # Values for loop (Ticker starts after a particle touches ground)
    hasTouchedGround = False
    tickCount = 0
    
    # Main execution loop
    while gui.running:
        # Check for ground contact
        if not hasTouchedGround:
            hasTouchedGround = checkTouchGround()
        # Start counter after contact
        else:
            tickCount += 1
        
        # Update velocities
        update_vel(tickCount, max_speed, coll_force, gravity)
        
        # Move all particles
        move_valley()
        
        # Draw all particles and map
        render(hasValley=True)

# Run function for box map
def run_box():
    
    # Initialize
    initBox()
    
    # Get physics variables
    gravity = glob_gravity
    max_speed = glob_max_speed
    coll_force = glob_coll_force
    
    # Main execution loop
    while gui.running:
        
        # Update velocities
        update_vel(0, max_speed, coll_force, gravity)
        
        # Move alle particles
        move_box()
        
        # Draw all particles and map
        render(hasValley=False)


# ================================================================
#         Code for RL training
# ================================================================



# Function to count eaten food particles
@ti.kernel
def cntEaten() -> int:
    cnt = 0
    
    # Loop through food particles and accumulate for dead ones
    for j in ti.ndrange(foodAmount):
        if not alive[1, j]:
            cnt += 1
            
    print("Eaten", cnt)
    return cnt


# Function to reward particle proximity to food
@ti.kernel
def particleCloseReward(maxPoints: int) -> int: 
    maxDist = ti.sqrt(2)        # Diagonal distance of GUI (Taichi GUI positions are defined from 0.0-1.0 in each dimension)
    reward = 0.0
    
    # Loop through main particles
    for j in ti.ndrange(amount):
        
        # Skip dead particles
        if not alive[0, j]:
            continue
        
        # Read particle position
        particlePos = positions[0, j]
        
        distSum = 0.0
        aliveCnt = 0
        
        # Loop through food particles
        for jj in ti.ndrange(foodAmount):
            # Skip dead particles
            if alive[1, jj]:
                foodPos = positions[1, jj]          # Read food position
                dir = foodPos - particlePos         # Vector from current to other particle 
                dist_sqr = dir.norm_sqr() + 1e-10       
                dist = ti.sqrt(dist_sqr)            # Distance between particles
                
                # Accumulate distance and counter
                distSum += dist
                aliveCnt += 1
        
        # Compute average distance to food
        avgDist = distSum / aliveCnt
        
        # Compare average distance to max possible distance and add reward
        scaledDist = avgDist / maxDist
        reward += (1 - scaledDist) * maxPoints
        
    print("ParticleCloseReward = ", reward)
    return reward

# Function to compute average nearby particles
@ti.kernel
def clustering_level_max(range: float) -> float:
    in_range_sum = 0
    
    # Loop through particles
    for j in ti.ndrange(amount):
        
        # Get particle position
        pos1 = positions[0, j]
        cnt = 0
        
        # Loop throgh all particles
        for jj in ti.ndrange(amount):
            # The particle skips itself
            if j == jj:
                continue
            
            pos2 = positions[0, jj]             # Get other particle position
            dir =  pos2 - pos1                  # Vector from current to other particle 
            dist_sqr = dir.norm_sqr() + 1e-10
            dist = ti.sqrt(dist_sqr)            # Distance between particles
            
            # Accumulate count if withing range
            if dist < range:
                cnt += 1
        # Add count to sum
        in_range_sum += cnt
        
    # Print and return the average particles within range     
    print("Average particles in range", range, ":", in_range_sum / amount)
    return in_range_sum / amount




            


# Objective function for training on valley map
def objective_forces_valley(trial):
    # Optuna suggests force values for main particle
    force_00 = trial.suggest_float('force_00', 0.1, 1.0)
    force_01 = trial.suggest_float('force_01', -1.0, 1.0)
    
    # Reset the simulation to the initial state
    initValley()

    # Apply suggestions to force field
    forces[0, 0] = force_00
    forces[0, 1] = force_01

    # Get physics variables
    gravity = glob_gravity
    max_speed = glob_max_speed
    coll_force = glob_coll_force
    
    # Values for loop (Ticker starts after a particle touches ground)
    total_reward = 0.0
    tickCount = 0
    hasTouchedGround = False
    cnt = 0
    
    # Run the simulation for N steps / ticks
    for step in range(4000):
        # Check for ground contact
        if not hasTouchedGround:
            hasTouchedGround = checkTouchGround()
        # Start counter after contact
        else:
            tickCount += 1
        
        # Update velocities
        update_vel(tickCount, max_speed, coll_force, gravity)
        
        # Move all particles
        move_valley()
        
        cnt += 1
        if cnt % 4 == 0 and drawing == 1:
            render(True)

    # Add reward for amount of food particles eaten
    total_reward += cntEaten() * 100
    
    # Add reward for proximity to food for alive none-food-particles
    total_reward += particleCloseReward(1)

    # Return total reward
    return total_reward


# Objective function for training on box map
def objective_clusterization_box(trial):
    # Optuna suggests physics variable values
    gravity = trial.suggest_float('gravity', 0.0, 0.2)
    coll_force = trial.suggest_int('coll_force', 0, 1000)
    max_speed = trial.suggest_float('max_speed', 0.0001, 0.08)
    
    # Reset the simulation to the initial state
    initBox()
    
    # Values for loop
    total_reward = 0.0
    cnt = 0
    
    # Run the simulation for N steps / ticks
    for step in range(2000):
        # Update velocities
        update_vel(0, max_speed, coll_force, gravity)
        
        # Move all particles
        move_box()
        
        cnt += 1
        if cnt % 3 == 0 and drawing == 1:
            render(False)

    # Add reward for average number of particles within '5 * radius'
    total_reward += clustering_level_max((5 * rad) / width)

    # Return total reward
    return total_reward


# Training function for valley map
def train_forces_valley(trials):
    # Create study
    study = optuna.create_study(
        direction='maximize', 
        storage="sqlite:///db.sqlite3", 
        study_name="Forces", 
        load_if_exists=True
        )
    
    # Optimize study with objective function
    study.optimize(objective_forces_valley, n_trials=trials)

    # Print best result and parameter values:
    print("Best trial:")
    trial = study.best_trial

    print(f" Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Draw visualizations
    optuna.visualization.plot_slice(study, params=['force_00', 'force_01']).show()
  
  
# Training function for box map    
def train_clusterization_box(trials):
    # Create study
    study = optuna.create_study(
        direction='maximize', 
        storage="sqlite:///db.sqlite3", 
        study_name="Clusterization", 
        load_if_exists=True
        )
    
    # Optimize study with objective function
    study.optimize(objective_clusterization_box, n_trials=trials)

    # Print best result and parameter values:
    print("Best trial:")
    trial = study.best_trial
    print(f" Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Draw visualizations
    optuna.visualization.plot_slice(study, params=['gravity', 'coll_force', 'max_speed']).show()
     
        
# train_forces_valley()
# train_clusterization_box()
# run_valley()
# run_box()

if __name__ == "__main__":
    # Check if argument 2, 3 and 4 is allowed
    if training != 0 and training != 1:
        print("Second argument must be 1 or 0")
    if trials < 1:
        print("Third argument must be at least 1")
    if drawing != 0 and drawing != 1:
        print("Fourth argument must be 1 or 0")
    
    # Run function based on arguments
    if map == "valley":
        if training == 1:
            train_forces_valley(trials)
        else:
            run_valley() 
            
    elif map == "box":
        if training == 1:
            train_clusterization_box(trials)
        else:
            run_box()
    else:
        print("First argument must be 'valley' or 'box'")