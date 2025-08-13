import taichi as ti
import numpy as np
import time
import random
import optuna
import optuna.visualization as vis



ti.init(arch=ti.cuda, random_seed=int(time.time()))

width = 1800
height = 600

boxHeight = 0.4
boxWidth = 0.4

rad = 3
types = 2

#   Parameters for general physics
amount = 500
foodAmount = 30
coll_range = 30          # Range for guranteed repulsion force in the unit for 'rad'
force_mult = 60         # Multiplier for stength of particle interaction forces

glob_gravity = 0.1
glob_max_speed = 0.0035
glob_coll_force = 300

positions = ti.Vector.field(2, dtype=float, shape=(types, amount))
velocities = ti.Vector.field(2, dtype=float, shape=(types, amount))
alive = ti.field(bool, shape=(types, amount))
forces = ti.field(float, shape=(types, types))
colors = [0xff5050, 0x80ff80, 0x0000ff, 0xff8800]
gui = ti.GUI("Particle Life", res=(width, height), background_color=0x000000)


def initValley():
    for j in range(amount):
        positions[0, j] = [random.random()*boxWidth, (random.random()*0.5) + boxHeight + 0.05]
        # velocities[i, j] = [ti.randn()/width, ti.randn()/height]
        velocities[0, j] = [0.0, 0.0]
        alive[0, j] = True
        # print("k", velocities[0, 0])
        # print(j)
    for k in range(foodAmount):
        positions[1, k] = [random.random()*boxWidth + (1-boxWidth), random.random()*0.5 + boxHeight+0.05]
        velocities[1, k] = [0.0, 0.0]
        alive[1, k] = True
    forces[0, 0] = 0.15
    forces[0, 1] = 0.9
    forces[1, 0] = -0.02
    forces[1, 1] = 1

def initBox():
    for j in range(amount):
        positions[0, j] = [random.random(), (random.random())]
        velocities[0, j] = [0.0, 0.0]
        alive[0, j] = True
    forces[0, 0] = 0.5

@ti.kernel
def update_vel(tickCount: int, max_speed: float, coll_force: int, gravity: float):
    # print(tickCount < 200)
    for i, j in ti.ndrange(types, amount):
        if i == 1 and j > foodAmount-1 or alive[i, j] == False:
            continue
        force_acc = ti.Vector([0.0, 0.0])
        pos1 = positions[i, j]

        for iOther, jOther in ti.ndrange(types, amount):
            if i == 0 and iOther == 1 and tickCount < 200:
                continue
            if iOther == 1 and jOther > foodAmount-1 or alive[iOther, jOther] == False:
                continue
            if i != iOther or j != jOther:
                dir = positions[iOther, jOther] - pos1  # Vector from current to other particle 
                dist_sqr = dir.norm_sqr() + 1e-10       # 
                dist = ti.sqrt(dist_sqr)                # Distance between particles
                r = dist                                # Distance to be used for function
                
                if i == 0 and iOther == 1 and r <= (40 * rad / width):
                    alive[iOther, jOther] = False
                    positions[iOther, jOther] = [-1, 1]
                    continue
                
                result_force = 0.0
                
                # if r < (coll_range * forces[i, iOther]) * rad / width and i == iOther:
                if r < (coll_range) * rad / width and i == iOther:
                    result_force = -coll_force
                else:
                    result_force = force_mult * forces[i, iOther]
                
                endForce = result_force * dir.normalized()

                force_acc += endForce
                # if (result_force < 0):
                #     # if result_force < 0.0000001: 
                #         # print(dist)
                #     force_acc += endForce
                # else:
                #     # print(dist)
                #     force = -forces[i, iOther]
                #     att_force = force / dist_sqr

                #     force_acc += dir * (att_force / dist)
                    
                    
        # force_acc += ti.Vector([0.0, -gravity]) * amount  
        # if i == 1:
        #     force_acc += ti.Vector([gravity, 0.0])
            # print(j, positions[i, j], force_acc, acc)
        
        if ti.math.isnan(force_acc.x) or ti.math.isnan(force_acc.y):
            force_acc = ti.Vector([0.0, 0.0])
        
        velocities[i, j] += force_acc / width
        speed = velocities[i, j].norm()
        if speed > max_speed:
            velocities[i, j] = velocities[i, j].normalized() * max_speed
        velocities[i, j] += ti.Vector([0.0, -gravity*0.015])
        if i == 1:
            velocities[i, j] += ti.Vector([gravity*0.015, 0.0])
   

@ti.kernel
def move_valley():
    for i, j in ti.ndrange(types, amount):
        if i == 1 and j > foodAmount-1 or alive[i, j] == False:
                continue
        oldPos = positions[i, j]
        newPos = positions[i, j] + velocities[i, j]
        vel = velocities[i, j]
        
        if newPos.y < 0.01:
            alive[i, j] = False
            positions[i, j] = [-1, 1]
            continue

        if oldPos.x >= boxWidth and oldPos.x <= 1-boxWidth and oldPos.y < boxHeight:
            positions[i, j] = ti.Vector([
                max(boxWidth, min(newPos.x, 1-boxWidth)), 
                max(0.0, min(newPos.y, 1.0))
            ])
        elif oldPos.x < boxWidth or oldPos.x > 1-boxWidth:
            positions[i, j] = ti.Vector([
                max(0.0, min(newPos.x, 1.0)), 
                max(boxHeight, min(newPos.y, 1.0))
            ])
            if vel.y < 0 and oldPos.y > boxHeight and positions[i, j].y == boxHeight:
                velocities[i, j] = ti.Vector([
                    vel.x,
                    -1*vel.y
                ])
                # print(vel, velocities[i, j])
        else:
            positions[i, j] = ti.Vector([
                max(0.0, min(newPos.x, 1.0)), 
                max(0.0, min(newPos.y, 1.0))
            ])
        # positions[i, j] += velocities[i, j]
        # positions[i, j].x = max(0.0, min(positions[i, j].x, 1.0))
        # positions[i, j].y = max(0.0, min(positions[i, j].y, 1.0))
        
@ti.kernel
def move_box():
    for j in ti.ndrange(amount):
        oldPos = positions[0, j]
        newPos = positions[0, j] + velocities[0, j]
        vel = velocities[0, j]
        
        positions[0, j] = ti.Vector([
                max(0.0, min(newPos.x, 1.0)), 
                max(0.0, min(newPos.y, 1.0))
        ])
        
        if vel.y < 0 and oldPos.y > 0 and positions[0, j].y == 0.0:
            velocities[0, j] = ti.Vector([
                    vel.x,
                    -1*vel.y
                ])


def render(hasValley):
    # gui.clear(0xFFFFFF)
    np_pos = positions.to_numpy().reshape(-1, 2)
    
    if hasValley:
        gui.rect([0, 0], [boxWidth, boxHeight], 1, color=0xAAAAAA)
        gui.rect([1-boxWidth, 0], [1, boxHeight], 1, color=0xAAAAAA)

    for i in range(types):
        gui.circles(np_pos[i * amount: (i + 1) * amount], radius=rad, color=colors[i])
    gui.show()

@ti.kernel
def checkTouchGround() -> bool:
    hasTouched = False
    for j in range(amount):
        if positions[0, j].y <= boxHeight + 0.01:
            hasTouched = True
    return hasTouched

def run_valley():
    initValley()
    
    gravity = glob_gravity
    max_speed = glob_max_speed
    coll_force = glob_coll_force
    # initFood()
    # printPos()
    hasTouchedGround = False
    tickCount = 0
    # gui.show()
    while gui.running:
    # while window.running:
        # gui.set_image()
        # update_vel()
        if not hasTouchedGround:
            hasTouchedGround = checkTouchGround()
        else:
            tickCount += 1
        update_vel(tickCount, max_speed, coll_force, gravity)
        # update_vel(tickCount)
        # print("2", velocities[0, 0])
        # check_coll()
        move_valley()
        # render()
        render(hasValley=True)
        # cntEaten()
        # gui.show()
        # window.show()

def run_box():
    initBox()
    gravity = glob_gravity
    max_speed = glob_max_speed
    coll_force = glob_coll_force
    while gui.running:
        update_vel(0, max_speed, coll_force, gravity)
        move_box()
        render(hasValley=False)


# ================================================================
#         Code for RL training
# ================================================================




@ti.kernel
def cntEaten() -> int:
    cnt = 0
    for j in ti.ndrange(foodAmount):
        if not alive[1, j]:
            cnt += 1
    print("Eaten", cnt)
    return cnt

@ti.kernel
def particleCloseReward(maxPoints: int) -> int: 
    maxDist = ti.sqrt(2)        # Diagonal distance of GUI (Taichi GUI positions are defined from 0.0-1.0 in each dimension)
    reward = 0.0
    for j in ti.ndrange(amount):
        if not alive[0, j]:
            continue
        particlePos = positions[0, j]
        distSum = 0.0
        aliveCnt = 0
        for jj in ti.ndrange(foodAmount):
            # print("123", alive[1, jj])
            if alive[1, jj]:
                foodPos = positions[1, jj]
                dir = foodPos - particlePos  # Vector from current to other particle 
                dist_sqr = dir.norm_sqr() + 1e-10       
                dist = ti.sqrt(dist_sqr)
                
                distSum += dist
                aliveCnt += 1
        avgDist = distSum / aliveCnt
        scaledDist = avgDist / maxDist
        reward += (1 - scaledDist) * maxPoints
    print("ParticleCloseReward = ", reward)
    return reward

@ti.kernel
def clustering_level_max(range: float) -> float:
    in_range_sum = 0
    for j in ti.ndrange(amount):
        pos1 = positions[0, j]
        cnt = 0
        for jj in ti.ndrange(amount):
            if j == jj:
                continue
            pos2 = positions[0, jj]
            dir =  pos2 - pos1  # Vector from current to other particle 
            dist_sqr = dir.norm_sqr() + 1e-10       
            dist = ti.sqrt(dist_sqr)
            
            if dist < range:
                cnt += 1
        in_range_sum += cnt        
    print("Average particles in range", range, ":", in_range_sum / amount)
    return in_range_sum / amount




            



def objective_forces_valley(trial):
    # Example: optimize forces[0,0]
    force_00 = trial.suggest_float('force_00', 0.1, 1.0)
    force_01 = trial.suggest_float('force_01', -1.0, 1.0)
    
    # Reset the simulation to the initial state
    initValley()

    # Apply to force field
    forces[0, 0] = force_00
    forces[0, 1] = force_01

    gravity = glob_gravity
    max_speed = glob_max_speed
    coll_force = glob_coll_force
    
    # print(max_speed, coll_force, gravity)
    
    total_reward = 0.0
    tickCount = 0
    hasTouchedGround = False
    cnt = 0
    # Run the simulation for N steps / ticks
    for step in range(4000):
        if not hasTouchedGround:
            hasTouchedGround = checkTouchGround()
        else:
            tickCount += 1
        
        update_vel(tickCount, max_speed, coll_force, gravity)
        move_valley()
        cnt += 1
        if cnt % 3 == 0:
            render(True)

    # Add reward for amount of food particles eaten
    total_reward += cntEaten() * 100
    
    # Add reward for proximity to food for alive none-food-particles
    total_reward += particleCloseReward(1)

    # Return total reward
    return total_reward


def objective_clusterization_box(trial):
    # Example: optimize forces[0,0]
    gravity = trial.suggest_float('gravity', 0.0, 0.2)
    coll_force = trial.suggest_int('coll_force', 0, 1000)
    max_speed = trial.suggest_float('max_speed', 0.0001, 0.08)
    
    # Reset the simulation to the initial state
    initBox()
    
    total_reward = 0.0
    cnt = 0
    # Run the simulation for N steps / ticks
    for step in range(2000):
        update_vel(0, max_speed, coll_force, gravity)
        move_box()
        cnt += 1
        if cnt % 2 == 0:
            render(False)

    # print(positions)
    total_reward += clustering_level_max((5 * rad) / width)

    # Return total reward
    return total_reward

def train_forces_valley():
    study = optuna.create_study(
        direction='maximize', 
        storage="sqlite:///db.sqlite3", 
        study_name="Forces", 
        load_if_exists=True
        )
    study.optimize(objective_forces_valley, n_trials=20)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    optuna.visualization.plot_slice(study, params=['force_00', 'force_01']).show()
        
def train_clusterization_box():
    study = optuna.create_study(
        direction='maximize', 
        storage="sqlite:///db.sqlite3", 
        study_name="Clusterization", 
        load_if_exists=True
        )
    study.optimize(objective_clusterization_box, n_trials=20)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    optuna.visualization.plot_slice(study, params=['gravity', 'coll_force', 'max_speed']).show()
        
# train_forces_valley()
# train_clusterization_box()
run_valley()
# run_box()