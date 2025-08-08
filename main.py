import taichi as ti
import numpy as np
import time
import random
import optuna
import optuna.visualization as vis



ti.init(arch=ti.cuda, random_seed=int(time.time()))
# window = ti.ui.Window(name='window', res = (1000, 1000), fps_limit=200, pos = (150, 150))

width = 1800
height = 600
ratio = width/height
rad = 3
sigma = 40 * rad / width
epsilon = 0.1
boxHeight = 0.4
boxWidth = 0.4
types = 2

#   Parameters for general physics
amount = 500
foodAmount = 30
gravity = 0.1
max_speed = 0.0035
coll_range = 30          # Range for guranteed repulsion force in the unit for 'rad'
coll_force = 300
force_mult = 60         # Multiplier for stength of particle interaction forces

positions = ti.Vector.field(2, dtype=float, shape=(types, amount))
velocities = ti.Vector.field(2, dtype=float, shape=(types, amount))
alive = ti.field(bool, shape=(types, amount))
forces = ti.field(float, shape=(types, types))
colors = [0xff5050, 0x80ff80, 0x0000ff, 0xff8800]
gui = ti.GUI("Particle Life", res=(width, height), background_color=0x000000)


def init():
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
    forces[0, 0] = 0.1
    forces[0, 1] = 0.9
    forces[1, 0] = -0.02
    forces[1, 1] = 1

@ti.kernel
def update_vel2(tickCount: int):
    # print(tickCount < 200)
    for i, j in ti.ndrange(types, amount):
        if i == 1 and j > foodAmount-1 or alive[i, j] == False:
            continue
        force_acc = ti.Vector([0.0, 0.0])
        pos1 = positions[i, j]
        
        acc = 0

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
                
                if i == 0 and iOther == 1 and r <= sigma:
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
                acc += 1
                    
                    
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
def update_vel(tickCount: int):
    ti.loop_config(block_dim=512)

    for i, j in ti.ndrange(types, amount):
        if i == 1 and j > foodAmount-1 or alive[i, j] == False:
            continue
        force_acc = ti.Vector([0.0, 0.0])
        pos1 = positions[i, j]

        for iOther, jOther in ti.ndrange(types, amount):
            if i == 0 and iOther == 1 and tickCount < 200:
                continue
            if iOther == 1 and jOther > foodAmount-1 or alive[i, j] == False:
                continue
            if i != iOther or j != jOther:
                dir = positions[iOther, jOther] - pos1  # Vector from current to other particle 
                dist_sqr = dir.norm_sqr() + 1e-10       # 
                dist = ti.sqrt(dist_sqr)                # Distance between particles
                r = dist                                # Distance to be used for function
                
                if i == 0 and iOther == 1 and r <= sigma*1.5:
                    alive[iOther, jOther] = False
                    positions[iOther, jOther] = [-1, -1]
                    continue


                m = 12.0
                n = 6.0
                temp = sigma / r
                repulsive_force = temp ** m
                attraction_force = temp ** n
                
                force_Mult = forces[i, iOther]
                newEpsilon = epsilon * force_Mult
                
                result_force = newEpsilon * ((m * repulsive_force - n * attraction_force) / r)
                endForce = result_force * dir.normalized()

                # force_acc += endForce
                if (result_force < 0):
                    # if result_force < 0.0000001: 
                        # print(dist)
                    force_acc += endForce
                else:
                    # print(dist)
                    force = -forces[i, iOther]
                    att_force = force / dist_sqr

                    force_acc += dir * (att_force / dist)
                    
        # print(velocities[i, j])
        velocities[i, j] += force_acc / width
        # print(force_acc / width)
        # velocities[i, j] += ti.Vector([0.0, gravity])
        speed = velocities[i, j].norm()
        if speed > max_speed:
            velocities[i, j] = velocities[i, j].normalized() * max_speed
            # print(velocities[i, j].normalized() * max_speed)
        else:
            velocities[i, j] = velocities[i, j].normalized() * speed*0.98
        # force_acc += ti.Vector([0.0, -1000])
        velocities[i, j] += ti.Vector([0.0, -gravity])
        

@ti.kernel
def check_coll():
    ti.loop_config(block_dim=512)
    for i, j in ti.ndrange(types, amount):
        force_acc = ti.Vector([0.0, 0.0])
        pos1 = positions[i, j]
        for iOther, jOther in ti.ndrange(types, amount):  
            if i != iOther or j != jOther:
                dir = positions[iOther, jOther] - pos1
                dist_sqr = dir.norm_sqr() + 1e-5
                dist = ti.sqrt(dist_sqr)
                
                if dist < (4.0 * rad / width):
                    n = dir.normalized()
                    oldVel = velocities[i, j]
                    velocities[i, j] = oldVel - 1.65 * oldVel.dot(n) * n
                    
                    dir2 = -dir
                    n2 = dir2.normalized()
                    oldVel2 = velocities[iOther, jOther]
                    velocities[iOther, jOther] = oldVel2 - 1.65 * oldVel2.dot(n2) * n2
                    
                    overlap = (4.0 * rad / width) - dist
                    push = 0.5 * overlap * n
                    positions[i, j] -= push
                    positions[iOther, jOther] += push        

@ti.kernel
def move():
    for i, j in ti.ndrange(types, amount):
        if i == 1 and j > foodAmount-1 or alive[i, j] == False:
                continue
        oldPos = positions[i, j]
        newPos = positions[i, j] + velocities[i, j]
        vel = velocities[i, j]

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

def drawRectangle(p0, p1, p2, p3, color):
    
    # Draw rectangle as two triangles
    gui.triangle(p0, p1, p2, color)
    gui.triangle(p0, p2, p3, color)


def render3():
    # gui.clear(0xFFFFFF)
    np_pos = positions.to_numpy().reshape(-1, 2)
    
    # Squarepoints
    p0 = [0.0, 0.0]
    p1 = [0.4, 0.0]
    p2 = [0.4, 0.4]
    p3 = [0.0, 0.4]
    # drawRectangle(p0, p1, p2, p3, 0xAAAAAA)
    # gui.rect([0.5, 0.5], [0.6, 0.4], 0xAAAAAA)
    

    gui.rect([0, 0], [boxWidth, boxHeight], 1, color=0xAAAAAA)
    gui.rect([1-boxWidth, 0], [1, boxHeight], 1, color=0xAAAAAA)

    for i in range(types):
        gui.circles(np_pos[i * amount: (i + 1) * amount], radius=rad, color=colors[i])
        # gui.circles(np_pos[i * amount: (i + 1) * amount], radius=rad, color=colors[i])
    gui.show()

@ti.kernel
def checkTouchGround() -> bool:
    hasTouched = False
    for j in range(amount):
        if positions[0, j].y <= boxHeight + 0.01:
            hasTouched = True
    return hasTouched

def run():
    init()
    # initFood()
    # printPos()
    print("1", velocities[0, 0])
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
        update_vel2(tickCount)
        # update_vel(tickCount)
        # print("2", velocities[0, 0])
        # check_coll()
        move()
        # render()
        render3()
        # gui.show()
        # window.show()




# ================================================================
#         Code for RL training
# ================================================================




@ti.kernel
def cntEaten() -> int:
    cnt = 0
    for j in ti.ndrange(foodAmount):
        if not alive[1, j]:
            cnt += 1
    print(cnt)
    return cnt

@ti.kernel
def particleCloseReward(maxPoints: int) -> int: 
    maxDist = ti.sqrt(boxHeight + boxWidth)
    reward = 0
    for j in ti.ndrange(amount):
        particlePos = positions[0, j]
        distSum = 0
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



            



def objective(trial):
    # Example: optimize forces[0,0]
    force_00 = trial.suggest_float('force_00', 0, 1.0)
    force_01 = trial.suggest_float('force_01', -1.0, 1.0)
    
    # Reset the simulation to the initial state
    init()

    # Apply to your force field
    forces[0, 0] = force_00
    forces[0, 1] = force_01

    total_reward = 0.0
    tickCount = 0
    hasTouchedGround = False
    second = 0
    # Run the simulation for N steps
    for step in range(5000):
        if not hasTouchedGround:
            hasTouchedGround = checkTouchGround()
        else:
            tickCount += 1
        
        update_vel2(tickCount)
        move()
        # check_coll()  # Optional
        # if second % 1 == 0:
            # render3()
        # second += 1 
        
        # Evaluate reward at each step
        # Example: reward could be distance minimization between type 0 and type 1
        # reward = compute_reward()
        # total_reward += reward

    total_reward += cntEaten() * 100
    total_reward += particleCloseReward(1)

    # Return cumulative reward (maximize this)
    return total_reward

def train():
    study = optuna.create_study(
        direction='maximize', 
        storage="sqlite:///db.sqlite3", 
        study_name="Forces", 
        load_if_exists=True
        )
    study.optimize(objective, n_trials=5)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

# train()
run()