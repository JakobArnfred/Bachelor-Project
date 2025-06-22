import taichi as ti
import numpy as np
import time


ti.init(arch=ti.cpu, random_seed=int(time.time()))
# window = ti.ui.Window(name='window', res = (1000, 1000), fps_limit=200, pos = (150, 150))

width = 1600
height = 800
ratio = width/height
amount = 100
types = 2
rad = 2
max_speed = 0.01
gravity = 0.001
sigma = 40 * rad / width
epsilon = 1
boxHeight = 0.4
boxWidth = 0.35
foodAmount = 40


positions = ti.Vector.field(2, dtype=float, shape=(types, amount))
velocities = ti.Vector.field(2, dtype=float, shape=(types, amount))
forces = ti.field(float, shape=(types, types))
colors = [0xff0000, 0x00ff00, 0x0000ff, 0xff8800]
gui = ti.GUI("Particle Life", res=(width, height), background_color=0x000000)

def printPos():
    for i, j in ti.ndrange(types, amount):
        print(positions[i, j])

@ti.kernel
def init():
    # for i, j in ti.ndrange(types, amount):
    #     positions[i, j] = [ti.random()*0.25, (ti.random()*0.5)+0.5]
    #     # velocities[i, j] = [ti.randn()/width, ti.randn()/height]
    #     velocities[i, j] = [0.0, 0.0]
    for j in range(amount):
        positions[0, j] = [ti.random()*boxWidth, (ti.random()*0.5) + boxHeight + 0.05]
        # velocities[i, j] = [ti.randn()/width, ti.randn()/height]
        velocities[0, j] = [0.0, 0.0]
        # print("k", velocities[0, 0])
    for k in range(foodAmount):
        # print(1, velocities[0, 0])
        positions[1, k] = [ti.random()*boxWidth + (1-boxWidth), ti.random()*0.5 + boxHeight+0.05]
        # print(2, velocities[0, 0])
        velocities[1, k] = [0.0, 0.0]
        # print(3, velocities[0, 0])
    # print(velocities[0, 0])
    forces[0, 0] = -0.1
    forces[0, 1] = -1
    forces[1, 0] = 0
    forces[1, 1] = -0.1
        
    # print(velocities[0, 0])
    # forces[0, 0] = 0.5
    # forces[1, 1] = 0.5
    # forces[2, 2] = 0.5
    # forces[2, 1] = -0.5

@ti.kernel
def update_vel():
    ti.loop_config(block_dim=512)

    for i, j in ti.ndrange(types, amount):
        if i == 1 and j > foodAmount-1:
            continue
        force_acc = ti.Vector([0.0, 0.0])
        pos1 = positions[i, j]

        for iOther, jOther in ti.ndrange(types, amount):
            if iOther == 1 and jOther > foodAmount-1:
                continue
            if i != iOther or j != jOther:
                dir = positions[iOther, jOther] - pos1  # Vector from current to other particle 
                dist_sqr = dir.norm_sqr() + 1e-10       # 
                dist = ti.sqrt(dist_sqr)                # Distance between particles
                r = dist                                # Distance to be used for function
                # print(dir, dist_sqr, dist)
                
                # if (dist > 4*sigma):
                #     r = ti.sqrt(ti.sqrt(dist))
                # #     # print((4*sigma), dist)
                
                # repulsive_force = (sigma**12) / (dist_sqr**6)
                # attraction_force = (sigma**6) / (dist_sqr**3)
                # if dist < 7*sigma:
                #     m = 6.0
                #     n = 4.0
                #     temp = sigma / r
                #     repulsive_force = temp ** m
                #     attraction_force = temp ** n
                    
                    
                #     rep_f = (sigma / dist)
                    
                #     force_Mult = forces[i, iOther]
                #     newEpsilon = epsilon * force_Mult
                    
                #     result_force = newEpsilon * (m * repulsive_force - n * attraction_force) / r
                #     endForce = result_force * dir.normalized()

                #     force_acc += endForce
                # else:
                #     force = -forces[i, iOther]
                #     att_force = force / dist_sqr

                #     force_acc += dir * (att_force / dist)


                m = 6.0
                n = 4.0
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
                
                # if force_Mult < 0 and dist <= sigma:
                #     newEpsilon = epsilon * force_Mult
                # else:
                #     newEpsilon = epsilon * force_Mult

                # result_force = 24 * newEpsilon * (2 * repulsive_force - attraction_force) / dist

                # print("hhh")
                # if force_Mult < 0:
                #     print("ggg")
                #     endForce = -abs(result_force) * dir.normalized()
                    
                # if (endForce > 0):
                #     endForce *= 10

                # force = forces[i, iOther]
                # att_force = force / dist_sqr
                # repul_force = force

                # force_acc += dir * (att_force / dist)
                    
        # print(velocities[i, j])
        velocities[i, j] += force_acc / width
        # print(force_acc / width)
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

init()
# printPos()
timer = 0
# gui.show()
while gui.running:
# while window.running:
    # gui.set_image()
    update_vel()
    # check_coll()
    move()
    # render()
    render3()
    # gui.show()
    # window.show()