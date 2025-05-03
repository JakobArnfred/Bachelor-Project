import taichi as ti
import numpy as np
import time


ti.init(arch=ti.gpu, random_seed=int(time.time()))
# window = ti.ui.Window(name='window', res = (1000, 1000), fps_limit=200, pos = (150, 150))

width = 800
height = 800
ratio = width/height
amount = 80
types = 1
rad = 2
max_speed = 0.005
sigma = 3 * rad / width
epsilon = 0.5


positions = ti.Vector.field(2, dtype=float, shape=(types, amount))
velocities = ti.Vector.field(2, dtype=float, shape=(types, amount))
forces = ti.field(float, shape=(types, types))
colors = [0xff0000, 0x00ff00, 0x0000ff, 0xff8800]
# colors = [
#     (1.0, 0.0, 0.0),  # Red
#     (0.0, 1.0, 0.0),  # Green
#     (0.0, 0.0, 1.0),  # Blue
#     (1.0, 0.5, 0.0)   # Orange
# ]
gui = ti.GUI("Particle Life", res=(width, height))
# window = ti.ui.Window(name='Particle Life', res=(width, height), fps_limit=60)

# canvas = window.get_canvas()

def printPos():
    for i, j in ti.ndrange(types, amount):
        print(positions[i, j])

@ti.kernel
def init():
    for i, j in ti.ndrange(types, amount):
        positions[i, j] = [ti.random()*0.25, (ti.random()*0.5)+0.5]
        # velocities[i, j] = [ti.randn()/width, ti.randn()/height]
        velocities[i, j] = [0.0, 0.0]
    forces[0, 0] = 0.0005
    forces[0, 1] = -0.00001
    forces[1, 0] = 0.00001
    forces[1, 1] = 0.00001
    forces[2, 2] = 0.00001
    forces[2, 1] = -0.00001
    forces[2, 0] = -0.00001
    # forces[0, 0] = 0.5
    # forces[1, 1] = 0.5
    # forces[2, 2] = 0.5
    # forces[2, 1] = -0.5

@ti.kernel
def update_vel():
    ti.loop_config(block_dim=512)

    for i, j in ti.ndrange(types, amount):
        force_acc = ti.Vector([0.0, 0.0])
        pos1 = positions[i, j]

        for iOther, jOther in ti.ndrange(types, amount):
            if i != iOther or j != jOther:
                # force = forces[i, iOther]
                dir = positions[iOther, jOther] - pos1
                dist_sqr = dir.norm_sqr() + 1e-10
                dist = ti.sqrt(dist_sqr)
                
                repulsive_force = (sigma**12) / (dist_sqr**6)
                attraction_force = (sigma**6) / (dist_sqr**3)

                result_force = 24 * epsilon * (2 * repulsive_force - attraction_force) / dist_sqr
                endForce = result_force * dir.normalized()
                
                # if (endForce > 0):
                #     endForce *= 10

                force_acc += endForce

                # force = forces[i, iOther]
                # att_force = force / dist_sqr
                # repul_force = force

                # force_acc += dir * (att_force / dist)
                    
        force_acc += ti.Vector([0.0, -0.8])
        velocities[i, j] += force_acc / width
        speed = velocities[i, j].norm()
        if speed > max_speed:
            velocities[i, j] = velocities[i, j].normalized() * max_speed

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
        newPos = positions[i, j] + velocities[i, j]
                
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
    drawRectangle(p0, p1, p2, p3, 0xAAAAAA)
    

    # gui.rect([0, 0], [0.4, 0.4], 1, color=0x999999)

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