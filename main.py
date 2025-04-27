import taichi as ti
import time


ti.init(arch=ti.cuda, random_seed=int(time.time()))
# window = ti.ui.Window(name='window', res = (1000, 1000), fps_limit=200, pos = (150, 150))

width = 800
height = 800
ratio = width/height
amount = 100
types = 3
rad = 5

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
        positions[i, j] = [ti.random(), ti.random()]
        velocities[i, j] = [ti.randn()/width, ti.randn()/height]
        # velocities[i, j] = [0.001, 0.001]
    forces[0, 0] = 0.00000005
    forces[1, 1] = 0.00000005
    forces[2, 2] = 0.00000005
    forces[2, 1] = -0.00000005

@ti.kernel
def update_vel():
    ti.loop_config(block_dim=512)
    for i, j in ti.ndrange(types, amount):
        force_acc = ti.Vector([0.0, 0.0])
        for iOther, jOther in ti.ndrange(types, amount):
            if i != iOther or j != jOther:
                # force = forces[i, iOther]
                dir = positions[iOther, jOther] - positions[i, j]
                dist = dir.norm_sqr() + 1e-5
                force = forces[i, iOther] / dist
                force_acc += dir * (force / ti.sqrt(dist))
        # velocities[i, j] = (velocities[i, j] + dir*force) 
        velocities[i, j] += force_acc
                    

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

# def render():
#     gui.clear(0x000000)
#     np_pos = positions.to_numpy()
    
#     for i in range(types):
#         for j in range(amount):
#             gui.circle(pos=np_pos[i, j], radius=rad, color=colors[i])
#             gui.circles
#     gui.show()
    
# def render2():
#     canvas.set_background_color((0.0, 0.0, 0.0))
#     np_pos = positions.to_numpy().reshape(-1, 2)
#     for i in range (types):
#         canvas.circles(np_pos[i * amount: (i + 1) * amount]).radius(rad).color(colors[i])

def render3():
    gui.clear(0x000000)
    np_pos = positions.to_numpy().reshape(-1, 2)
    
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
    move()
    # render()
    render3()
    # gui.show()
    # window.show()