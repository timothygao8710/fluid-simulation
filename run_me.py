import numpy as np
import math
from numba import njit
from IPython.display import clear_output

######### SETTINGS START #########
# N x N grid
N = 50

# Time between frames (in seconds)
dt = 0.01
# Number of discrete time steps
N_time_steps = 300

# Number of iterations for the Gauss-Seidel Solver in Diffuse and Advect stages
# Larger values results in marginally more accurate results 
lin_solve_iter = 20

##### FLUID SETTINGS ######

diff = 1e8
visc = 0.3

# Controls the % of liquid removed each iterations
# Larger values => more liquid removed
# Domain is [0, 1], 0 = no liquid removed
constraint_constant = 0
# Warning: Values too high are not recommended because it will leave too little fluid left.

# List all fluid sources you would like to include
# Each source is a dictionary, please use the same syntax and same keys () as the example.
# Each dictionary and the next should be separated by a comma like in the example

# For each source, the grid cell (y_idx, x_idx) will emit fluid with specified x velocity, y velocity, and emission rate
sources = [
    
    {
        'emission density': 1,
        
        'y location (y_idx)': N-1,
        'x location (x_idx)': N//2,
        
        'y velocity': -1e3,
        'x velocity': 0
    },
    
    {
        'emission density': 1,
        
        'y location (y_idx)': 2,
        'x location (x_idx)': N//2,
        
        'y velocity': 1e3,
        'x velocity': 0
    }
    
]

# Warning: Too High Velocity and too many sources super close to one another can cause unexpected / divergent behavior

# Note 1: For larger N, adding more sources is recommended so there is enough fluid inflow
# Note 2: x in interval [1, N], y in interval [1, N]
# Note 3: that (0,0) is the upper left cell. So to convert traditional (x,y) coordinates to grid indicies: (x, N+1-y) = (x_idx, y_idx)
# Note 4: x and y velocities are coefficients proportional to the actual units/seconds velocity
# Note 5: Emission density are relative

# save_location = "Demos/"
save_location = ""
name = "small test script"

# NOTE: If you are experiencing a weird color flickering effect (dimmer/brighter each frame), make this value larger (try x2, x5, etc.)
color_correction_coef = 0.0005 * 4

#Color Map used for display
cmap = "viridis"
#Select from following list: 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'

# Set to false if you don't want to include the vector field plot in the video
showVectorField = False

#multiplier for how fast animation should be
animation_speed = 0.5

# vector sizes in final display are scaled by this constant, larger values -> smaller vector magnitudes
vscale = 4
# color of vectors in final display
vcolor = 'b'

# Dots per inches / image "resolution"
# Larger value => less blurry animation
dpi = 300

# NOTE: If you cannot play the .mp4 file, you may have to change this section
# Default settings work for QuickTime Player on Mac (https://stackoverflow.com/questions/14430593/encoding-a-readable-movie-by-quicktime-using-ffmpeg/14437323#14437323)
extra_args = ['-pix_fmt', 'yuv420p']

print("Final video will be " + str(dt * N_time_steps / animation_speed) + " seconds long with " + str(1/dt*animation_speed) + " fps.")

if lin_solve_iter > 100 or N > 500 or dpi > 500 or N_time_steps > 1000:
    print("WARNING: These settings may result in long computation times.")
######### SETTINGS END #########

######### SIMULATION START #########
def addSource(i, j, X, x):
    X[i][j] += dt * x

def lerp(a, b, x):
    return a + (b - a) * x

def set_bnd(b, X):
    X[0, 1:N+1] = np.where(b == "ud", -X[1, 1:N+1], X[1, 1:N+1])
    X[N+1, 1:N+1] = np.where(b == "ud", -X[N, 1:N+1], X[N, 1:N+1])
    X[1:N+1, 0] = np.where(b == "lr", -X[1:N+1, 1], X[1:N+1, 1])
    X[1:N+1, N+1] = np.where(b == "lr", -X[1:N+1, N], X[1:N+1, N])
    X[0, 0] = (X[N, N+1] + X[N+1, N]) * 0.5
    X[N+1, N+1] = (X[1, 0] + X[0, 1]) * 0.5
    X[0, N+1] = (X[N, 0] + X[N+1, 1]) * 0.5
    X[N+1, 0] = (X[1, N+1] + X[0, N]) * 0.5

@njit(fastmath=True)
def diffuse(X, X0, coef, b="full", n_iter=lin_solve_iter):
    a = dt * coef * N * N
    for k in range(n_iter):
        X[1:N+1, 1:N+1] = (X0[1:N+1, 1:N+1] + a * (X[:-2, 1:N+1] + X[1:N+1, :-2] + X[1:N+1, 2:N+2] + X[2:N+2, 1:N+1])) / (1 + 4 * a)
        set_bnd(b, X)

def advect(X, X0, u, v, b="full"):
    for i in range(1, N+1):
        for j in range(1, N+1):
            y, x = i - dt * N * v[i][j], j - dt * N * u[i][j]
            x, y = max(min(x, N + 0.5), 0.5), max(min(y, N + 0.5), 0.5)
            x0, x1 = lerp(X0[math.floor(y)][math.floor(x)], X0[math.floor(y)][math.floor(x) + 1], x - math.floor(x)), lerp(X0[math.floor(y) + 1][math.floor(x)], X0[math.floor(y) + 1][math.floor(x) + 1], x - math.floor(x))
            X[i][j] += lerp(x0, x1, y - math.floor(y))

    set_bnd(b, X)
    
def project(u, v, p, div, n_iter=lin_solve_iter):
    div[1:N+1, 1:N+1] = -0.5 * (u[1:N+1, 2:N+2] - u[1:N+1, :N] + v[2:N+2, 1:N+1] - v[:N, 1:N+1]) / N
    p[:N+2, :N+2] = 0
    
    set_bnd("full", div)
    set_bnd("full", p)

    for k in range(n_iter):
        p[1:N+1, 1:N+1] = (div[1:N+1, 1:N+1] + p[:N, 1:N+1] + p[2:N+2, 1:N+1] + p[1:N+1, :N] + p[1:N+1, 2:N+2]) / 4
        set_bnd("full", p)

    u[1:N+1, 1:N+1] -= 0.5 * (p[1:N+1, 2:N+2] - p[1:N+1, :N]) * N
    v[1:N+1, 1:N+1] -= 0.5 * (p[2:N+2, 1:N+1] - p[:N, 1:N+1]) * N

    set_bnd("lr", u)
    set_bnd("ud", v)
######### SIMULATION END #########

######## RUNNING SIMULATION ##########
from tqdm import tqdm

# 0 and N+1 reserved for boundaries
sz = N+2

# Initialize variables
animation_frames, u_frames, v_frames = [], [], []
dens0, v0, u0 = np.zeros((sz, sz)), np.zeros((sz, sz)), np.zeros((sz, sz))

print("Generating Frames...")

# tqdm wrapper is what makes the progress bar
for i in tqdm(range(N_time_steps), position=0, leave=True):
    dens, u, v = np.zeros((sz, sz)), np.zeros((sz, sz)), np.zeros((sz, sz))
    
    # Add sources
    for source in sources:
        e, y_idx, x_idx, y_v, x_v = list(source.values())
        addSource (y_idx, x_idx, v0, dt * N * y_v)
        addSource (y_idx, x_idx, u0, dt * N * x_v)
        addSource(y_idx, x_idx, dens0, e)
        
    # Velocity Step
    diffuse (u, u0, visc, b="lr")
    diffuse (v, v0, visc, b="ud")
    project(u0, v0, u, v)
    advect (u, u0, u0, v0, b="lr")
    advect (v, v0, u0, v0, b="ud")
    project(u, v, u0, v0)
    
    # Density Step
    diffuse (dens, dens0, diff)
    advect (dens, dens0, u, v)
    
    # Remove some liquid
    dens *= 1 - constraint_constant
    
    # Append to frames
    animation_frames.append(dens.copy())
    u_frames.append(u.copy())
    v_frames.append(v.copy())
    
    # Next time step
    dens0 = dens
    v0 = v
    u0 = u
######## END RUNNING SIMULATION ##########

######## CREATING ANIMATION ##########
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
matplotlib.use("Agg")

# Frame by frame, there are varying # of cells near the source with outlier densities that disportionately skew the data range
# This causing large differences in the normalization process for the colormap, causing a weird flickering color effect between frames
def correctColor(frame, N_sources = len(sources), coef = color_correction_coef):
    temp = np.reshape(frame, (sz * sz))
    temp = np.sort(temp)
    
    mx_idx = sz * sz - 1 - N_sources * int(coef * sz * sz)

    for i in range(sz):
        for j in range(sz):                
            frame[i][j] = min(frame[i][j], temp[mx_idx])
    
    return frame

# Initialize animation writer
Writer = animation.writers['ffmpeg']
writer = Writer(extra_args=extra_args, fps=1/dt*animation_speed)

#Make plot
if showVectorField:
    fig, (ax1, ax2) = plt.subplots(1, 2)
else:
    fig, ax1 = plt.subplots(1)
    
# Change plot settings
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
fig.tight_layout()

if showVectorField:
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.invert_yaxis()

# Plot the first frame 
fluid = ax1.imshow(correctColor(animation_frames[0]), animated=True, cmap=cmap, aspect = ("auto" if showVectorField else None))

if showVectorField:
    vfield = ax2.quiver(u_frames[0], v_frames[0], animated=True, color=vcolor, pivot='mid', units='xy', scale=vscale)

# Displaying the next index in frames
def updatefig(idx, fluid, vfield):
    fluid.autoscale()
    fluid.set_data(correctColor(animation_frames[idx]))
    if showVectorField:
        vfield.set_UVC(u_frames[idx], v_frames[idx])
                
    return fluid, vfield

print("Generating Animation...")
ani = FuncAnimation(fig, updatefig, frames=tqdm(range(len(animation_frames)), position=0, leave=True), fargs=(fluid, (vfield if showVectorField else None)), blit=False)
ani.save(save_location + name + ".mp4", dpi=dpi, writer=writer)