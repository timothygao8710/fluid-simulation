# Demo Video

[![Fluid Simulation Demo](https://img.youtube.com/vi/IT1tAGbmo7s/0.jpg)](http://www.youtube.com/watch?v=IT1tAGbmo7s "Demo")

In this project, we develop a digital program to simulate and animate Newtonian fluid flow utilizing computational methods outlined in Jos Stam’s 2003 paper “Real-Time Fluid Simulation for Games”. Users can customize settings or add custom components to the simulation engine to achieve desired effects like gravity, toroidal boundaries, and wind tunnel. The simulation can be visualized as a playable .mp4 file. 

To characterize the fluid, our simulation engine utilizes three NxN grids: dens, U, and V. These store the density of fluids in each grid cell, a horizontal velocity component, and a vertical velocity component, respectively. The fluid dynamics solver calculates how these three arrays evolve over discrete time frames as the fluid “flows,” while obeying the Naiver-Stokes equations which describe the real-life physics of Newtonian Fluids.

At each time step, the density grid values are mapped to a linear colormap to generate frames that are then collated together to create the animation. The U and V grids define the vector field that the fluid flows through. Four functions are core to the solver, implementable in 50 lines using just Python’s Numpy and Math libraries. They are: add force, diffusion, advection, and projection. Add force accounts for the external forces term in Naiver-Stokes. Diffusion accounts for the tendency of fluid particles to spread out. Advection accounts for the movement of particles and their momentum along the vector field. Projection is used to replace the vector field with its divergence-free component, to ensure compliance with the incompressibility equation of Naiver-Stokes.

[Slides](https://docs.google.com/presentation/d/1TGygrkTte1WPdamOsbPzJm_g6-6Z6zAYX9Hz2j-tGlI) | [More Demos](https://github.com/timothygao8710/fluid-simulation/tree/main/Demos)

# Learn to Build This:

Video

# Dependencies:
- numpy==1.22.4 &rarr; Numpy array and basic operations
- matplotlib==3.7.1 &rarr; Generating vector field plot and writing animation from frames
- tqdm==4.36.0 &rarr; Displaying a progress bar
- numba==0.53.1 &rarr; Just-in-time compiler for Python that speeds up [execution time](https://www.youtube.com/watch?v=x58W9A2lnQc)
- ipython==8.13.2 &rarr; "clear_output from IPython.display" used to clear output from notebook cell

# How to use:
1. Make sure the Dependencies are installed
2. Open run_me.py
3. Customize 15+ Settings (e.g., diffusion coefficient, viscosity, N_time_steps, color map, etc.)
4. [Run](https://pythonbasics.org/execute-python-scripts/) the Python script
5. After the code finishes executing, check the respository folder (or your custom save location in settings) for the output .mp4 file

# References:
- http://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/papers/StamFluidforGames.pdf
- https://www.youtube.com/watch?v=qsYE1wMEMPA
- https://www.youtube.com/watch?v=alhpH6ECFvQ
- https://www.youtube.com/watch?v=x58W9A2lnQc
- https://mikeash.com/pyblog/fluid-simulation-for-dummies.html
- https://mofu-dev.com/en/blog/stable-fluids/#vector-calculus
- https://www.youtube.com/watch?v=iKAVRgIrUOU
- https://www.youtube.com/watch?v=sSJmUmCHAJY
