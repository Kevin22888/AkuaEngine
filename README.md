# Position-Based Fluid Simulation

This is a C++ fluid simulation based on the technique of Position-Based Fluids [(Macklin and Müller 2013)](https://mmacklin.com/pbf_sig_preprint.pdf), implemented with CUDA and OpenGL.

**NOTE**: This is an ongoing project. It is a working prototype but not all implementations are complete.

The program can currently support a basic fluid behaviour but still needs to be fine-tuned. A few key implementations are:
- SPH density estimator
- Neighbour Search with Spatial Hashing
- Density constraint solver
- Correction for tensile instability (requires tweaking)
And lastly it uses CUDA-OpenGL interoperability to make rendering more efficient.

Again, the fluid can break with even just slight changes in the simulation parameters, so there must be parts done incorrectly. The structure of the codebase is also not refactored cleanly. In addition, viscosity and vorticity in the fluid are not implemented yet.

## Dependencies

`GLFW`, `GLM`, `CUDA Toolkit`. 

- Note that `GLAD` and `GLM` are in the `include/` folder so you may skip setting these up.
- `GLFW`'s linking library is also contained in this project, but it is for Windows 64-bit Visual C++ 2022, so you may want to set up your own.

Besides that, if you have CUDA Toolkit installed and you are on Windows, you may not require any additional setup.

## Build

CMakeLists.txt is not yet added. You may compile the project with this command:
```
nvcc -diag-suppress 20012 --extended-lambda -o PBF.exe src/main.cpp src/Camera.cpp src/Shader.cpp src/glad.c src/SPH_Estimator.cu -Iinclude -Llib -lglfw3dll -lgdi32 -lopengl32
```
Additionally you can look up your NVIDIA GPU's compute capability version number (something like `X.Y`) and add the flag `-arch=sm_XY` (e.g. `-arch=sm_89`).
