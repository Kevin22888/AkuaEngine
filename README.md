# AquaForge

## Overview

AquaForge is a real-time fluid simulation engine based on the Position-Based Fluids (PBF) algorithm by [Macklin and Müller (2013)](https://mmacklin.com/pbf_sig_preprint.pdf). It implements the core PBF solver from scratch, without relying on existing physics engines or software libraries. The solver runs entirely on the GPU, using custom CUDA kernels for neighbor search, constraint solving, and integration. Thrust is used selectively for sorting. Rendering is done via OpenGL, using a lightweight scene system and CUDA-OpenGL interop.

This project is part of my self-driven journey into graphics programming——an exploration of physically based animation through building something from the ground up.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

## Table of Contents

- [Overview](#overview)
- [Simulation Preview](#simulation-preview)
- [Building the Project](#building-the-project)
- [Background](#background)
- [Implementation Highlights](#implementation-highlights)
- [License](#license)
- [Contact](#contact)
- [References](#references)

## Simulation Preview

Below are some real-time captures of a dam break simulation with 27000 particles, running at about 90 FPS with a timestep of 0.0083 seconds (the GIFs are 30 FPS to reduce size). This also shows a comparison of the visual result with and without artificial pressure as described in the paper.

<table>
  <tr>
    <td align="center">
      <img src="assets/gifs/DamBreak.gif" width="400"/><br>
      <em>Dam break with artificial pressure.</em>
    </td>
    <td align="center">
      <img src="assets/gifs/DamBreak_NoPressure.gif" width="400"/><br>
      <em>Dam break. No artificial pressure.</em>
    </td>
  </tr>
</table>

## Building the Project

### Dependencies

- Windows (tested on Windows 11)
- CUDA Toolkit
- OpenGL 3.3+
- CMake
- MSVC with `cl.exe` (due to CUDA compatibility)

The project also depends on GLFW and GLM, but they are included in the project files, so there is no need to set them up.

### Build Instructions

Clone the repository:
```bash
git clone https://github.com/Kevin22888/AquaForge.git
```
Build with CMake:
```bash
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```
Note that the provided configuration in `CMakeLists.txt` will generate the executable and the required files in `./Release`.

### Usage

**Controls**:
- WASD for movement
- Mouse for look
- Mouse Scroll for changing FOV
- Press `Space` to begin/resume simulation
- Press `B` to pause simulation
- Press `ESC` to quit the application

## Background

The numerical simulation of physical systems traditionally focused on force-based methods, which are rooted in Newton's second law——forces give rise to accelerations, which are numerically integrated to obtain velocities and then positions. This paradigm dominates computational physics and is popular in early graphics techniques. However, it comes with problems like energy gain during explicit integration, instability under large time steps, and the indirect control over positions. These challenges are also found in computational physics research problems.

The goal of physically based animation in computer graphics is often not about strict physical accuracy, but visually plausible results with stability and efficiency for real-time applications. This led to the development of alternative approaches, most notably the Position-Based Dynamics (PBD) framework [(Müller et al. 2007)](https://matthias-research.github.io/pages/publications/posBasedDyn.pdf). PBD shifts the focus from forces to directly manipulating positions to satisfy constraints. Constraints in PBD are designed to capture the essential physical behavior of the system, such as maintaining particle spacing, preserving volumes, or enforcing incompressibility. Velocities are still integrated, but are updated based on the corrected positions. The resulting behavior of the system is more stable, especially great for real-time simulations or video games.

For fluid simulation, an early method is Smoothed Particle Hydrodynamics (SPH) [(Monaghan 1992)](https://ui.adsabs.harvard.edu/abs/1992ARA%26A..30..543M/abstract), originally invented for studying astrophysics problems [(Gingold and Monaghan 1977](https://ui.adsabs.harvard.edu/abs/1977MNRAS.181..375G/abstract), [Lucy 1977)](https://ui.adsabs.harvard.edu/abs/1977AJ.....82.1013L/abstract). SPH discretizes the fluid continuum into particles. Each particle has a mass and represents a fluid parcel. It is a particle method that approximates the Navier-Stokes equations through discretization. SPH estimates field quantities (like density and pressure) through kernel-weighted sums, then computes the forces at each particle, and finally integrates for velocity and position.

The Position-Based Fluids (PBF) algorithm [(Macklin and Müller 2013)](https://mmacklin.com/pbf_sig_preprint.pdf) combines these two philosophies. It uses the SPH density estimator to define a density constraint, enforcing incompressibility directly in position space. This constraint is then solved using the PBD framework, eliminating the need for forces or pressures and directly computing particle positions. Like SPH and PBD, PBF is in the Lagrangian formulation, tracking particles through space, which contrasts with Eulerian grid-based solvers. This position-based approach makes it highly suitable in dynamic, real-time environments.

## Implementation Highlights

Here are a few aspects of the implementation that reflect some of my design choices and trade-offs, beyond the plain implementation of the PBF algorithm.

### Neighbor Searching with the Spatial Hashing algorithm

Efficient neighbor search is one of the key performance bottlenecks in particle-based fluid simulation. To avoid the naive $O(n^2)$ approach, I implemented a spatial hashing method that offers $O(n)$ time complexity. I studied the technique from various sources, including [Ihmsen et al. (2014)](https://cg.informatik.uni-freiburg.de/publications/2014_EG_SPH_STAR.pdf), [Teschner et al. (2003)](https://matthias-research.github.io/pages/publications/tetraederCollision.pdf), [Ihmsen et al. (2011)](https://cg.informatik.uni-freiburg.de/publications/2011_CGF_dataStructuresSPH.pdf), and the approachable tutorial from [Ten Minute Physics](https://matthias-research.github.io/pages/tenMinutePhysics/index.html). My implementation discretizes space into grid cells and maintains particle-to-cell mappings using a custom GPU-side hash table. While there’s room for optimization, the system is functional and fully integrated into the CUDA simulation loop.

### CUDA implementation of the PBF algorithm

The core PBF solver is implemented entirely in CUDA, with custom kernels for neighbor search, density constraint solving, and position updates. CUDA-OpenGL interoperability enables direct buffer sharing between simulation and rendering, avoiding the redundant memory copying.

To keep CUDA-specific details encapsulated, I structured all device operations behind wrapper functions and namespaces, and defined an `InteropResource` class to isolate the CUDA-OpenGL interop setup. This allows the simulation system to be used independently from rendering logic. The `PBFSolver` contains the main logic behind each simulation step. A blog post is under preparation to share how I’ve come to understand the mathematical parts of the algorithm.

### OpenGL-based rendering and scene system

The rendering system is built on modern OpenGL, with a lightweight but scalable scene architecture. A `Scene` holds `SceneObject`s, which can be either meshes or particle systems. Each object is paired with a `Material`, which manages shader bindings and parameters. Rendering responsibilities are centralized in the `Renderer` class, which handles OpenGL state management, buffer uploads, and CUDA interop. The rest of the application remains decoupled from graphics code. This structure follows SOLID principles and lays the groundwork for extending AquaForge into a general-purpose graphics engine.

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

If you use this code or adapt any part of it in your project, I would appreciate a mention or credit. And if you’ve found this project helpful and are building something with it, feel free to reach out because I would love to hear about it.

## Contact

Contact: kevin.graphics.dev@gmail.com

I would love meet people in the field!
- If you work in computer graphics and have feedback, insights, or ideas, I'd be grateful to learn from you.
- If you're a fellow beginner or indie developer working on similar problems, feel free to reach out. I'm always happy to exchange ideas and share what I’ve learned.
- If you're part of a team working in graphics, simulation, or interactive tech, and think my work aligns with what you do, I'd love to connect.


## References

Macklin, M. & Müller, M. (2013). [Position Based Fluids](https://mmacklin.com/pbf_sig_preprint.pdf). \
Müller, M. et al. (2007). [Position Based Dynamics](https://matthias-research.github.io/pages/publications/posBasedDyn.pdf). \
Ihmsen, M., Cornelis, J., Solenthaler, B., Horvath, C., & Teschner, M. (2014). [SPH Fluids in Computer Graphics](https://cg.informatik.uni-freiburg.de/publications/2014_EG_SPH_STAR.pdf). \
Teschner, M., Heidelberger, B., Müller, M., Pomeranets, D., & Gross, M. (2003). [Optimized Spatial Hashing for Collision Detection of Deformable Objects](https://matthias-research.github.io/pages/publications/tetraederCollision.pdf). \
Ihmsen, M., Orthmann, J., & Teschner, M. (2011). [Data Structures for Particle-Based Fluid Simulation](https://cg.informatik.uni-freiburg.de/publications/2011_CGF_dataStructuresSPH.pdf). \
Monaghan, J. J. (1992). [Smoothed Particle Hydrodynamics](https://ui.adsabs.harvard.edu/abs/1992ARA%26A..30..543M/abstract). \
Gingold, R. A., & Monaghan, J. J. (1977). [Smoothed Particle Hydrodynamics: Theory and Application to Non-spherical Stars](https://ui.adsabs.harvard.edu/abs/1977MNRAS.181..375G/abstract). \
Lucy, L. B. (1977). [A Numerical Approach to the Testing of the Fission Hypothesis](https://ui.adsabs.harvard.edu/abs/1977AJ.....82.1013L/abstract).
