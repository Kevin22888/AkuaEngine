#ifndef AQUAFORGE_PBF_CONFIG_H
#define AQUAFORGE_PBF_CONFIG_H

#include <glm/glm.hpp>

namespace AquaForge {

// Parameters used to compute a correction term to lambda.
// The correction acts as an artifical pressure to mimic surface tension.
struct LambdaCorrParams {
    bool enabled = true;
    float k = 0.0001f;
    float n = 4.0f;
    float delta_q = 0.03;
};

// Consider moving these structs into PBFTypes.h and making a configuration file with global variables
struct PBFConfig {
    float restDensity = 7600.0f; // kg/m^3
    float particle_spacing = 0.05f;
    float smoothRadius = 0.1f;
    float spatialHashCellSize = 0.1f; // take cellsize equal to smooth radius
    float relaxation = 600.0f;
    float vorticityEpsilon = 0.00001f;
    float viscosity = 0.01f;
    int maxNeighbours = 128;
    int solverIterations = 4;
    glm::vec3 gravity = {0.0f, -9.8f, 0.0f};
};

} // namespace AquaForge

#endif // AQUAFORGE_PBF_CONFIG_H