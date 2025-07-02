#ifndef AQUAFORGE_PBF_CONFIG_H
#define AQUAFORGE_PBF_CONFIG_H

#include <glm/glm.hpp>

namespace AquaForge {

// Parameters used to compute a correction term to lambda.
// The correction acts as an artifical pressure to mimic surface tension.
struct LambdaCorrParams {
    bool enabled = true;
    float k = 0.001f;
    float n = 4.0f;
    float delta_q = 0;  // This is actually a vector but I'm keeping it simple for now
};

// Consider moving these structs into PBFTypes.h and making a configuration file with global variables
struct PBFConfig {
    float restDensity = 1.0f;
    float particle_spacing = 0.5f;
    float smoothRadius = 2.0f * particle_spacing;
    float spatialHashCellSize = 2.0f * particle_spacing; // take cellsize equal to smooth radius
    float relaxation = 0.01f;
    int maxNeighbours = 512;
    int solverIterations = 4;
    glm::vec3 gravity = {0.0f, -9.8f, 0.0f};
};

} // namespace AquaForge

#endif // AQUAFORGE_PBF_CONFIG_H