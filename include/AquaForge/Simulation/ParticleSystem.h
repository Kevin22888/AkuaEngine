#ifndef AQUAFORGE_PARTICLE_SYSTEM_H
#define AQUAFORGE_PARTICLE_SYSTEM_H

#include <AquaForge/Simulation/Particle.h>
#include <vector>

namespace AquaForge {

class ParticleSystem {
public:
    explicit ParticleSystem(std::vector<Particle>&& particles);

    const std::vector<Particle>& getParticles() const;
    int getParticleCount() const;

private:
    std::vector<Particle> _particles;
};

// ============================= Inline functions ==============================

inline ParticleSystem::ParticleSystem(std::vector<Particle>&& particles)
    : _particles(std::move(particles)) {}

inline const std::vector<Particle>& ParticleSystem::getParticles() const {
    return _particles;
}

inline int ParticleSystem::getParticleCount() const {
    // Technically this can overflow but we won't create that many particles
    return static_cast<int>(_particles.size());
}

} // namespace AquaForge

#endif // AQUAFORGE_PARTICLE_SYSTEM_H