#ifndef AQUAFORGE_INTEROP_RESOURCE_H
#define AQUAFORGE_INTEROP_RESOURCE_H

#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <utility>

namespace AquaForge {

class InteropResource {
public:
    explicit InteropResource(GLuint bufferID);
    ~InteropResource();

    // Disable copy, allow move
    InteropResource(const InteropResource&) = delete;
    InteropResource& operator=(const InteropResource&) = delete;
    InteropResource(InteropResource&& other) noexcept;
    InteropResource& operator=(InteropResource&& other) noexcept;

    cudaGraphicsResource* getGraphicsResource() const;

    // Allow the manager of this resource to call it before GL context terminates
    void release();
private:
    cudaGraphicsResource* _resource = nullptr;    
};

// ============================= Inline functions ==============================

inline InteropResource::InteropResource(GLuint bufferID) {
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&_resource, bufferID, cudaGraphicsMapFlagsNone);
    if (err != cudaSuccess) {
        std::cerr << "[AquaForge::InteropResource::InteropResource] Failed to register OpenGL buffer for access by CUDA." << std::endl;
    }
}

inline InteropResource::~InteropResource() { 
    release(); 
}

inline InteropResource::InteropResource(InteropResource&& other) noexcept 
    : _resource(std::exchange(other._resource, nullptr)) {}

inline InteropResource& InteropResource::operator=(InteropResource&& other) noexcept {
    if (this != &other) {
        release();
        _resource = std::exchange(other._resource, nullptr);
    }
    return *this;
}

inline cudaGraphicsResource* InteropResource::getGraphicsResource() const { 
    return _resource; 
}

inline void InteropResource::release() {
    if (_resource) {
        cudaGraphicsUnregisterResource(_resource);
        _resource = nullptr;
    }
}

} // namespace AquaForge

#endif // AQUAFORGE_INTEROP_RESOURCE_H