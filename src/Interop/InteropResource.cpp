#include <AquaForge/Interop/InteropResource.h>
#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <utility>

namespace AquaForge {

InteropResource::InteropResource(GLuint bufferID) {
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&_resource, bufferID, cudaGraphicsMapFlagsNone);
    if (err != cudaSuccess) {
        std::cerr << "[AquaForge::InteropResource::InteropResource] Failed to register OpenGL buffer for access by CUDA." << std::endl;
    }
}

InteropResource::~InteropResource() { 
    release(); 
}

InteropResource::InteropResource(InteropResource&& other) noexcept : _resource(std::exchange(other._resource, nullptr)) {}

InteropResource& InteropResource::operator=(InteropResource&& other) noexcept {
    if (this != &other) {
        release();
        _resource = std::exchange(other._resource, nullptr);
    }
    return *this;
}

void InteropResource::release() {
    if (_resource) {
        cudaGraphicsUnregisterResource(_resource);
        _resource = nullptr;
    }
}

} // namespace AquaForge
