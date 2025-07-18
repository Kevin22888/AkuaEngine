#ifndef AKUAENGINE_INTEROP_RESOURCE_H
#define AKUAENGINE_INTEROP_RESOURCE_H

#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <utility>

namespace AkuaEngine {

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

inline cudaGraphicsResource* InteropResource::getGraphicsResource() const { 
    return _resource; 
}

} // namespace AkuaEngine

#endif // AKUAENGINE_INTEROP_RESOURCE_H