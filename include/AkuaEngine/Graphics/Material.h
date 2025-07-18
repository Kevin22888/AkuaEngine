#ifndef AKUAENGINE_MATERIAL_H
#define AKUAENGINE_MATERIAL_H

#include <AkuaEngine/Shader/ShaderProgram.h>

namespace AkuaEngine {

class Material {
public:
    Material(const char* vertexShaderFilePath, const char* fragmentShaderFilePath);
    ~Material();

    Material(const Material& other) = delete;
    Material& operator=(const Material& other) = delete;
    Material(Material&& other) noexcept;
    Material& operator=(Material&& other) noexcept;

    ShaderProgram* getShaderProgram() const;
    
private:
    ShaderProgram* _shaderProgram;
};

} // namespace AkuaEngine

#endif // AKUAENGINE_MATERIAL_H