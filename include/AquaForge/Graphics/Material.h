#ifndef AQUAFORGE_MATERIAL_H
#define AQUAFORGE_MATERIAL_H

#include <AquaForge/Shader/ShaderProgram.h>

namespace AquaForge {

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

} // namespace AquaForge

#endif // AQUAFORGE_MATERIAL_H