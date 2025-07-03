#include <AquaForge/Graphics/Material.h>
#include <AquaForge/Shader/ShaderProgram.h>

namespace AquaForge {

Material::Material(const char* vertexShaderFilePath, const char* fragmentShaderFilePath) {
    _shaderProgram = new ShaderProgram(vertexShaderFilePath, fragmentShaderFilePath);
}

Material::~Material() {
    delete _shaderProgram;
}

Material::Material(Material&& other) noexcept : _shaderProgram(other._shaderProgram) {
    other._shaderProgram = nullptr;
}

Material& Material::operator=(Material&& other) noexcept {
    if (this != &other) {
        delete _shaderProgram;
        _shaderProgram = other._shaderProgram;
        other._shaderProgram = nullptr;
    }
    return *this;
}

ShaderProgram* Material::getShaderProgram() const {
    return _shaderProgram;
}

} // namespace AquaForge
