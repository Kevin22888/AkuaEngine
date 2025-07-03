#ifndef AQUAFORGE_SHADER_PROGRAM_H
#define AQUAFORGE_SHADER_PROGRAM_H

#include <string>
#include <glad/glad.h>
#include <glm/glm.hpp>

namespace AquaForge {

/**
 * ShaderProgram represents a compiled and linked OpenGL shader program.
 * 
 * All methods that invoke OpenGL functions require a valid OpenGL context to be 
 * current on the calling thread. This includes:
 *   - Constructor (which loads and compiles shaders)
 *   - bind()
 *   - setUniform()
 *   - destroy()
 * 
 * It is the caller's responsibility to ensure that the GL context is active.
 * No safety checks for context validity are performed in this class.
 */
class ShaderProgram {
public:
    ShaderProgram(const char* vertexShaderFilePath, const char* fragmentShaderFilePath);
    ~ShaderProgram();

    ShaderProgram(const ShaderProgram& other) = delete;
    ShaderProgram& operator=(const ShaderProgram& other) = delete;
    ShaderProgram(ShaderProgram&& other) noexcept;
    ShaderProgram& operator=(ShaderProgram&& other) noexcept;

    void bind();
    void destroy();
    bool isValid() const;
    bool isAlive() const;

    template<typename T>
    void setUniform(const std::string& name, const T& value);
    
private:
    GLuint _programID;
    bool _isAlive;
    bool _isValid;

    enum class TargetType {
        VertexShader,
        FragmentShader,
        Program
    };

    bool checkCompileErrors(GLuint targetID, TargetType type);
};

// ============================= Inline functions ==============================

inline bool ShaderProgram::isValid() const { return _isValid; }
inline bool ShaderProgram::isAlive() const { return _isAlive; }

// ========================== Template implementation ==========================

template<typename T>
struct AlwaysFalse : std::false_type {};

template<typename T>
void ShaderProgram::setUniform(const std::string& name, const T& value) {
    GLint location = glGetUniformLocation(_programID, name.c_str());

    if constexpr (std::is_same_v<T, float>) {
        glUniform1f(location, value);
    } else if constexpr (std::is_same_v<T, glm::vec3>) {
        glUniform3fv(location, 1, &value[0]);
    } else if constexpr (std::is_same_v<T, glm::mat4>) {
        glUniformMatrix4fv(location, 1, GL_FALSE, &value[0][0]);
    } else {
        static_assert(AlwaysFalse<T>::value, "[AquaForge::ShaderProgram::setUniform] Unsupported uniform type T.");
    }
}

} // namespace AquaForge

#endif // AQUAFORGE_SHADER_PROGRAM_H