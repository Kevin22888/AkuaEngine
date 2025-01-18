#ifndef PBF_SHADER_H
#define PBF_SHADER_H

#include <string>
#include <glm/glm.hpp>

class Shader {
public:
    unsigned int programID;

    Shader(const char* vertexShaderFilePath, const char* fragmentShaderFilePath);
    void useProgram();
    void setUniformMat4(const std::string &name, const glm::mat4 &mat) const;
    void setUniformVec3(const std::string &name, const glm::vec3 &vec) const;
    void setUniformFloat(const std::string &name, float value) const;
    
private:
    void checkCompileErrors(unsigned int shader, const std::string& type);
};

#endif // PBF_SHADER_H