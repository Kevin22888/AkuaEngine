#include "ShaderProgram.h"

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <fstream>
#include <sstream>
#include <iostream>

namespace AquaForge {

ShaderProgram::ShaderProgram(const char* vertexShaderFilePath, const char* fragmentShaderFilePath) {
    // Define strings to hold shader code and define file input streams
    std::string vShaderString, fShaderString;
    std::ifstream vShaderFile, fShaderFile;
    std::stringstream vShaderStream, fShaderStream;

    vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    try {
        // Open files
        vShaderFile.open(vertexShaderFilePath);
        fShaderFile.open(fragmentShaderFilePath);
        // read file's buffer contents into string streams
        vShaderStream << vShaderFile.rdbuf();
        fShaderStream << fShaderFile.rdbuf();
        // close file streams (handlers)
        vShaderFile.close();
        fShaderFile.close();
    } catch (std::ifstream::failure& e) {
        std::cerr << "[AquaForge::ShaderProgram::ShaderProgram] Failed to load file: " << e.what() << std::endl;
    }

    // convert string streams into string
    vShaderString = vShaderStream.str();
    fShaderString = fShaderStream.str();

    // Set up OpenGL shader program
    // Create shader objects, attach shader source code, and compile, check for errors
    const char* vShaderSource = vShaderString.c_str();
    const char* fShaderSource = fShaderString.c_str();

    unsigned int vertexShader, fragmentShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vShaderSource, NULL);
    glCompileShader(vertexShader);
    checkCompileErrors(vertexShader, TargetType::VertexShader);

    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fShaderSource, NULL);
    glCompileShader(fragmentShader);
    checkCompileErrors(fragmentShader, TargetType::FragmentShader);

    _programID = glCreateProgram();
    glAttachShader(_programID, vertexShader);
    glAttachShader(_programID, fragmentShader);
    glLinkProgram(_programID);
    checkCompileErrors(_programID, TargetType::Program);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void ShaderProgram::bind() {
    if (!_isValid || !_isAlive) return;
    
    glUseProgram(_programID);
}

void ShaderProgram::destroy() {
    if (!_isAlive) return;

    glDeleteProgram(_programID);
    _programID = 0;
    _isAlive = false;
}

// Later on, let the application handle shader errors and support hot reload
void ShaderProgram::checkCompileErrors(GLuint targetID, TargetType type) {
    GLint success;
    char infoLog[1024];
    switch (type) {
        case TargetType::VertexShader:
        case TargetType::FragmentShader:
            glGetShaderiv(targetID, GL_COMPILE_STATUS, &success);
            if (!success) {
                glGetShaderInfoLog(targetID, 1024, NULL, infoLog);
                std::cerr << "[AquaForge::ShaderProgram::checkCompileErrors] Shader Compile Error:\n" << infoLog << std::endl;
            }
            break;

        case TargetType::Program:
            glGetProgramiv(targetID, GL_LINK_STATUS, &success);
            if (!success) {
                glGetProgramInfoLog(targetID, 1024, NULL, infoLog);
                std::cerr << "[AquaForge::ShaderProgram::checkCompileErrors] Program Linking Error:\n" << infoLog << std::endl;
            }
            break;
    }
}

} // namespace AquaForge