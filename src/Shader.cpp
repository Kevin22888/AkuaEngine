#include <Shader.h>

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <fstream>
#include <sstream>
#include <iostream>

Shader::Shader(const char* vertexShaderFilePath, const char* fragmentShaderFilePath) {
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
        std::cout << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: " << e.what() << std::endl;
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
    checkCompileErrors(vertexShader, "VERTEX");

    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fShaderSource, NULL);
    glCompileShader(fragmentShader);
    checkCompileErrors(fragmentShader, "FRAGMENT");

    programID = glCreateProgram();
    glAttachShader(programID, vertexShader);
    glAttachShader(programID, fragmentShader);
    glLinkProgram(programID);
    checkCompileErrors(programID, "PROGRAM");

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void Shader::useProgram() {
    glUseProgram(programID);
}

// Set a uniform of a given variable name (name) with a given data (mat)
// Remember uniforms are global so they aren't associated with any shader, they associate with the whole program
// third argument: GL_FALSE, tells OpenGL we dont need the transpose, has to do with column-major ordering
// last argument equivalent: glm::value_ptr(mat)
void Shader::setUniformMat4(const std::string &name, const glm::mat4 &mat) const {
    glUniformMatrix4fv(glGetUniformLocation(programID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}

void Shader::setUniformVec3(const std::string &name, const glm::vec3 &vec) const {
    glUniform3fv(glGetUniformLocation(programID, name.c_str()), 1, &vec[0]);
}

void Shader::setUniformFloat(const std::string &name, float value) const {
    glUniform1f(glGetUniformLocation(programID, name.c_str()), value);
}

/*
Should add some more proper clean up if an error does occur
*/
void Shader::checkCompileErrors(unsigned int target, const std::string& type) {
    int success;
    char infoLog[1024];
    if (type != "PROGRAM") {
        glGetShaderiv(target, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(target, 1024, NULL, infoLog);
            std::cout << "ERROR::SHADER_COMPILATION_ERROR: " << type << "\n" << infoLog << "\n---" << std::endl;
        }
    } else {
        glGetProgramiv(target, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(target, 1024, NULL, infoLog);
            std::cout << "ERROR::PROGRAM_LINKING_ERROR: " << "\n" << infoLog << "\n---" << std::endl;
        }
    }
}