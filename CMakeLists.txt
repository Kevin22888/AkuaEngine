cmake_minimum_required(VERSION 3.20)

project(AquaForge LANGUAGES C CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Set output directory to project root
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# Set include directory and link directory
include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_SOURCE_DIR}/lib)

# Add source files
set(SOURCES
    src/main.cpp
    src/glad.c

    src/Application/Application.cpp
    src/Application/InputSystem.cpp
    
    src/Interop/InteropManager.cpp
    src/Interop/InteropResource.cpp
    
    src/Camera/Camera.cpp
    src/Graphics/Material.cpp
    src/Shader/ShaderProgram.cpp
    src/Rendering/Renderer.cpp
    src/Scene/Scene.cpp
    src/Scene/SceneObject.cpp

    src/Simulation/PBFSolver.cpp

    src/CUDA/ConstraintSolverCUDA.cu
    src/CUDA/NeighbourSearchCUDA.cu
    src/CUDA/IntegrationCUDA.cu
)

add_executable(AquaForge ${SOURCES})

# Copy glfw3.dll to the output directory after build
add_custom_command(TARGET AquaForge POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/glfw3.dll
        $<TARGET_FILE_DIR:AquaForge>
)

# Copy shaders folder to output directory
add_custom_command(TARGET AquaForge POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/assets/shaders
        $<TARGET_FILE_DIR:AquaForge>/assets/shaders
)

# Link GLFW and OpenGL (currently only supporting Windows)
target_link_libraries(AquaForge
    glfw3dll
    opengl32
    gdi32
)

# Add nvcc flags
target_compile_options(AquaForge
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -diag-suppress=20012>
)

message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
