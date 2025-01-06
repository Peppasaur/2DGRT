cmake_minimum_required(VERSION 3.18)
set(TCNN_CUDA_ARCHITECTURES 86)

# 项目信息
project(GaussianTracer
    VERSION 1.0
    DESCRIPTION "3D Gaussian Ray Tracer"
    LANGUAGES C CXX CUDA
)

# 配置构建类型
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type specified. Setting to 'Release'.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# CUDA 编译选项
list(APPEND CUDA_NVCC_FLAGS
    "-Xcompiler=-Wno-float-conversion"
    "-Xcompiler=-fno-strict-aliasing"
    "-Xcompiler=-fPIC"
    "--expt-extended-lambda"
    "--expt-relaxed-constexpr"
    "--use_fast_math"
)

# 检查 CUDA 架构
if (NOT DEFINED TCNN_CUDA_ARCHITECTURES)
    message(FATAL_ERROR "TCNN_CUDA_ARCHITECTURES is not defined.")
endif()
set(CMAKE_CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})

find_package(pybind11 REQUIRED PATHS /home/qinhaoran/.local/lib/python3.6/site-packages/pybind11/share/cmake/pybind11)
find_package(Torch REQUIRED PATHS /home/qinhaoran/.local/lib/python3.6/site-packages)

# 包含目录
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/optix
    ${CMAKE_SOURCE_DIR}/include/glm
    ${CMAKE_BINARY_DIR} # For PTX headers
	${pybind11_INCLUDE_DIRS}
	${TORCH_INCLUDE_DIRS}
)


# Python 扩展模块
add_library(gtracer_C SHARED
    src/bvh.cu
    src/bindings.cu
)

set_target_properties(gtracer_C PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    OUTPUT_NAME "_C"
    PREFIX ""
    SUFFIX ".so"
)

target_compile_options(gtracer_C PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>
)

# OptiX PTX 编译
add_library(optix_program OBJECT
    src/optix/gaussiantrace_forward.cu
    src/optix/gaussiantrace_backward.cu
)

set_target_properties(optix_program PROPERTIES
    CUDA_PTX_COMPILATION ON
    CUDA_ARCHITECTURES OFF
)

# 使用 bin2c 将 PTX 转换为头文件
find_program(bin_to_c NAMES bin2c PATHS ${CMAKE_CUDA_COMPILER} PATH_SUFFIXES ../../bin)
if (NOT bin_to_c)
    message(FATAL_ERROR "bin2c not found.")
endif()

set(OPTIX_PTX_HEADER ${CMAKE_BINARY_DIR}/optix_ptx.h)
add_custom_command(
    OUTPUT ${OPTIX_PTX_HEADER}
    COMMAND ${CMAKE_COMMAND}
    -DBIN_TO_C_COMMAND=${bin_to_c}
    -DOBJECTS=$<TARGET_OBJECTS:optix_program>
    -DOUTPUT=${OPTIX_PTX_HEADER}
    -P ${CMAKE_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
    VERBATIM
    DEPENDS optix_program src/optix/gaussiantrace_forward.cu src/optix/gaussiantrace_backward.cu
    COMMENT "Converting PTX files to a C header"
)
add_dependencies(gtracer_C optix_program)


target_include_directories(optix_program PUBLIC ${CMAKE_BINARY_DIR})

# 将 PTX 头文件添加到 Python 扩展
target_sources(gtracer_C PRIVATE ${OPTIX_PTX_HEADER})

# 链接库
target_link_libraries(gtracer_C PUBLIC pybind11::module ${CMAKE_DL_LIBS} ${TORCH_LIBRARIES})

# 输出路径
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Python 包的安装规则
set(CMAKE_INSTALL_PREFIX /home/qinhaoran/.local/lib/python3.6/site-packages)
install(TARGETS gtracer_C
    LIBRARY DESTINATION gtracer
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/gtracer
    DESTINATION ${CMAKE_INSTALL_PREFIX}
    FILES_MATCHING PATTERN "*.py"
)