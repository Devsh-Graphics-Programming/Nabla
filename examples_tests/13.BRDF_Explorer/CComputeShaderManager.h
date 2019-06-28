#ifndef C_COMPUTE_SHADER_MANAGER_H_INCLUDED
#define C_COMPUTE_SHADER_MANAGER_H_INCLUDED

#include <string>
#include <irr/core/Types.h>

class CComputeShaderManager
{
    CComputeShaderManager() = delete;

    static irr::core::unordered_map<std::string, uint32_t> Shaders;
public:
    static uint32_t getShader(const std::string& _path_to_source);
    static void clear();
};

#endif