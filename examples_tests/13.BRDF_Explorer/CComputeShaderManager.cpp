#include "CComputeShaderManager.h"

#include "../source/Irrlicht/COpenGLDriver.h"
#include "createComputeShader.h"

irr::core::unordered_map<std::string, uint32_t> CComputeShaderManager::Shaders;

uint32_t CComputeShaderManager::getShader(const std::string& _path_to_source)
{
    auto found = Shaders.find(_path_to_source);
    if (found != Shaders.end())
        return found->second;

    return Shaders[_path_to_source] = createComputeShaderFromFile(_path_to_source.c_str());
}

void CComputeShaderManager::clear()
{
    for (auto& s : Shaders)
        irr::video::COpenGLExtensionHandler::extGlDeleteProgram(s.second);
    Shaders.clear();
}
