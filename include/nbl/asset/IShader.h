// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_SHADER_H_INCLUDED__
#define __NBL_ASSET_I_SHADER_H_INCLUDED__

#include <algorithm>
#include <string>

#include "nbl/core/core.h"

namespace spirv_cross
{
class ParsedIR;
class Compiler;
struct SPIRType;
}

namespace nbl
{
namespace asset
{
//! Interface class for Unspecialized Shaders
/*
	The purpose for the class is for storing raw GLSL code
	to be compiled or already compiled (but unspecialized) 
	SPIR-V code. Such a shader has to be passed
	to Specialized Shader constructor.
*/

class IShader : public virtual core::IReferenceCounted
{
public:
    struct buffer_contains_glsl_t
    {
    };
    _NBL_STATIC_INLINE const buffer_contains_glsl_t buffer_contains_glsl = {};

    static inline void insertGLSLExtensionsDefines(std::string& _glsl, const core::refctd_dynamic_array<std::string>* _exts)
    {
        if(!_exts)
            return;

        const std::string insertion = genGLSLExtensionDefines(_exts);

        insertAfterVersionAndPragmaShaderStage(_glsl, insertion);
    }

protected:
    static inline std::string genGLSLExtensionDefines(const core::refctd_dynamic_array<std::string>* _exts)
    {
        std::string insertion = "\n";
        for(const std::string& ext : (*_exts))
        {
            std::string str;
            //str += "#ifndef " + ext + "\n";
            str += "\t#define NBL_IMPL_" + ext + "\n";
            //str += "#endif //" + ext + "\n";

            insertion += str;
        }

        return insertion;
    }
    static inline void insertAfterVersionAndPragmaShaderStage(std::string& _glsl, const std::string& _ins)
    {
        auto findLineJustAfterVersionOrPragmaShaderStageDirective = [&_glsl] {
            size_t hashPos = _glsl.find_first_of('#');
            if(hashPos >= _glsl.length())
                return _glsl.npos;
            if(_glsl.compare(hashPos, 8, "#version"))
                return _glsl.npos;

            size_t searchPos = hashPos + 8ull;

            size_t hashPos2 = _glsl.find_first_of('#', hashPos + 8ull);
            if(hashPos2 < _glsl.length())
            {
                char pragma_stage_str[] = "#pragma shader_stage";
                if(_glsl.compare(hashPos2, sizeof(pragma_stage_str) - 1ull, pragma_stage_str) == 0)
                    searchPos = hashPos2 + sizeof(pragma_stage_str) - 1ull;
            }
            size_t nlPos = _glsl.find_first_of('\n', searchPos);

            return (nlPos >= _glsl.length()) ? _glsl.npos : nlPos + 1ull;
        };

        const size_t pos = findLineJustAfterVersionOrPragmaShaderStageDirective();
        if(pos == _glsl.npos)
            return;

        const size_t ln = std::count(_glsl.begin(), _glsl.begin() + pos, '\n') + 1;  //+1 to count from 1

        _glsl.insert(pos, _ins + "#line " + std::to_string(ln) + "\n");
    }
};

}
}

#endif
