#ifndef __IRR_I_SHADER_H_INCLUDED__
#define __IRR_I_SHADER_H_INCLUDED__

#include <algorithm>
#include <string>


#include "irr/core/core.h"

namespace spirv_cross
{
    class ParsedIR;
    class Compiler;
    struct SPIRType;
}

namespace irr
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
		struct buffer_contains_glsl_t {};
		_IRR_STATIC_INLINE const buffer_contains_glsl_t buffer_contains_glsl = {};

		static inline void insertGLSLExtensionsDefines(std::string& _glsl, const core::refctd_dynamic_array<std::string>* _exts)
		{
            if (!_exts)
                return;

			auto findLineJustAfterVersionOrPragmaShaderStageDirective = [&_glsl]
			{
				size_t hashPos = _glsl.find_first_of('#');
                if (hashPos >= _glsl.length())
                    return _glsl.npos;
				if (_glsl.compare(hashPos, 8, "#version"))
					return _glsl.npos;

				size_t searchPos = hashPos + 8ull;

				size_t hashPos2 = _glsl.find_first_of('#', hashPos+8ull);
				if (hashPos2<_glsl.length())
				{
					char pragma_stage_str[] = "#pragma shader_stage";
					if (_glsl.compare(hashPos2, sizeof(pragma_stage_str)-1ull, pragma_stage_str) == 0)
						searchPos = hashPos2 + sizeof(pragma_stage_str) - 1ull;
				}
                size_t nlPos = _glsl.find_first_of('\n', searchPos);

				return (nlPos >= _glsl.length()) ? _glsl.npos : nlPos+1ull;
			};

			const size_t pos = findLineJustAfterVersionOrPragmaShaderStageDirective();
			if (pos == _glsl.npos)
				return;

			const size_t ln = std::count(_glsl.begin(),_glsl.begin()+pos, '\n')+1;//+1 to count from 1

			std::string insertion = "\n";
			for (const std::string& ext : (*_exts))
			{
				std::string str = "#ifndef " + ext + "\n";
				str += "\t#define IRR_" + ext + "\n";
				str += "#endif //" + ext + "\n";

				insertion += str;
			}
			insertion += "#line " + std::to_string(ln) + "\n";

			_glsl.insert(pos, insertion);
		}
};

}
}

#endif//__IRR_I_SHADER_H_INCLUDED__
