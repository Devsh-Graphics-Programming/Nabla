#ifndef __IRR_I_SHADER_H_INCLUDED__
#define __IRR_I_SHADER_H_INCLUDED__

#include <algorithm>
#include <string>


#include "irr/core/SRange.h"
#include "irr/core/IBuffer.h"

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

template<typename BufferType>
class IShader
{
	public:
		static inline void insertGLSLExtensionsDefines(std::string& _glsl, const core::refctd_dynamic_array<std::string>* _exts)
		{
            if (!_exts)
                return;

			auto findLineJustAfterVersionOrPragmaShaderStageDirective = [&_glsl]
			{
				size_t hashPos = _glsl.find_first_of('#');
                if (hashPos >= _glsl.length())
                    return ~0ull;
				if (_glsl.compare(hashPos, 8, "#version"))
					return ~0ull;

				size_t searchPos = hashPos + 8ull;

				size_t hashPos2 = _glsl.find_first_of('#', hashPos+8ull);
				if (hashPos2<_glsl.length())
				{
					char pragma_stage_str[] = "#pragma shader_stage";
					if (_glsl.compare(hashPos2, sizeof(pragma_stage_str)-1ull, pragma_stage_str) == 0)
						searchPos = hashPos2 + sizeof(pragma_stage_str) - 1ull;
				}
                size_t nlPos = _glsl.find_first_of('\n', searchPos);

				return (nlPos >= _glsl.length()) ? ~0ull : nlPos+1ull;
			};

			const size_t pos = findLineJustAfterVersionOrPragmaShaderStageDirective();
			if (pos == ~0ull)
				return;

			std::string insertion = "\n";
			for (const std::string& ext : (*_exts))
			{
				std::string str = "#ifndef " + ext + "\n";
				str += "\t#define IRR_" + ext + "\n";
				str += "#endif //" + ext + "\n";

				insertion += str;
			}

			_glsl.insert(pos, insertion);
		}
};

}
}

#endif//__IRR_I_SHADER_H_INCLUDED__
