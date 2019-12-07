#include "irr/asset/ICPUShader.h"
#include "spirv_cross/spirv_parser.hpp"
#include "spirv_cross/spirv_cross.hpp"
#include "irr/asset/spvUtils.h"
#include "irr/asset/format/EFormat.h"


using namespace irr;
using namespace irr::asset;


void ICPUShader::insertGLSLExtensionsDefines(std::string& _glsl, const core::refctd_dynamic_array<std::string>* _exts)
{
    auto findLineJustAfterVersionOrPragmaShaderStageDirective = [&_glsl]
	{
        size_t hashPos = _glsl.find_first_of('#');
		if (_glsl.compare(0, 8, "#version", hashPos, 8))
			return ~0ull;

        size_t searchPos = hashPos + 8ull;

        size_t hashPos2 = _glsl.find_first_of('#', hashPos+8ull);
		if (hashPos2<_glsl.length())
		{
			char pragma_stage_str[] = "#pragma shader_stage";
			if (_glsl.compare(0, sizeof(pragma_stage_str)-1ull, pragma_stage_str, hashPos2, sizeof(pragma_stage_str)-1ull) == 0)
				searchPos = hashPos2 + sizeof(pragma_stage_str) - 1ull;
		}

        return _glsl.find_first_of('\n', searchPos)+1ull;
    };

    const size_t pos = findLineJustAfterVersionOrPragmaShaderStageDirective();
	if (pos == ~0ull)
		return;

    std::string insertion = "\n";
    for (const std::string& ext : (*_exts))
    {
        std::string str = "#ifndef " + ext + "\n";
        str += "\t#define " + ext + "\n";
        str += "#endif //" + ext + "\n";

        insertion += str;
    }

    _glsl.insert(pos, insertion);
}