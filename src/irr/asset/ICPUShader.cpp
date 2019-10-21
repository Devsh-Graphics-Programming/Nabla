#include "irr/asset/ICPUShader.h"
#include "spirv_cross/spirv_parser.hpp"
#include "spirv_cross/spirv_cross.hpp"
#include "irr/asset/spvUtils.h"
#include "irr/asset/format/EFormat.h"

namespace irr { namespace asset
{


ICPUShader::ICPUShader(ICPUBuffer* _spirv) : m_code(_spirv), m_containsGLSL(false)
{
    if (m_code)
        m_code->grab();
}

ICPUShader::ICPUShader(const char* _glsl) : 
    m_code(new ICPUBuffer(strlen(_glsl)+1u)), m_containsGLSL(true)
{
    memcpy(m_code->getPointer(), _glsl, m_code->getSize());
}

void ICPUShader::insertGLSLExtensionsDefines(std::string& _glsl, const core::refctd_dynamic_array<std::string>* _exts)
{
    auto findLineJustAfterVersionOrPragmaShaderStageDirective = 
    [&_glsl] {
        size_t hashPos = _glsl.find_first_of('#');
        assert(_glsl.compare(0, 8, "#version", hashPos, 8) == 0);

        size_t searchPos = hashPos + 8ull;

        size_t hashPos2 = _glsl.find_first_of('#', hashPos+8ull);
        char pragma_stage_str[] = "#pragma shader_stage";
        if (_glsl.compare(0, sizeof(pragma_stage_str)-1ull, pragma_stage_str, hashPos2, sizeof(pragma_stage_str)-1ull) == 0)
            searchPos = hashPos2 + sizeof(pragma_stage_str) - 1ull;

        return _glsl.find_first_of('\n', searchPos)+1ull;
    };

    const size_t pos = findLineJustAfterVersionOrPragmaShaderStageDirective();

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

}}