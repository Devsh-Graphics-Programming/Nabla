#include "irr/asset/ICPUShader.h"
#include "spirv_cross/spirv_parser.hpp"
#include "spirv_cross/spirv_cross.hpp"
#include "irr/asset/EFormat.h"
#include "irr/asset/spvUtils.h"
#include "CMemoryFile.h"

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

}}