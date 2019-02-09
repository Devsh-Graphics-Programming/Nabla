#include "irr/asset/ICPUShader.h"
#include "spirv_cross/spirv_parser.hpp"

namespace irr { namespace asset
{

const SIntrospectionData& ICPUShader::enableIntrospection(const std::string& _entryPoint, E_SHADER_STAGE _stage)
{
    if (!m_parsed)
    {
        spirv_cross::Parser parser(reinterpret_cast<uint32_t*>(m_spirvBytecode->getPointer()), m_spirvBytecode->getSize()/4u);
        m_parsed = new spirv_cross::ParsedIR{std::move(parser.get_parsed_ir())};
    }

    auto found = m_introspectionCache.find({_entryPoint, _stage});
    if (found != m_introspectionCache.end())
        return found->second;


}

}}