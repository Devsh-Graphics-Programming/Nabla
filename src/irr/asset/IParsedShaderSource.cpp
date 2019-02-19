#include "irr/asset/IParsedShaderSource.h"
#include "spirv_cross/spirv_parser.hpp"

namespace irr { namespace asset
{

IParsedShaderSource::~IParsedShaderSource()
{
    if (m_parsed)
        _IRR_DELETE(const_cast<spirv_cross::ParsedIR*>(m_parsed));
    if (m_raw)
        m_raw->drop();
}

IParsedShaderSource::IParsedShaderSource(const asset::ICPUBuffer* _spirvBytecode) : IParsedShaderSource(_spirvBytecode, core::defer)
{
    parse();
}

IParsedShaderSource::IParsedShaderSource(const asset::ICPUBuffer* _spirvBytecode, core::defer_t) : m_parsed(nullptr), m_raw(_spirvBytecode)
{
    m_raw->grab();
}

void IParsedShaderSource::parse() const
{
    if (m_parsed || !m_raw)
        return;
    spirv_cross::Parser parser(reinterpret_cast<const uint32_t*>(m_raw->getPointer()), m_raw->getSize()/4u);
    m_parsed = _IRR_NEW(spirv_cross::ParsedIR, std::move(parser.get_parsed_ir()));

    m_raw->drop();
    m_raw = nullptr; // actually we don't need raw SPIR-V any more at this point
}

}}//irr::asset