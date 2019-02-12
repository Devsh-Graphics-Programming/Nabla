#include "irr/asset/ICPUShader.h"
#include "spirv_cross/spirv_parser.hpp"
#include "spirv_cross/spirv_cross.hpp"


namespace irr { namespace asset
{

namespace
{
E_SHADER_STAGE spvExecModel2ESS(spv::ExecutionModel _em)
{
    using namespace spv;
    switch (_em)
    {
    case ExecutionModelVertex: return ESS_VERTEX;
    case ExecutionModelTessellationControl: return ESS_TESSELATION_CONTROL;
    case ExecutionModelTessellationEvaluation: return ESS_TESSELATION_EVALUATION;
    case ExecutionModelGeometry: return ESS_GEOMETRY;
    case ExecutionModelFragment: return ESS_FRAGMENT;
    case ExecutionModelGLCompute: return ESS_COMPUTE;
    default: return ESS_UNKNOWN;
    }
}
spv::ExecutionModel ESS2spvExecModel(E_SHADER_STAGE _ss)
{
    using namespace spv;
    switch (_ss)
    {
    case ESS_VERTEX: return ExecutionModelVertex;
    case ESS_TESSELATION_CONTROL: return ExecutionModelTessellationControl;
    case ESS_TESSELATION_EVALUATION: return ExecutionModelTessellationEvaluation;
    case ESS_GEOMETRY: return ExecutionModelGeometry;
    case ESS_FRAGMENT: return ExecutionModelFragment;
    case ESS_COMPUTE: return ExecutionModelGLCompute;
    default: return ExecutionModelMax;
    }
}
}

void ICPUShader::enableIntrospection()
{
    if (!m_parsed)
        m_parsed = parseSPIR_V();

    spirv_cross::Compiler comp(*m_parsed);
    auto eps = getStageEntryPoints(comp);

    for (const auto& ep : eps)
        m_introspectionCache.emplace(ep, doIntrospection(comp, ep));
}

auto ICPUShader::getStageEntryPoints() -> const core::vector<SEntryPointStagePair>&
{
    if (m_entryPoints.size())
        return m_entryPoints;
    if (!m_parsed)
        m_parsed = parseSPIR_V();

    spirv_cross::Compiler comp(*m_parsed);

    return getStageEntryPoints(comp);
}

auto ICPUShader::getStageEntryPoints(spirv_cross::Compiler & _comp) -> const core::vector<SEntryPointStagePair>&
{
    if (m_entryPoints.size())
        return m_entryPoints;
    if (!m_parsed)
        m_parsed = parseSPIR_V();

    auto eps = _comp.get_entry_points_and_stages();
    m_entryPoints.reserve(eps.size());

    for (const auto& ep : eps)
        m_entryPoints.emplace_back(ep.name, spvExecModel2ESS(ep.execution_model));
    std::sort(m_entryPoints.begin(), m_entryPoints.end());

    return m_entryPoints;
}

spirv_cross::ParsedIR* ICPUShader::parseSPIR_V() const
{
    assert(!m_parsed && "probably memleak"); // shouldn't be called more then once per object
    spirv_cross::Parser parser(reinterpret_cast<uint32_t*>(m_spirvBytecode->getPointer()), m_spirvBytecode->getSize()/4u);
    return _IRR_NEW(spirv_cross::ParsedIR, std::move(parser.get_parsed_ir()));
}

SIntrospectionData ICPUShader::doIntrospection(spirv_cross::Compiler& _comp, const SEntryPointStagePair& _ep) const
{
    spv::ExecutionModel stage = ESS2spvExecModel(_ep.second);
    if (stage == spv::ExecutionModelMax)
        return SIntrospectionData();

    SIntrospectionData introData;
    auto addResource_common = [&introData, &_comp] (const spirv_cross::Resource& r, E_SHADER_RESOURCE_TYPE restype) -> SShaderResourceVariant& {
        const uint32_t descSet = _comp.get_decoration(r.id, spv::DecorationDescriptorSet);
        assert(descSet < 4u);
        introData.descriptorSetBindings[descSet].emplace_back();
        SShaderResourceVariant& res = introData.descriptorSetBindings[descSet].back();
        res.type = restype;
        res.binding = _comp.get_decoration(r.id, spv::DecorationBinding);
        return res;
    };

    _comp.set_entry_point(_ep.first, stage);
    spirv_cross::ShaderResources resources = _comp.get_shader_resources(_comp.get_active_interface_variables());
    for (const spirv_cross::Resource& r : resources.uniform_buffers)
    {
        SShaderResourceVariant& res = addResource_common(r, ESRT_UNIFORM_BUFFER);
    }
    for (const spirv_cross::Resource& r : resources.storage_buffers)
    {
        SShaderResourceVariant& res = addResource_common(r, ESRT_STORAGE_BUFFER);
    }
    for (const spirv_cross::Resource& r : resources.stage_inputs)
    {
        SShaderResourceVariant& res = addResource_common(r, ESRT_STAGE_INPUT);
    }
    for (const spirv_cross::Resource& r : resources.stage_outputs)
    {
        SShaderResourceVariant& res = addResource_common(r, ESRT_STAGE_OUTPUT);
    }
    for (const spirv_cross::Resource& r : resources.subpass_inputs)
    {
        SShaderResourceVariant& res = addResource_common(r, ESRT_INPUT_ATTACHMENT);
    }
    for (const spirv_cross::Resource& r : resources.storage_images)
    {
        const spirv_cross::SPIRType type = _comp.get_type(r.id);
        const bool buffer = type.image.dim == spv::DimBuffer;
        SShaderResourceVariant& res = addResource_common(r, buffer ? ESRT_STORAGE_TEXEL_BUFFER : ESRT_STORAGE_IMAGE);
    }
    for (const spirv_cross::Resource& r : resources.sampled_images)
    {
        SShaderResourceVariant& res = addResource_common(r, ESRT_COMBINED_IMAGE_SAMPLER);
    }
    for (const spirv_cross::Resource& r : resources.push_constant_buffers)
    {
        SShaderResourceVariant& res = addResource_common(r, ESRT_PUSH_CONSTANT_BLOCK);
    }
    for (const spirv_cross::Resource& r : resources.separate_images)
    {
        const spirv_cross::SPIRType type = _comp.get_type(r.id);
        const bool buffer = type.image.dim == spv::DimBuffer;
        SShaderResourceVariant& res = addResource_common(r, buffer ? ESRT_UNIFORM_TEXEL_BUFFER : ESRT_SAMPLED_IMAGE);
    }
    for (const spirv_cross::Resource& r : resources.separate_samplers)
    {
        SShaderResourceVariant& res = addResource_common(r, ESRT_SAMPLER);
    }
}

}}