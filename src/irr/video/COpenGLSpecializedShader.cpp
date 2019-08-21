#include "COpenGLSpecializedShader.h"
#include "spirv_cross/spirv_glsl.hpp"
#include "COpenGLDriver.h"
#include "irr/asset/spvUtils.h"

namespace irr { namespace video
{

namespace impl
{
static void specialize(spirv_cross::CompilerGLSL& _comp, const asset::ISpecializationInfo* _specData)
{
    spirv_cross::SmallVector<spirv_cross::SpecializationConstant> sconsts = _comp.get_specialization_constants();
    for (const auto& sc : sconsts)
    {
        spirv_cross::SPIRConstant& val = _comp.get_constant(sc.id);
        val.specialization = false; // despecializing. If spec-data is not provided, the spec constant is left with its default value. Default value can be queried through introspection mechanism.

        auto specVal = _specData->getSpecializationByteValue(sc.constant_id);
        if (!specVal.first)
            continue;
        memcpy(&val.m.c[0].r[0].u64, specVal.first, specVal.second); // this will do for spec constants of simple types (uint, int, float,...) regardless of size
        // but it's ok since ,according to SPIR-V specification, only specialization constants of scalar types are supported (only scalar spec constants can have SpecId decoration)
    }
}

static void reorderBindings(spirv_cross::CompilerGLSL& _comp)
{
    auto sort_ = [&_comp](spirv_cross::SmallVector<spirv_cross::Resource>& res) {
        using R_t = spirv_cross::Resource;
        auto cmp = [&_comp](const R_t& a, const R_t& b) {
            uint32_t a_set = _comp.get_decoration(a.id, spv::DecorationDescriptorSet);
            uint32_t b_set = _comp.get_decoration(b.id, spv::DecorationDescriptorSet);
            if (a_set < b_set)
                return true;
            if (a_set > b_set)
                return false;
            uint32_t a_bnd = _comp.get_decoration(a.id, spv::DecorationBinding);
            uint32_t b_bnd = _comp.get_decoration(b.id, spv::DecorationBinding);
            return a_bnd < b_bnd;
        };
        std::sort(res.begin(), res.end(), cmp);
    };
    auto reorder_ = [&_comp](spirv_cross::SmallVector<spirv_cross::Resource>& res) {
        for (uint32_t i = 0u; i < res.size(); ++i) {
            const spirv_cross::Resource& r = res[i];
            _comp.set_decoration(r.id, spv::DecorationBinding, i);
            _comp.unset_decoration(r.id, spv::DecorationDescriptorSet);
        }
    };

    spirv_cross::ShaderResources resources = _comp.get_shader_resources(_comp.get_active_interface_variables());

    auto samplers = std::move(resources.sampled_images);
    for (const auto& samplerBuffer : resources.separate_images)
        if (_comp.get_type(samplerBuffer.type_id).image.dim == spv::DimBuffer)
            samplers.push_back(samplerBuffer);//see https://github.com/KhronosGroup/SPIRV-Cross/wiki/Reflection-API-user-guide for explanation
    sort_(samplers);
    reorder_(samplers);

    auto images = std::move(resources.storage_images);
    sort_(images);
    reorder_(images);

    auto ubos = std::move(resources.uniform_buffers);
    sort_(ubos);
    reorder_(ubos);

    auto ssbos = std::move(resources.storage_buffers);
    sort_(ssbos);
    reorder_(ssbos);
}
}//namesapce impl

COpenGLSpecializedShader::COpenGLSpecializedShader(video::IVideoDriver* _driver, const asset::ICPUSpecializedShader* _cpushader)
{
    video::COpenGLDriver* driver = static_cast<video::COpenGLDriver*>(_driver);

    core::smart_refctd_ptr<const asset::ISpecializationInfo> specData(_cpushader->getSpecializationInfo());
    core::smart_refctd_ptr<const asset::IParsedShaderSource> parsed(_cpushader->getUnspecialized()->getParsed());

    spirv_cross::CompilerGLSL comp(parsed->getUnderlyingRepresentation());
    comp.set_entry_point(specData->entryPoint.c_str(), asset::ESS2spvExecModel(specData->shaderStage));
    spirv_cross::CompilerGLSL::Options options;
    options.version = driver->ShaderLanguageVersion;
    //vulkan_semantics=false cases spirv_cross to translate push_constants into non-UBO uniform of struct type! Exactly like we wanted!
    options.vulkan_semantics = false; // with this it's likely that SPIRV-Cross will take care of built-in variables renaming, but need testing
    options.separate_shader_objects = true;
    comp.set_common_options(options);

    impl::specialize(comp, specData.get());
    impl::reorderBindings(comp);

    std::string glslCode = comp.compile();
    //printf(glslCode.c_str());

    // TODO:
    // pass GLSL code into API and get GL object
    // what should be interface for setting push_constants (regular uniform on GL backend) for now?
}

}}//irr::video