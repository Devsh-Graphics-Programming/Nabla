#include "COpenGLSpecializedShader.h"
#include "spirv_cross/spirv_glsl.hpp"

namespace irr { namespace video
{

static void specialize(spirv_cross::CompilerGLSL& _comp, const asset::SSpecializationInfo* _specData)
{
    std::vector<spirv_cross::SpecializationConstant> sconsts = _comp.get_specialization_constants();
    for (const auto& sc : sconsts)
    {
        spirv_cross::SPIRConstant& val = _comp.get_constant(sc.id);
        val.specialization = false;
        if (val.subconstants.size())
            continue; // means it's composite constant and we'll come through its subconstants later in the loop

        auto specVal = _specData->getSpecializationByteValue(sc.constant_id);
        if (!specVal.first)
            continue;
        memcpy(&val.m.c[0].r[0], specVal.first, specVal.second); // this will do for spec constants of simple types (uint, int, float,...) regardless of size
    }
}

COpenGLSpecializedShader::COpenGLSpecializedShader(const asset::ICPUSpecializedShader* _cpushader)
{
    const asset::SSpecializationInfo* specData = _cpushader->getSpecializationInfo();
    const asset::IParsedShaderSource* parsed = _cpushader->getUnspecialized()->getParsed();

    spirv_cross::CompilerGLSL comp(parsed->getUnderlyingRepresentation());
    spirv_cross::CompilerGLSL::Options options;
    options.version = 450; // in following commits, this version should be taken from driver
    options.vulkan_semantics = false; // with this it's likely that SPIRV-Cross will take care of translating push_constants block and built-in variables renaming, but need testing
    comp.set_common_options(options);

    specialize(comp, specData);
    //todo: binding reordering (replacing set,binding tuple with just binding etc.)

    std::string glslCode = comp.compile();

    // todo:
    // supposedly some manipulation on text form
    // pass GLSL code into API and get GL object
}

}}//irr::video