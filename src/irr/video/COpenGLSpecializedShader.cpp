#include "COpenGLSpecializedShader.h"
#include "spirv_cross/spirv_glsl.hpp"
#include "COpenGLDriver.h"

namespace irr { namespace video
{

static void specialize(spirv_cross::CompilerGLSL& _comp, const asset::ISpecializationInfo* _specData)
{
    std::vector<spirv_cross::SpecializationConstant> sconsts = _comp.get_specialization_constants();
    for (const auto& sc : sconsts)
    {
        spirv_cross::SPIRConstant& val = _comp.get_constant(sc.id);
        val.specialization = false; // despecializing. If spec-data is not provided, the spec constant is left with its default value. Default value can be queried through introspection mechanism.

        auto specVal = _specData->getSpecializationByteValue(sc.constant_id);
        if (!specVal.first)
            continue;
        memcpy(&val.m.c[0].r[0].u64, specVal.first, specVal.second); // this will do for spec constants of simple types (uint, int, float,...) regardless of size
        // only specialization constants of scalar types are supported (only scalar spec constants can have SpecId decoration)
    }
}

COpenGLSpecializedShader::COpenGLSpecializedShader(video::IVideoDriver* _driver, const asset::ICPUSpecializedShader* _cpushader)
{
    video::COpenGLDriver* driver = static_cast<video::COpenGLDriver*>(_driver);

    const asset::ISpecializationInfo* specData = _cpushader->getSpecializationInfo();
    const asset::IParsedShaderSource* parsed = _cpushader->getUnspecialized()->getParsed();

    spirv_cross::CompilerGLSL comp(parsed->getUnderlyingRepresentation());
    spirv_cross::CompilerGLSL::Options options;
    options.version = driver->ShaderLanguageVersion;
    options.vulkan_semantics = false; // with this it's likely that SPIRV-Cross will take care of translating push_constants block and built-in variables renaming, but need testing
    // note: push_constants should be translated into non-UBO uniform block (every member pushed with glProgramUniform* func)
    comp.set_common_options(options);

    specialize(comp, specData);
    //todo: binding reordering (replacing set,binding tuple with just binding etc.)

    std::string glslCode = comp.compile();

    // todo:
    // supposedly some manipulation on text form
    // pass GLSL code into API and get GL object
}

}}//irr::video