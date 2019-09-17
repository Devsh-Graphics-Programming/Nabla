#include "COpenGLSpecializedShader.h"
#include "spirv_cross/spirv_glsl.hpp"
#include "COpenGLDriver.h"
#include "irr/asset/spvUtils.h"
#include <algorithm>

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
        memcpy(&val.m.c[0].r[0].u64, specVal.first, specVal.second); // this will do for spec constants of scalar types (uint, int, float,...) regardless of size
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

static GLenum ESS2GLenum(asset::E_SHADER_STAGE _stage)
{
    using namespace asset;
    switch (_stage)
    {
    case ESS_VERTEX: return GL_VERTEX_SHADER;
    case ESS_TESSELATION_CONTROL: return GL_TESS_CONTROL_SHADER;
    case ESS_TESSELATION_EVALUATION: return GL_TESS_EVALUATION_SHADER;
    case ESS_GEOMETRY: return GL_GEOMETRY_SHADER;
    case ESS_FRAGMENT: return GL_FRAGMENT_SHADER;
    case ESS_COMPUTE: return GL_COMPUTE_SHADER;
    default: return 0u;
    }
}

}//namesapce impl

COpenGLSpecializedShader::COpenGLSpecializedShader(uint32_t _glslVersion, const asset::ICPUBuffer* _spirv, const asset::ISpecializationInfo* _specInfo, const asset::CIntrospectionData* _introspection) :
    m_GLname(0u),
    m_stage(impl::ESS2GLenum(_specInfo->shaderStage))
{
    spirv_cross::CompilerGLSL comp(reinterpret_cast<const uint32_t*>(_spirv->getPointer()), _spirv->getSize()/4u);
    comp.set_entry_point(_specInfo->entryPoint.c_str(), asset::ESS2spvExecModel(_specInfo->shaderStage));
    spirv_cross::CompilerGLSL::Options options;
    options.version = _glslVersion;
    //vulkan_semantics=false cases spirv_cross to translate push_constants into non-UBO uniform of struct type! Exactly like we wanted!
    options.vulkan_semantics = false; // with this it's likely that SPIRV-Cross will take care of built-in variables renaming, but need testing
    options.separate_shader_objects = true;
    comp.set_common_options(options);

    impl::specialize(comp, _specInfo);
    impl::reorderBindings(comp);

    std::string glslCode = comp.compile();
    const char* glslCode_cstr = glslCode.c_str();
    //printf(glslCode.c_str());

    m_GLname = COpenGLExtensionHandler::extGlCreateShaderProgramv(m_stage, 1u, &glslCode_cstr);

    GLchar logbuf[1u<<12]; //4k
    COpenGLExtensionHandler::extGlGetProgramInfoLog(m_GLname, sizeof(logbuf), nullptr, logbuf);
    if (logbuf[0])
        os::Printer::log(logbuf, ELL_ERROR);

    m_introspectionData = core::smart_refctd_ptr<asset::CIntrospectionData>(_introspection);
}

void COpenGLSpecializedShader::setUniformsImitatingPushConstants(const uint8_t* _pcData)
{
    if (m_uniformsList.empty())
        buildUniformsList();

    using gl = COpenGLExtensionHandler;
    for (const SUniform& u : m_uniformsList)
    {
        const SMember& m = u.m;
        auto is_mtx = [&m] { return (m.mtxRowCnt>1u && m.mtxColCnt>1u); };
        auto is_single_or_vec = [&m] { return (m.mtxRowCnt>=1u && m.mtxColCnt==1u); };

        if (is_mtx() && m.type==asset::EGVT_F32) {
            std::array<GLfloat, IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE/sizeof(GLfloat)> matrix_data;
            const uint32_t count = std::min<uint32_t>(m.count, matrix_data.size()/(m.count*m.mtxRowCnt*m.mtxColCnt));
            for (uint32_t i = 0u; i < count; ++i)
            {
                const uint32_t rowOrColCnt = m.rowMajor ? m.mtxRowCnt : m.mtxColCnt;
                for (uint32_t c = 0u; c < rowOrColCnt; ++c) {
                    const GLfloat* col = reinterpret_cast<const GLfloat*>(_pcData + m.offset + i*m.arrayStride + c*m.mtxStride);
                    const uint32_t len = m.rowMajor ? m.mtxColCnt : m.mtxRowCnt;
                    GLfloat* ptr = matrix_data.data() + (i*m.mtxRowCnt*m.mtxColCnt) + (c*len);
                    memcpy(ptr, col, len*sizeof(GLfloat));
                }
            }
            PFNGLPROGRAMUNIFORMMATRIX4FVPROC glProgramUniformMatrixNxMfv_fptr[3][3]{ //N - num of columns, M - num of rows because of weird OpenGL naming convention
                {&gl::extGlProgramUniformMatrix2fv, &gl::extGlProgramUniformMatrix2x3fv, &gl::extGlProgramUniformMatrix2x4fv},//2xM
                {&gl::extGlProgramUniformMatrix3x2fv, &gl::extGlProgramUniformMatrix3fv, &gl::extGlProgramUniformMatrix3x4fv},//3xM
                {&gl::extGlProgramUniformMatrix4x2fv, &gl::extGlProgramUniformMatrix4x3fv, &gl::extGlProgramUniformMatrix4fv} //4xM
            };
            glProgramUniformMatrixNxMfv_fptr[m.mtxColCnt-2u][m.mtxRowCnt-2u](m_GLname, u.location, m.count, m.rowMajor?GL_TRUE:GL_FALSE, matrix_data.data());
        }
        else if (is_single_or_vec()) {
            std::array<GLuint, IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE/sizeof(GLuint)> vector_data;
            const uint32_t count = vector_data.size()/(m.count*m.mtxRowCnt);
            for (uint32_t i = 0u; i < count; ++i) {
                const GLuint* vec = reinterpret_cast<const GLuint*>(_pcData + i*m.mtxRowCnt);
                GLuint* ptr = vector_data.data() + i*m.mtxRowCnt;
                memcpy(ptr, vec, sizeof(GLuint)*m.mtxRowCnt);
            }
            switch (m.type) {
            case asset::EGVT_F32:
            {
                PFNGLPROGRAMUNIFORM1FVPROC glProgramUniformNfv_fptr[4]{
                    &gl::extGlProgramUniform1fv, &gl::extGlProgramUniform2fv, &gl::extGlProgramUniform3fv, &gl::extGlProgramUniform4fv
                };
                glProgramUniformNfv_fptr[m.mtxRowCnt-1u](m_GLname, u.location, m.count, reinterpret_cast<const GLfloat*>(vector_data.data()));
                break;
            }
            case asset::EGVT_I32:
            {
                PFNGLPROGRAMUNIFORM1IVPROC glProgramUniformNiv_fptr[4]{
                    &gl::extGlProgramUniform1iv, &gl::extGlProgramUniform2iv, &gl::extGlProgramUniform3iv, &gl::extGlProgramUniform4iv
                };
                glProgramUniformNiv_fptr[m.mtxRowCnt-1u](m_GLname, u.location, m.count, reinterpret_cast<const GLint*>(vector_data.data()));
                break;
            }
            case asset::EGVT_U32:
            {
                PFNGLPROGRAMUNIFORM1UIVPROC glProgramUniformNuiv_fptr[4]{
                    &gl::extGlProgramUniform1uiv, &gl::extGlProgramUniform2uiv, &gl::extGlProgramUniform3uiv, &gl::extGlProgramUniform4uiv
                };
                glProgramUniformNuiv_fptr[m.mtxRowCnt-1u](m_GLname, u.location, m.count, vector_data.data());
                break;
            }
            }
        }
    }
}

void COpenGLSpecializedShader::buildUniformsList()
{
    const auto& pc = m_introspectionData->pushConstant;
    if (!pc.present)
        return;

    const auto& pc_layout = pc.info;
    core::queue<SMember> q;
    SMember initial;
    initial.type = asset::EGVT_UNKNOWN_OR_STRUCT;
    initial.members = pc_layout.members;
    q.push(initial);
    while (!q.empty())
    {
        const SMember top = q.front();
        q.pop();
        if (top.type == asset::EGVT_UNKNOWN_OR_STRUCT && top.members.count) {
            for (size_t i = 0ull; i < top.members.count; ++i) {
                SMember m = top.members.array[i];
                m.name = top.name + "." + m.name;
                if (m.count > 1u)
                    m.name += "[0]";
                q.push(m);
            }
            continue;
        }
        using gl = COpenGLExtensionHandler;
        const GLint location = gl::extGlGetUniformLocation(m_GLname, top.name.c_str());
        assert(location != -1);
        m_uniformsList.emplace_back(top, location);
    }
    std::sort(m_uniformsList.begin(), m_uniformsList.end(), [](const SUniform& a, const SUniform& b) { return a.location < b.location; });
}

}}//irr::video