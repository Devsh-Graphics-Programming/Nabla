#include "COpenGLSpecializedShader.h"
#include "spirv_cross/spirv_glsl.hpp"
#include "COpenGLDriver.h"
#include "irr/asset/spvUtils.h"
#include <algorithm>


#ifdef _IRR_COMPILE_WITH_OPENGL_

namespace irr
{
namespace video
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
        uint32_t availableBinding = 0u;
        for (uint32_t i = 0u; i < res.size(); ++i) {
            const spirv_cross::Resource& r = res[i];
            const spirv_cross::SPIRType& type = _comp.get_type(r.type_id);
            _comp.set_decoration(r.id, spv::DecorationBinding, availableBinding);
            availableBinding += type.array[0];
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
    default: return GL_INVALID_ENUM;
    }
}

}//namesapce impl

}
}//irr::video

using namespace irr;
using namespace irr::video;

COpenGLSpecializedShader::COpenGLSpecializedShader(size_t _ctxCount, uint32_t _ctxID, uint32_t _GLSLversion, const asset::ICPUBuffer* _spirv, const asset::ISpecializationInfo* _specInfo, const asset::CIntrospectionData* _introspection) :
    IGPUSpecializedShader(_specInfo->shaderStage),
    m_GLnames(core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<GLuint>>(_ctxCount)),
    m_GLstage(impl::ESS2GLenum(_specInfo->shaderStage))
{
    for (auto& nm : (*m_GLnames))
        nm = 0u;
    m_GLnames->operator[](_ctxID) = compile(_GLSLversion, _spirv, _specInfo, _introspection);
}

GLuint COpenGLSpecializedShader::compile(uint32_t _GLSLversion, const asset::ICPUBuffer* _spirv, const asset::ISpecializationInfo* _specInfo, const asset::CIntrospectionData* _introspection)
{
    if (!_spirv || !_specInfo)
        return 0u;

    spirv_cross::CompilerGLSL comp(reinterpret_cast<const uint32_t*>(_spirv->getPointer()), _spirv->getSize()/4u);
    comp.set_entry_point(_specInfo->entryPoint.c_str(), asset::ESS2spvExecModel(_specInfo->shaderStage));
    spirv_cross::CompilerGLSL::Options options;
    options.version = _GLSLversion;
    //vulkan_semantics=false causes spirv_cross to translate push_constants into non-UBO uniform of struct type! Exactly like we wanted!
    options.vulkan_semantics = false; // with this it's likely that SPIRV-Cross will take care of built-in variables renaming, but need testing
    options.separate_shader_objects = true;
    comp.set_common_options(options);

    impl::specialize(comp, _specInfo);
    impl::reorderBindings(comp);

    std::string glslCode = comp.compile();
    const char* glslCode_cstr = glslCode.c_str();

    GLuint GLname = COpenGLExtensionHandler::extGlCreateShaderProgramv(m_GLstage, 1u, &glslCode_cstr);

    GLchar logbuf[1u<<12]; //4k
    COpenGLExtensionHandler::extGlGetProgramInfoLog(GLname, sizeof(logbuf), nullptr, logbuf);
    if (logbuf[0])
        os::Printer::log(logbuf, ELL_ERROR);

    GLint binaryLength = 0;
    COpenGLExtensionHandler::extGlGetProgramiv(GLname, GL_PROGRAM_BINARY_LENGTH, &binaryLength);
    m_binary.binary = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<uint8_t>>(binaryLength);
    COpenGLExtensionHandler::extGlGetProgramBinary(GLname, binaryLength, nullptr, &m_binary.format, m_binary.binary->data());

    if (m_uniformsList.empty())
	{
		assert(_introspection);
		const auto& pc = _introspection->pushConstant;
		if (pc.present)
		{
			const auto& pc_layout = pc.info;
			core::queue<SMember> q;
			SMember initial;
			initial.type = asset::EGVT_UNKNOWN_OR_STRUCT;
			initial.members = pc_layout.members;
			initial.name = pc.info.name;
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
				const GLint location = gl::extGlGetUniformLocation(GLname, top.name.c_str());
				assert(location != -1);
				m_uniformsList.emplace_back(top, location, m_uniformValues+top.offset);
			}
			std::sort(m_uniformsList.begin(), m_uniformsList.end(), [](const SUniform& a, const SUniform& b) { return a.location < b.location; });
		}
	}

    return GLname;
}

void COpenGLSpecializedShader::setUniformsImitatingPushConstants(const uint8_t* _pcData, uint32_t _ctxID) const
{
    IRR_ASSUME_ALIGNED(_pcData, 128);

    const GLuint GLname = getGLnameForCtx(_ctxID);

    using gl = COpenGLExtensionHandler;
    for (const SUniform& u : m_uniformsList)
    {
        const SMember& m = u.m;
		assert(m.mtxStride==0u || m.arrayStride%m.mtxStride==0u);
		IRR_ASSUME_ALIGNED(u.value, sizeof(float));
		IRR_ASSUME_ALIGNED(u.value, m.arrayStride);
		
		auto* baseOffset = _pcData+m.offset;
		IRR_ASSUME_ALIGNED(baseOffset, sizeof(float));
		IRR_ASSUME_ALIGNED(baseOffset, m.arrayStride);

		constexpr uint32_t MAX_DWORD_SIZE = IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE/sizeof(GLfloat);
		alignas(128u) std::array<GLfloat,MAX_DWORD_SIZE> packed_data;

        const uint32_t count = std::min<uint32_t>(m.count, MAX_DWORD_SIZE/(m.count*m.mtxRowCnt*m.mtxColCnt));
		if (!std::equal(baseOffset, baseOffset+m.arrayStride*count, u.value) || m_uniformsSetForTheVeryFirstTime)
		{
			auto is_scalar_or_vec = [&m] { return (m.mtxRowCnt>=1u && m.mtxColCnt==1u); };
			// pack the constant data as OpenGL uniform update functions expect packed arrays
			{
				const bool isRowMajor = is_scalar_or_vec() || m.rowMajor;
				const uint32_t rowOrColCnt = isRowMajor ? m.mtxRowCnt : m.mtxColCnt;
				const uint32_t len = isRowMajor ? m.mtxColCnt : m.mtxRowCnt;
				for (uint32_t i = 0u; i < count; ++i)
				for (uint32_t c = 0u; c < rowOrColCnt; ++c)
				{
					const GLfloat* in = reinterpret_cast<const GLfloat*>(baseOffset + i*m.arrayStride + c*m.mtxStride);
					GLfloat* out = packed_data.data() + (i*m.mtxRowCnt*m.mtxColCnt) + (c*len);
					std::copy(in, in+len, out);
				}
			}

			auto is_mtx = [&m] { return (m.mtxRowCnt>1u && m.mtxColCnt>1u); };
			if (is_mtx() && m.type==asset::EGVT_F32)
			{
					PFNGLPROGRAMUNIFORMMATRIX4FVPROC glProgramUniformMatrixNxMfv_fptr[3][3]{ //N - num of columns, M - num of rows because of weird OpenGL naming convention
						{&gl::extGlProgramUniformMatrix2fv, &gl::extGlProgramUniformMatrix2x3fv, &gl::extGlProgramUniformMatrix2x4fv},//2xM
						{&gl::extGlProgramUniformMatrix3x2fv, &gl::extGlProgramUniformMatrix3fv, &gl::extGlProgramUniformMatrix3x4fv},//3xM
						{&gl::extGlProgramUniformMatrix4x2fv, &gl::extGlProgramUniformMatrix4x3fv, &gl::extGlProgramUniformMatrix4fv} //4xM
					};

					glProgramUniformMatrixNxMfv_fptr[m.mtxColCnt - 2u][m.mtxRowCnt - 2u](GLname, u.location, m.count, m.rowMajor ? GL_TRUE : GL_FALSE, packed_data.data());
			}
			else if (is_scalar_or_vec())
			{
				switch (m.type) 
				{
					case asset::EGVT_F32:
					{
						PFNGLPROGRAMUNIFORM1FVPROC glProgramUniformNfv_fptr[4]{
							&gl::extGlProgramUniform1fv, &gl::extGlProgramUniform2fv, &gl::extGlProgramUniform3fv, &gl::extGlProgramUniform4fv
						};
						glProgramUniformNfv_fptr[m.mtxRowCnt-1u](GLname, u.location, m.count, packed_data.data());
						break;
					}
					case asset::EGVT_I32:
					{
						PFNGLPROGRAMUNIFORM1IVPROC glProgramUniformNiv_fptr[4]{
							&gl::extGlProgramUniform1iv, &gl::extGlProgramUniform2iv, &gl::extGlProgramUniform3iv, &gl::extGlProgramUniform4iv
						};
						glProgramUniformNiv_fptr[m.mtxRowCnt-1u](GLname, u.location, m.count, reinterpret_cast<const GLint*>(packed_data.data()));
						break;
					}
					case asset::EGVT_U32:
					{
						PFNGLPROGRAMUNIFORM1UIVPROC glProgramUniformNuiv_fptr[4]{
							&gl::extGlProgramUniform1uiv, &gl::extGlProgramUniform2uiv, &gl::extGlProgramUniform3uiv, &gl::extGlProgramUniform4uiv
						};
						glProgramUniformNuiv_fptr[m.mtxRowCnt-1u](GLname, u.location, m.count, reinterpret_cast<const GLuint*>(packed_data.data()));
						break;
					}
				}
			}
			std::copy(baseOffset, baseOffset+m.arrayStride*count, u.value);
        }
    }

    m_uniformsSetForTheVeryFirstTime = false;
}
#endif