#ifndef __IRR_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__
#define __IRR_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__

#include "spirv_cross/spirv_glsl.hpp"
#include "irr/asset/ICPUSpecializedShader.h"
#include "irr/video/IGPUSpecializedShader.h"
#include "COpenGLExtensionHandler.h"
#include "irr/video/COpenGLShader.h"
#include "irr/asset/CShaderIntrospector.h"
#include "irr/core/memory/refctd_dynamic_array.h"
#include "irr/video/IGPUMeshBuffer.h"
#include <algorithm>

#ifdef _IRR_COMPILE_WITH_OPENGL_

namespace irr
{
namespace video
{

class COpenGLSpecializedShader : public core::impl::ResolveAlignment<IGPUSpecializedShader,core::AllocationOverrideBase<128> >
{
		using SMember = asset::impl::SShaderMemoryBlock::SMember;

	public:
		struct SProgramBinary {
			GLenum format = 0;
			core::smart_refctd_dynamic_array<uint8_t> binary;
		};
		struct SUniform {
			SUniform(const SMember& _m) : m(_m) {}
			SMember m;
		};

		COpenGLSpecializedShader(size_t _ctxCount, uint32_t _ctxID, uint32_t _GLSLversion, const asset::ICPUBuffer* _spirv, const asset::ISpecializedShader::SInfo& _specInfo, const asset::CIntrospectionData* _introspection);

		void setUniformsImitatingPushConstants(const uint8_t* _pcData, GLuint _GLname, const GLint* _locations, uint8_t* _state) const;

		inline GLenum getOpenGLStage() const { return m_GLstage; }

		std::pair<GLuint, SProgramBinary> compile(const COpenGLPipelineLayout* _layout) const;

		const SInfo& getSpecializationInfo() const { return m_specInfo; }
		const std::array<uint64_t, 4>& getSpirvHash() const { return m_spirvHash; }

		core::SRange<const SUniform> getUniforms() const { return {m_uniformsList.data(),m_uniformsList.data()+m_uniformsList.size()}; }

	protected:
		~COpenGLSpecializedShader()
		{
			//shader programs can be shared so all names can be freed by any thread
			for (auto& n : *m_GLnames)
				COpenGLExtensionHandler::extGlDeleteProgram(n);
		}

	private:
		//! @returns GL name or zero if already compiled once or there were compilation errors.
		//to del
		GLuint compile(uint32_t _GLSLversion, const asset::ICPUBuffer* _spirv, const asset::ISpecializedShader::SInfo& _specInfo, const asset::CIntrospectionData* _introspection);

	private:
		mutable core::smart_refctd_dynamic_array<GLuint> m_GLnames;//to del
		GLenum m_GLstage;
		SProgramBinary m_binary;//to del

		SInfo m_specInfo;
		core::smart_refctd_ptr<const asset::ICPUBuffer> m_spirv;
		std::array<uint64_t, 4> m_spirvHash;
		spirv_cross::CompilerGLSL::Options m_options;

		//mutable bool m_uniformsSetForTheVeryFirstTime = true;
		//alignas(128) uint8_t m_uniformValues[IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE];
		core::vector<SUniform> m_uniformsList;
};

}
}
#endif

#endif//__IRR_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__
