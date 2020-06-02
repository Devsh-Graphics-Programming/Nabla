#ifndef __IRR_C_OPENGL_SHADER_H_INCLUDED__
#define __IRR_C_OPENGL_SHADER_H_INCLUDED__

#include "irr/asset/ICPUBuffer.h"
#include "irr/video/IGPUShader.h"

namespace irr
{
namespace video
{

class COpenGLShader : public IGPUShader
{
	public:
		COpenGLShader(core::smart_refctd_ptr<asset::ICPUBuffer>&& _spirv) : m_code(std::move(_spirv)), m_containsGLSL(false) {}
		COpenGLShader(core::smart_refctd_ptr<asset::ICPUBuffer>&& _glsl, buffer_contains_glsl_t buffer_contains_glsl) : m_code(std::move(_glsl)), m_containsGLSL(true) {}

		const asset::ICPUBuffer* getSPVorGLSL() const { return m_code.get(); };
		bool containsGLSL() const { return m_containsGLSL; }

	private:
		friend class COpenGLDriver;
		//! Might be GLSL null-terminated string or SPIR-V bytecode (denoted by m_containsGLSL)
		core::smart_refctd_ptr<asset::ICPUBuffer>	m_code;
		const bool									m_containsGLSL;
};

}
}

#endif