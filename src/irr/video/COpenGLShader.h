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
		COpenGLShader(const char* _glsl) : m_code(core::make_smart_refctd_ptr<asset::ICPUBuffer>(strlen(_glsl)+1u)), m_containsGLSL(true)
		{
			memcpy(m_code->getPointer(), _glsl, m_code->getSize());
		}

		const asset::ICPUBuffer* getSPVorGLSL() const { return m_code.get(); };
		bool containsGLSL() const { return m_containsGLSL; }

	private:
		//! Might be GLSL null-terminated string or SPIR-V bytecode (denoted by m_containsGLSL)
		core::smart_refctd_ptr<asset::ICPUBuffer>	m_code;
		const bool									m_containsGLSL;
};

}
}

#endif