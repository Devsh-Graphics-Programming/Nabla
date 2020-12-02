// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_OPENGL_SHADER_H_INCLUDED__
#define __NBL_VIDEO_C_OPENGL_SHADER_H_INCLUDED__

#include "nbl/asset/ICPUBuffer.h"
#include "nbl/video/IGPUShader.h"

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