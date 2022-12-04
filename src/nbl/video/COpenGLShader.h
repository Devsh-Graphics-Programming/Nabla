// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_OPENGL_SHADER_H_INCLUDED__
#define __NBL_VIDEO_C_OPENGL_SHADER_H_INCLUDED__


#include "nbl/asset/ICPUBuffer.h"

#include "nbl/video/IGPUShader.h"


namespace nbl::video
{

class COpenGLShader : public IGPUShader
{
	public:

		COpenGLShader(
			core::smart_refctd_ptr<const ILogicalDevice>&& dev,
			core::smart_refctd_ptr<asset::ICPUBuffer>&& _spirv,
			const IShader::E_SHADER_STAGE _stage,
			const IShader::E_CONTENT_TYPE _contentType,
			std::string&& _filepathHint)
			: IGPUShader(std::move(dev), _stage, std::move(_filepathHint)),
			m_code(std::move(_spirv)), m_containsGLSL(_contentType == IShader::E_CONTENT_TYPE::ECT_GLSL)
		{
			assert(false); // OpenGL backend officially killed
		}

		inline const asset::ICPUBuffer* getContent() const { return m_code.get(); };
		inline const core::smart_refctd_ptr<asset::ICPUBuffer>& getSPVorGLSL_refctd() const { return m_code; };
		inline bool containsGLSL() const { return m_containsGLSL; }

		// TODO: eventually remove
		static const char* k_openGL2VulkanExtensionMap;
	private:
		//! Might be GLSL null-terminated string or SPIR-V bytecode (denoted by m_containsGLSL)
		core::smart_refctd_ptr<asset::ICPUBuffer>	m_code;
		const bool									m_containsGLSL;
};

}

#endif
