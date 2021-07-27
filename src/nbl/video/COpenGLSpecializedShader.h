// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__
#define __NBL_VIDEO_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__


#include <algorithm>

#include "nbl/core/containers/refctd_dynamic_array.h"

#include "nbl/asset/ICPUSpecializedShader.h"
#include "nbl/video/IGPUSpecializedShader.h"
#include "nbl/asset/utils/CShaderIntrospector.h"
#include "nbl/video/COpenGLShader.h"
#include "nbl/video/IGPUSpecializedShader.h"
#include "nbl/video/COpenGLPipelineLayout.h"

#include "spirv_cross/spirv_glsl.hpp"


#ifdef _NBL_COMPILE_WITH_OPENGL_

namespace nbl
{
namespace video
{

class IOpenGL_FunctionTable;

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

		static inline bool getUniformsFromPushConstants(core::vector<SUniform>* uniformList,const asset::CIntrospectionData* _introspection, system::ILogger* logger)
		{
			assert(_introspection);
			const auto& pc = _introspection->pushConstant;
			if (!pc.present)
				return true;
			if (!pc.info.name.size()) // cannot handle anonymous push constant blocks (we loose the names)
			{
				logger->log("Push Constant blocks need to be named (limitation of SPIR-V Cross). Creation of COpenGLSpecializedShader failed.", system::ILogger::ELL_ERROR);
				return false;
			}
		
			const auto& pc_layout = pc.info;
			core::queue<SMember> q;
			SMember initial;
			initial.type = asset::EGVT_UNKNOWN_OR_STRUCT;
			initial.members = pc_layout.members;
			initial.name = pc.info.name;
			q.push(initial);

			struct UniformHash
			{
				inline size_t operator()(const SUniform& other) const
				{
					return std::hash<std::string>()(other.m.name);
				}
			};
			struct UniformKeyEqualTo
			{
				inline size_t operator()(const SUniform& A, const SUniform& B) const
				{
					return A.m.name==B.m.name;
				}
			};
			core::unordered_set<SUniform,UniformHash,UniformKeyEqualTo> uniformSet;
			while (!q.empty())
			{
				const SMember top = q.front();
				q.pop();
				if (top.type == asset::EGVT_UNKNOWN_OR_STRUCT && top.members.count) {
					for (size_t i = 0ull; i < top.members.count; ++i) {
						SMember m = top.members.array[i];
						m.name = (top.name.size() ? (top.name+"."):"")+m.name;
						if (m.count > 1u)
							m.name += "[0]";
						q.push(m);
					}
					continue;
				}
				auto result = uniformSet.insert(top);
				if (!result.second)
					return false;
			}
			uniformList->clear();
			uniformList->insert(uniformList->begin(),uniformSet.begin(),uniformSet.end());
			return true;
		}

		COpenGLSpecializedShader(ILogicalDevice* dev, uint32_t _GLSLversion, const asset::ICPUBuffer* _spirv, const asset::ISpecializedShader::SInfo& _specInfo, core::vector<SUniform>&& uniformList);

		inline GLenum getOpenGLStage() const { return m_GLstage; }

		std::pair<GLuint, SProgramBinary> compile(IOpenGL_FunctionTable* gl, bool needClipControlWorkaround, const COpenGLPipelineLayout* _layout, const spirv_cross::ParsedIR* _parsedSpirv) const;

		const SInfo& getSpecializationInfo() const { return m_specInfo; }
		const std::array<uint64_t, 4>& getSpirvHash() const { return m_spirvHash; }
		const asset::ICPUBuffer* getSpirv() const { return m_spirv.get(); }
		core::SRange<const SUniform> getUniforms() const { return {m_uniformsList.data(), m_uniformsList.data()+m_uniformsList.size()}; }
		core::SRange<const GLint> getLocations() const { return {m_locations.data(), m_locations.data()+m_locations.size()}; }

	protected:
		~COpenGLSpecializedShader() = default;

	private:
		void gatherUniformLocations(IOpenGL_FunctionTable* gl, GLuint _GLname) const;

		GLenum m_GLstage;

		SInfo m_specInfo;
		core::smart_refctd_ptr<const asset::ICPUBuffer> m_spirv;
		std::array<uint64_t, 4> m_spirvHash;
		spirv_cross::CompilerGLSL::Options m_options;

		//mutable bool m_uniformsSetForTheVeryFirstTime = true;
		//alignas(128) uint8_t m_uniformValues[IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE];
		core::vector<SUniform> m_uniformsList;
		mutable core::vector<GLint> m_locations;
};

}
}
#endif

#endif
