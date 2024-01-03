// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_SPIRV_INTROSPECTOR_H_INCLUDED_
#define _NBL_ASSET_C_SPIRV_INTROSPECTOR_H_INCLUDED_

#include "nbl/core/declarations.h"

#include <cstdint>
#include <memory>

#include "nbl/asset/ICPUShader.h"
#include "nbl/asset/ICPUImageView.h"
#include "nbl/asset/ICPUComputePipeline.h"
#include "nbl/asset/ICPURenderpassIndependentPipeline.h"
#include "nbl/asset/utils/ShaderRes.h"
#include "nbl/asset/utils/CGLSLCompiler.h"


#include "nbl/core/definitions.h"


namespace spirv_cross
{
    class ParsedIR;
    class Compiler;
    struct SPIRType;
}

namespace nbl::asset
{
class NBL_API2 CSPIRVIntrospector : public core::Uncopyable
{
	public:
		static IDescriptor::E_TYPE resType2descType(E_SHADER_RESOURCE_TYPE _t)
		{
			switch (_t)
			{
				case ESRT_COMBINED_IMAGE_SAMPLER:
					return IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER;
					break;
				case ESRT_STORAGE_IMAGE:
					return IDescriptor::E_TYPE::ET_STORAGE_IMAGE;
					break;
				case ESRT_UNIFORM_TEXEL_BUFFER:
					return IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER;
					break;
				case ESRT_STORAGE_TEXEL_BUFFER:
					return IDescriptor::E_TYPE::ET_STORAGE_TEXEL_BUFFER;
					break;
				case ESRT_UNIFORM_BUFFER:
					return IDescriptor::E_TYPE::ET_UNIFORM_BUFFER;
					break;
				case ESRT_STORAGE_BUFFER:
					return IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
					break;
				default:
					break;
			}
			return IDescriptor::E_TYPE::ET_COUNT;
		}

	class NBL_API2 CIntrospectionData : public core::IReferenceCounted
	{
		protected:
			~CIntrospectionData();

		public:
			struct SSpecConstant
			{
				uint32_t id;
				size_t byteSize;
				E_GLSL_VAR_TYPE type;
				std::string name;
				union {
					uint64_t u64;
					int64_t i64;
					uint32_t u32;
					int32_t i32;
					double f64;
					float f32;
				} defaultValue;
			};
			//! Sorted by `id`
			core::vector<SSpecConstant> specConstants;
			//! Each vector is sorted by `binding`
			core::vector<SShaderResourceVariant> descriptorSetBindings[4];
			//! Sorted by `location`
			core::vector<SShaderInfoVariant> inputOutput;

			struct {
				bool present;
				core::string name;
				SShaderPushConstant info;
			} pushConstant;

			bool canSpecializationlesslyCreateDescSetFrom() const
			{
				for (const auto& descSet : descriptorSetBindings)
				{
					auto found = std::find_if(descSet.begin(), descSet.end(), [](const SShaderResourceVariant& bnd) { return bnd.descCountIsSpecConstant; });
					if (found != descSet.end())
						return false;
				}
				return true;
			}
		};

		struct SIntrospectionParams
		{
			std::string entryPoint;
			core::smart_refctd_ptr<const ICPUShader> cpuShader;

			bool operator==(const SIntrospectionParams& rhs) const
			{
				if (entryPoint != rhs.entryPoint)
					return false;
				if (!rhs.cpuShader)
					return false;
				if (cpuShader->getStage() != rhs.cpuShader->getStage())
					return false;
				if (cpuShader->getContentType() != rhs.cpuShader->getContentType())
					return false;
				if (cpuShader->getContent()->getSize() != rhs.cpuShader->getContent()->getSize())
					return false;
				return memcmp(cpuShader->getContent()->getPointer(), rhs.cpuShader->getContent()->getPointer(), cpuShader->getContent()->getSize()) == 0;;
			}
		};

		//In the future there's also going list of enabled extensions
		CSPIRVIntrospector() = default;

		//! params.cpuShader.contentType should be ECT_SPIRV
		//! the compiled SPIRV must be compiled with IShaderCompiler::SCompilerOptions::debugInfoFlags enabling EDIF_SOURCE_BIT implicitly or explicitly, with no `spirvOptimizer` used in order to include names in introspection data
		core::smart_refctd_ptr<const CIntrospectionData> introspect(const SIntrospectionParams& params, bool insertToCache = true);

		// When the methods take a span of shaders, they are computing things for an imaginary pipeline that includes **all** of them
		std::pair<bool/*is shadow sampler*/, IImageView<ICPUImage>::E_TYPE> getImageInfoFromIntrospection(uint32_t set, uint32_t binding, const std::span<const ICPUShader::SSpecInfo> _infos);
		
		inline core::smart_refctd_dynamic_array<SPushConstantRange> createPushConstantRangesFromIntrospection(const std::span<const ICPUShader::SSpecInfo> _infos)
		{
			core::vector<core::smart_refctd_ptr<const CIntrospectionData>> introspections(_infos.size());
			if (!introspectAllShaders(introspections.data(),_infos))
				return nullptr;

			return createPushConstantRangesFromIntrospection_impl(introspections.data(),_infos);
		}
		inline core::smart_refctd_ptr<ICPUDescriptorSetLayout> createApproximateDescriptorSetLayoutFromIntrospection(uint32_t set, const std::span<const ICPUShader::SSpecInfo> _infos)
		{
			core::vector<core::smart_refctd_ptr<const CIntrospectionData>> introspections(_infos.size());
			if (!introspectAllShaders(introspections.data(),_infos))
				return nullptr;

			return createApproximateDescriptorSetLayoutFromIntrospection_impl(set,introspections.data(), _infos);
		}
		inline core::smart_refctd_ptr<ICPUPipelineLayout> createApproximatePipelineLayoutFromIntrospection(const std::span<const ICPUShader::SSpecInfo> _infos)
		{
			core::vector<core::smart_refctd_ptr<const CIntrospectionData>> introspections(_infos.size());
			if (!introspectAllShaders(introspections.data(),_infos))
				return nullptr;

			return createApproximatePipelineLayoutFromIntrospection_impl(introspections.data(),_infos);
		}

		//
		inline core::smart_refctd_ptr<ICPUComputePipeline> createApproximateComputePipelineFromIntrospection(const ICPUShader::SSpecInfo& info)
		{
			if (info.shader->getStage()!=IShader::ESS_COMPUTE)
				return nullptr;

			core::smart_refctd_ptr<const CIntrospectionData> introspection = nullptr;
			if (!introspectAllShaders(&introspection,{&info,1}))
				return nullptr;

			auto layout = createApproximatePipelineLayoutFromIntrospection_impl(&introspection,{&info,1});
			ICPUComputePipeline::SCreationParams params = {{.layout = layout.get()}};
			params.shader = info;
			return ICPUComputePipeline::create(params);
		}

		//
		core::smart_refctd_ptr<ICPURenderpassIndependentPipeline> createApproximateRenderpassIndependentPipelineFromIntrospection(const std::span<const ICPUShader::SSpecInfo> _infos);
	
	private:
		using mapId2SpecConst_t = core::unordered_map<uint32_t, const CIntrospectionData::SSpecConstant*>;

		core::smart_refctd_dynamic_array<SPushConstantRange> createPushConstantRangesFromIntrospection_impl(core::smart_refctd_ptr<const CIntrospectionData>* const introspections, const std::span<const ICPUShader::SSpecInfo> shaders);
		core::smart_refctd_ptr<ICPUDescriptorSetLayout> createApproximateDescriptorSetLayoutFromIntrospection_impl(uint32_t _set, core::smart_refctd_ptr<const CIntrospectionData>* const introspections, const std::span<const ICPUShader::SSpecInfo> shaders);
		core::smart_refctd_ptr<ICPUPipelineLayout> createApproximatePipelineLayoutFromIntrospection_impl(core::smart_refctd_ptr<const CIntrospectionData>* const introspections, const std::span<const ICPUShader::SSpecInfo> shaders);

		bool introspectAllShaders(core::smart_refctd_ptr<const CIntrospectionData>* introspection, const std::span<const ICPUShader::SSpecInfo> _infos);

		core::smart_refctd_ptr<const CIntrospectionData> doIntrospection(spirv_cross::Compiler& _comp, const std::string& entryPoint, const IShader::E_SHADER_STAGE stage) const;
		void shaderMemBlockIntrospection(spirv_cross::Compiler& _comp, impl::SShaderMemoryBlock& _res, uint32_t _blockBaseTypeID, uint32_t _varID, const mapId2SpecConst_t& _sortedId2sconst) const;
		size_t calcBytesizeforType(spirv_cross::Compiler& _comp, const spirv_cross::SPIRType& _type) const;

	private:

		struct KeyHasher
		{
			size_t operator()(const SIntrospectionParams& param) const 
			{
				auto stringViewHasher = std::hash<std::string_view>();

				auto code = std::string_view(reinterpret_cast<const char*>(param.cpuShader->getContent()->getPointer()), param.cpuShader->getContent()->getSize());
				size_t hash = stringViewHasher(code);

				core::hash_combine<std::string_view>(hash, std::string_view(param.entryPoint));
				core::hash_combine<uint32_t>(hash, static_cast<uint32_t>(param.cpuShader->getStage()));

				return hash;
			}
		};

		using ParamsToDataMap = core::unordered_map<SIntrospectionParams,core::smart_refctd_ptr<const CIntrospectionData>, KeyHasher>;
		ParamsToDataMap m_introspectionCache;
};

} // nbl::asset

#endif
