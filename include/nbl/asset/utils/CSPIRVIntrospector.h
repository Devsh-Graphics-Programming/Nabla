// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_SPIRV_INTROSPECTOR_H_INCLUDED_
#define _NBL_ASSET_C_SPIRV_INTROSPECTOR_H_INCLUDED_
#if 0
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

// podzielic CIntrospectionData na dwie klasy
// jedna bez inputOutput i bez push constant blocka `CIntrospectionData`
// druga dziedziczy z pierwszej i dodaje te 2 rzeczy `CStageIntrospectionData`

// wszystkie struktury w CIntrospecionData powininny u¿ywaæ bit flagi, ozaczaj¹cej shader stage (core::unordered_map)
// CStageIntrospecionData nie powinien u¿ywaæ bit flagi, ozaczaj¹cej shader stage (core::vector)

// hashowane s¹ tylko set i binding
// dla spec constant tylko specConstantID
// validacja kolizji (dla SSpecConstants mo¿e siê jedynie ró¿niæ name)
// ogarn¹æ sytuacje gdy jeden descriptor binding ma wiêcej arrayElementCount ni¿ w SPIR-V
// w `CStageIntrospectionData` powinien byæ trzymana struktura `SIntrospectionParams`

// 
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

			//! Push constants uniform block
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
				return memcmp(cpuShader->getContent()->getPointer(), rhs.cpuShader->getContent()->getPointer(), cpuShader->getContent()->getSize()) == 0;
			}
		};

		//In the future there's also going list of enabled extensions
		CSPIRVIntrospector() = default;

		//! params.cpuShader.contentType should be ECT_SPIRV
		//! the compiled SPIRV must be compiled with IShaderCompiler::SCompilerOptions::debugInfoFlags enabling EDIF_SOURCE_BIT implicitly or explicitly, with no `spirvOptimizer` used in order to include names in introspection data
		// powinna zwracac CStageIntrospectionData
		core::smart_refctd_ptr<const CIntrospectionData> introspect(const SIntrospectionParams& params, bool insertToCache = true);

		// 
		//core::smart_refctd_ptr<const CIntrospectionData> merge(const std::span<const CStageIntrospectionData>& asdf, const ICPUShader::SSPecInfo::spec_constant_map_t& = {});

		// When the methods take a span of shaders, they are computing things for an imaginary pipeline that includes **all** of them
		// przeniesc do CIntrospectionData
		std::pair<bool/*is shadow sampler*/, IImageView<ICPUImage>::E_TYPE> getImageInfoFromIntrospection(uint32_t set, uint32_t binding, const std::span<const ICPUShader::SSpecInfo> _infos);

		//
		inline core::smart_refctd_ptr<ICPUComputePipeline> createApproximateComputePipelineFromIntrospection(const ICPUShader::SSpecInfo& info)
		//TODO: inline core::smart_refctd_ptr<ICPUComputePipeline> createApproximateComputePipelineFromIntrospection(CStageIntrospectionData* asdf)
		{
			if (info.shader->getStage()!=IShader::ESS_COMPUTE)
				return nullptr;

			core::smart_refctd_ptr<const CIntrospectionData> introspection = nullptr;
			
			//TODO: zamiast tego mergujemy `CStageIntrospectionData` w `CIntrospectionData` u¿ywaj¹c `merge`
			if (!introspectAllShaders(&introspection,{&info,1}))
				return nullptr;

			auto layout = createApproximatePipelineLayoutFromIntrospection_impl(&introspection,{&info,1});
			ICPUComputePipeline::SCreationParams params = {{.layout = layout.get()}};
			params.shader = info;
			return ICPUComputePipeline::create(params);
		}

		//
		core::smart_refctd_ptr<ICPURenderpassIndependentPipeline> createApproximateRenderpassIndependentPipelineFromIntrospection(const std::span<const ICPUShader::SSpecInfo> _infos);

		struct CShaderStages
		{
			const CStageIntrospectionData* vertex = nullptr;
			const CStageIntrospectionData* fragment = nullptr;
			const CStageIntrospectionData* control = nullptr;
			const CStageIntrospectionData* evaluation = nullptr;
			const CStageIntrospectionData* geometry = nullptr;
		}
		core::smart_refctd_ptr<ICPUGraphicsPipeline> createApproximateGraphicsPipeline(const CShaderStages& shaderStages);
	
	private:
		//TODO: przenieœæ jako members do CIntrospectionData
		core::smart_refctd_dynamic_array<SPushConstantRange> createPushConstantRangesFromIntrospection_impl();
		core::smart_refctd_ptr<ICPUDescriptorSetLayout> createApproximateDescriptorSetLayoutFromIntrospection_impl(const uint32_t setID);
		core::smart_refctd_ptr<ICPUPipelineLayout> createApproximatePipelineLayoutFromIntrospection_impl();

		core::smart_refctd_ptr<CStageIntrospectionData> introspectShader(const ICPUShader::SSpecInfo _infos);

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
		// using ParamsToDataMap = core::unordered_set<core::smart_refctd_ptr<const CStageIntrospectionData>, KeyHasher, KeyEquals>;
		ParamsToDataMap m_introspectionCache;
};

} // nbl::asset

#endif
#endif
