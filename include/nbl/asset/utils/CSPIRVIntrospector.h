// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_SHADER_INTROSPECTOR_H_INCLUDED__
#define __NBL_ASSET_C_SHADER_INTROSPECTOR_H_INCLUDED__

#include "nbl/core/declarations.h"

#include <cstdint>
#include <memory>

#include "nbl/asset/ICPUSpecializedShader.h"
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

class NBL_API CIntrospectionData : public core::IReferenceCounted
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

class NBL_API CSPIRVIntrospector : public core::Uncopyable
{
		using mapId2SpecConst_t = core::unordered_map<uint32_t, const CIntrospectionData::SSpecConstant*>;

	public:
		struct SIntrospectionParams
		{
			std::string entryPoint;
			core::smart_refctd_ptr<const ICPUShader> cpuShader;

			bool operator==(const SIntrospectionParams& rhs) const { return false; /*TODO*/ }
		};

		//In the future there's also going list of enabled extensions
		CSPIRVIntrospector() = default;

		//! params.cpuShader.contentType should be ECT_SPIRV
		//! the compiled SPIRV must be compiled with IShaderCompiler::SCompilerOptions::genDebugInfo in order to include names in introspection data
		const core::smart_refctd_ptr<CIntrospectionData> introspect(const SIntrospectionParams& params, bool insertToCache = true);

		//
		std::pair<bool/*is shadow sampler*/, IImageView<ICPUImage>::E_TYPE> getImageInfoFromIntrospection(uint32_t set, uint32_t binding, const core::SRange<const ICPUSpecializedShader* const>& _shaders);
		
		inline core::smart_refctd_dynamic_array<SPushConstantRange> createPushConstantRangesFromIntrospection(const core::SRange<const ICPUSpecializedShader* const>& _shaders)
		{
			core::smart_refctd_ptr<CIntrospectionData> introspections[MAX_STAGE_COUNT] = { nullptr };
			if (!introspectAllShaders(introspections,_shaders))
				return nullptr;

			return createPushConstantRangesFromIntrospection_impl(introspections,_shaders);
		}
		inline core::smart_refctd_ptr<ICPUDescriptorSetLayout> createApproximateDescriptorSetLayoutFromIntrospection(uint32_t set, const core::SRange<const ICPUSpecializedShader* const>& _shaders)
		{
			core::smart_refctd_ptr<CIntrospectionData> introspections[MAX_STAGE_COUNT] = { nullptr };
			if (!introspectAllShaders(introspections,_shaders))
				return nullptr;

			return createApproximateDescriptorSetLayoutFromIntrospection_impl(set,introspections,_shaders);
		}
		inline core::smart_refctd_ptr<ICPUPipelineLayout> createApproximatePipelineLayoutFromIntrospection(const core::SRange<const ICPUSpecializedShader* const>& _shaders)
		{
			core::smart_refctd_ptr<CIntrospectionData> introspections[MAX_STAGE_COUNT] = { nullptr };
			if (!introspectAllShaders(introspections,_shaders))
				return nullptr;

			return createApproximatePipelineLayoutFromIntrospection_impl(introspections,_shaders);
		}

		//
		inline core::smart_refctd_ptr<ICPUComputePipeline> createApproximateComputePipelineFromIntrospection(ICPUSpecializedShader* shader)
		{
			if (shader->getStage() != IShader::ESS_COMPUTE)
				return nullptr;

			const core::SRange<const ICPUSpecializedShader* const> shaders = {&shader,&shader+1};
			core::smart_refctd_ptr<CIntrospectionData> introspection = nullptr;
			if (!introspectAllShaders(&introspection,shaders))
				return nullptr;

			auto layout = createApproximatePipelineLayoutFromIntrospection_impl(&introspection,shaders);
			return core::make_smart_refctd_ptr<ICPUComputePipeline>(
				std::move(layout),
				core::smart_refctd_ptr<ICPUSpecializedShader>(shader)
			);
		}

		//
		core::smart_refctd_ptr<ICPURenderpassIndependentPipeline> createApproximateRenderpassIndependentPipelineFromIntrospection(const core::SRange<ICPUSpecializedShader* const>& _shaders);
	
	private:
		core::smart_refctd_dynamic_array<SPushConstantRange> createPushConstantRangesFromIntrospection_impl(core::smart_refctd_ptr<CIntrospectionData>* const introspections, const core::SRange<const ICPUSpecializedShader* const>& shaders);
		core::smart_refctd_ptr<ICPUDescriptorSetLayout> createApproximateDescriptorSetLayoutFromIntrospection_impl(uint32_t _set, core::smart_refctd_ptr<CIntrospectionData>* const introspections, const core::SRange<const ICPUSpecializedShader* const>& shaders);
		core::smart_refctd_ptr<ICPUPipelineLayout> createApproximatePipelineLayoutFromIntrospection_impl(core::smart_refctd_ptr<CIntrospectionData>* const introspections, const core::SRange<const ICPUSpecializedShader* const>& shaders);

		_NBL_STATIC_INLINE_CONSTEXPR size_t MAX_STAGE_COUNT = 14ull;
		bool introspectAllShaders(core::smart_refctd_ptr<CIntrospectionData>* introspection, const core::SRange<const ICPUSpecializedShader* const>& _shaders);

		core::smart_refctd_ptr<CIntrospectionData> doIntrospection(spirv_cross::Compiler& _comp, const std::string& entryPoint, const IShader::E_SHADER_STAGE stage) const;
		void shaderMemBlockIntrospection(spirv_cross::Compiler& _comp, impl::SShaderMemoryBlock& _res, uint32_t _blockBaseTypeID, uint32_t _varID, const mapId2SpecConst_t& _sortedId2sconst) const;
		size_t calcBytesizeforType(spirv_cross::Compiler& _comp, const spirv_cross::SPIRType& _type) const;

	private:

		struct KeyHasher
		{
			std::size_t operator()(const SIntrospectionParams& t) const { return 0; /*TODO*/ }
		};

		using ParamsToDataMap = core::unordered_map<SIntrospectionParams,core::smart_refctd_ptr<CIntrospectionData>, KeyHasher>;
		ParamsToDataMap m_introspectionCache;
};

} // nbl::asset

#endif
