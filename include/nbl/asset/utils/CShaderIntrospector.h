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
#include "nbl/asset/utils/IGLSLCompiler.h"


#include "nbl/core/definitions.h"


namespace spirv_cross
{
    class ParsedIR;
    class Compiler;
    struct SPIRType;
}

namespace nbl::asset
{

class CIntrospectionData : public core::IReferenceCounted
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

class CShaderIntrospector : public core::Uncopyable
{
		using mapId2SpecConst_t = core::unordered_map<uint32_t, const CIntrospectionData::SSpecConstant*>;

	public:
		struct SIntrospectionParams
		{
			ISpecializedShader::E_SHADER_STAGE stage;
			std::string entryPoint;
			core::smart_refctd_dynamic_array<std::string> GLSLextensions;
			std::filesystem::path filePathHint;

            inline bool operator<(const SIntrospectionParams& rhs) const
            {
                if (stage == rhs.stage)
                {
                    if (entryPoint == rhs.entryPoint)
                    {
                        const size_t mySz = GLSLextensions ? GLSLextensions->size() : 0ull;
                        const size_t rhsSz = rhs.GLSLextensions ? rhs.GLSLextensions->size() : 0ull;
                        if (mySz < rhsSz)
                            return true;
                        else if (mySz == rhsSz)
                        {
                            for (size_t i = 0ull; i < mySz; ++i)
                            {
                                const int cmpres = (*GLSLextensions)[i].compare((*rhs.GLSLextensions)[i]);
                                if (cmpres == 0)
                                    continue;
                                return (cmpres < 0);
                            }
							// if got out the loop, all extensions are equal
							return filePathHint<rhs.filePathHint;
                        }
                    }
                    return entryPoint < rhs.entryPoint;
                }
                return stage < rhs.stage;
            }
		};

		//In the future there's also going list of enabled extensions
		CShaderIntrospector(const IGLSLCompiler* _glslcomp) : m_glslCompiler(_glslcomp) {}

		//
		const CIntrospectionData* introspect(const ICPUShader* _shader, const SIntrospectionParams& _params);

		//
		std::pair<bool/*is shadow sampler*/, IImageView<ICPUImage>::E_TYPE> getImageInfoFromIntrospection(uint32_t set, uint32_t binding, ICPUSpecializedShader** const begin, ICPUSpecializedShader** const end, const std::string* _extensionsBegin, const std::string* _extensionsEnd);
		
		inline core::smart_refctd_dynamic_array<SPushConstantRange> createPushConstantRangesFromIntrospection(ICPUSpecializedShader** const begin, ICPUSpecializedShader** const end, const std::string* _extensionsBegin, const std::string* _extensionsEnd)
		{
			const CIntrospectionData* introspections[MAX_STAGE_COUNT] = { nullptr };
			if (!introspectAllShaders(introspections, begin, end, _extensionsBegin, _extensionsEnd))
				return nullptr;

			return createPushConstantRangesFromIntrospection_impl(introspections,introspections+std::distance(begin,end),begin);
		}
		inline core::smart_refctd_ptr<ICPUDescriptorSetLayout> createApproximateDescriptorSetLayoutFromIntrospection(uint32_t set, ICPUSpecializedShader** const begin, ICPUSpecializedShader** const end, const std::string* _extensionsBegin, const std::string* _extensionsEnd) {
			const CIntrospectionData* introspections[MAX_STAGE_COUNT] = { nullptr };
			if (!introspectAllShaders(introspections, begin, end, _extensionsBegin, _extensionsEnd))
				return nullptr;

			return createApproximateDescriptorSetLayoutFromIntrospection_impl(set, introspections, introspections+std::distance(begin,end), begin);
		}
		inline core::smart_refctd_ptr<ICPUPipelineLayout> createApproximatePipelineLayoutFromIntrospection(ICPUSpecializedShader** const begin, ICPUSpecializedShader** const end, const std::string* _extensionsBegin, const std::string* _extensionsEnd) {
			const CIntrospectionData* introspections[MAX_STAGE_COUNT] = { nullptr };
			if (!introspectAllShaders(introspections, begin, end, _extensionsBegin, _extensionsEnd))
				return nullptr;

			return createApproximatePipelineLayoutFromIntrospection_impl(introspections, introspections+std::distance(begin,end), begin);
		}

		//
		inline core::smart_refctd_ptr<ICPUComputePipeline> createApproximateComputePipelineFromIntrospection(ICPUSpecializedShader* shader, const std::string* _extensionsBegin, const std::string* _extensionsEnd)
		{
			if (shader->getStage() != ICPUSpecializedShader::ESS_COMPUTE)
				return nullptr;

			const CIntrospectionData* introspection = nullptr;
			if (!introspectAllShaders(&introspection, &shader, &shader+1, _extensionsBegin, _extensionsEnd))
				return nullptr;

			auto layout = createApproximatePipelineLayoutFromIntrospection_impl(&introspection,&introspection+1,&shader);
			return core::make_smart_refctd_ptr<ICPUComputePipeline>(
				std::move(layout),
				core::smart_refctd_ptr<ICPUSpecializedShader>(shader)
			);
		}

		//
		core::smart_refctd_ptr<ICPURenderpassIndependentPipeline> createApproximateRenderpassIndependentPipelineFromIntrospection(ICPUSpecializedShader** const begin, ICPUSpecializedShader** const end, const std::string* _extensionsBegin, const std::string* _extensionsEnd);
	
	private:
		core::smart_refctd_dynamic_array<SPushConstantRange> createPushConstantRangesFromIntrospection_impl(const CIntrospectionData** const begin, const CIntrospectionData** const end, const ICPUSpecializedShader* const* const shaders);
		core::smart_refctd_ptr<ICPUDescriptorSetLayout> createApproximateDescriptorSetLayoutFromIntrospection_impl(uint32_t _set, const CIntrospectionData** const begin, const CIntrospectionData** const end, const ICPUSpecializedShader* const* const shaders);
		core::smart_refctd_ptr<ICPUPipelineLayout> createApproximatePipelineLayoutFromIntrospection_impl(const CIntrospectionData** const begin, const CIntrospectionData** const end, const ICPUSpecializedShader* const* const shaders);

		_NBL_STATIC_INLINE_CONSTEXPR size_t MAX_STAGE_COUNT = 14ull;
		bool introspectAllShaders(const CIntrospectionData** introspection, ICPUSpecializedShader** const begin, ICPUSpecializedShader** const end, const std::string* _extensionsBegin, const std::string* _extensionsEnd);

		core::smart_refctd_ptr<CIntrospectionData> doIntrospection(spirv_cross::Compiler& _comp, const SIntrospectionParams& _ep) const;
		void shaderMemBlockIntrospection(spirv_cross::Compiler& _comp, impl::SShaderMemoryBlock& _res, uint32_t _blockBaseTypeID, uint32_t _varID, const mapId2SpecConst_t& _sortedId2sconst) const;
		size_t calcBytesizeforType(spirv_cross::Compiler& _comp, const spirv_cross::SPIRType& _type) const;

        core::smart_refctd_ptr<CIntrospectionData> findIntrospection(const ICPUShader* _shader, const SIntrospectionParams& _params) const
        {
            auto introspectionMap = m_introspectionCache.find(_params);
            if (introspectionMap == m_introspectionCache.end())
                return nullptr;

            auto introspection = introspectionMap->second.find(core::smart_refctd_ptr<const ICPUShader>(_shader));
            if (introspection == introspectionMap->second.end())
                return nullptr;

            return introspection->second;
        }
        CIntrospectionData* cacheIntrospection(core::smart_refctd_ptr<CIntrospectionData>&& _introspection, const ICPUShader* _shader, const SIntrospectionParams& _params)
        {
            return m_introspectionCache[_params].insert({core::smart_refctd_ptr<const ICPUShader>(_shader), std::move(_introspection)}).first->second.get();
        }

	private:
		const IGLSLCompiler* m_glslCompiler;

        using Shader2IntrospectionMap = core::unordered_map<core::smart_refctd_ptr<const ICPUShader>, core::smart_refctd_ptr<CIntrospectionData>>;
        using Params2ShaderMap = core::map<SIntrospectionParams, Shader2IntrospectionMap>;
        Params2ShaderMap m_introspectionCache;
};

} // nbl::asset

#endif
