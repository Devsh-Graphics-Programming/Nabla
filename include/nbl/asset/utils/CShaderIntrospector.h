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
        union
        {
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

    struct
    {
        bool present;
        SShaderPushConstant info;
    } pushConstant;

    bool canSpecializationlesslyCreateDescSetFrom() const
    {
        for(const auto& descSet : descriptorSetBindings)
        {
            auto found = std::find_if(descSet.begin(), descSet.end(), [](const SShaderResourceVariant& bnd) { return bnd.descCountIsSpecConstant; });
            if(found != descSet.end())
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
        const char* entryPoint;
        // if the Shader is already compiled to SPV, it will ignore this member
        core::SRange<const char* const> extraDefines;
    };

    //In the future there's also going list of enabled extensions
    CShaderIntrospector(const IGLSLCompiler* _glslcomp)
        : m_glslCompiler(_glslcomp) {}

    //
    const CIntrospectionData* introspect(const ICPUShader* _shader, const SIntrospectionParams& _params);

    //
    std::pair<bool /*is shadow sampler*/, IImageView<ICPUImage>::E_TYPE> getImageInfoFromIntrospection(uint32_t set, uint32_t binding, const core::SRange<const ICPUSpecializedShader* const>& _shaders, const core::SRange<const char* const>& _extraDefines);

    inline core::smart_refctd_dynamic_array<SPushConstantRange> createPushConstantRangesFromIntrospection(const core::SRange<const ICPUSpecializedShader* const>& _shaders, const core::SRange<const char* const>& _extraDefines)
    {
        const CIntrospectionData* introspections[MAX_STAGE_COUNT] = {nullptr};
        if(!introspectAllShaders(introspections, _shaders, _extraDefines))
            return nullptr;

        return createPushConstantRangesFromIntrospection_impl(introspections, _shaders);
    }
    inline core::smart_refctd_ptr<ICPUDescriptorSetLayout> createApproximateDescriptorSetLayoutFromIntrospection(uint32_t set, const core::SRange<const ICPUSpecializedShader* const>& _shaders, const core::SRange<const char* const>& _extraDefines)
    {
        const CIntrospectionData* introspections[MAX_STAGE_COUNT] = {nullptr};
        if(!introspectAllShaders(introspections, _shaders, _extraDefines))
            return nullptr;

        return createApproximateDescriptorSetLayoutFromIntrospection_impl(set, introspections, _shaders);
    }
    inline core::smart_refctd_ptr<ICPUPipelineLayout> createApproximatePipelineLayoutFromIntrospection(const core::SRange<const ICPUSpecializedShader* const>& _shaders, const core::SRange<const char* const>& _extraDefines)
    {
        const CIntrospectionData* introspections[MAX_STAGE_COUNT] = {nullptr};
        if(!introspectAllShaders(introspections, _shaders, _extraDefines))
            return nullptr;

        return createApproximatePipelineLayoutFromIntrospection_impl(introspections, _shaders);
    }

    //
    inline core::smart_refctd_ptr<ICPUComputePipeline> createApproximateComputePipelineFromIntrospection(ICPUSpecializedShader* shader, const core::SRange<const char* const>& _extraDefines)
    {
        if(shader->getStage() != IShader::ESS_COMPUTE)
            return nullptr;

        const core::SRange<const ICPUSpecializedShader* const> shaders = {&shader, &shader + 1};
        const CIntrospectionData* introspection = nullptr;
        if(!introspectAllShaders(&introspection, shaders, _extraDefines))
            return nullptr;

        auto layout = createApproximatePipelineLayoutFromIntrospection_impl(&introspection, shaders);
        return core::make_smart_refctd_ptr<ICPUComputePipeline>(
            std::move(layout),
            core::smart_refctd_ptr<ICPUSpecializedShader>(shader));
    }

    //
    core::smart_refctd_ptr<ICPURenderpassIndependentPipeline> createApproximateRenderpassIndependentPipelineFromIntrospection(const core::SRange<ICPUSpecializedShader* const>& _shaders, const core::SRange<const char* const>& _extraDefines);

private:
    core::smart_refctd_dynamic_array<SPushConstantRange> createPushConstantRangesFromIntrospection_impl(const CIntrospectionData** const introspections, const core::SRange<const ICPUSpecializedShader* const>& shaders);
    core::smart_refctd_ptr<ICPUDescriptorSetLayout> createApproximateDescriptorSetLayoutFromIntrospection_impl(uint32_t _set, const CIntrospectionData** const introspections, const core::SRange<const ICPUSpecializedShader* const>& shaders);
    core::smart_refctd_ptr<ICPUPipelineLayout> createApproximatePipelineLayoutFromIntrospection_impl(const CIntrospectionData** const introspections, const core::SRange<const ICPUSpecializedShader* const>& shaders);

    _NBL_STATIC_INLINE_CONSTEXPR size_t MAX_STAGE_COUNT = 14ull;
    bool introspectAllShaders(const CIntrospectionData** introspection, const core::SRange<const ICPUSpecializedShader* const>& _shaders, const core::SRange<const char* const>& _extraDefines);

    core::smart_refctd_ptr<CIntrospectionData> doIntrospection(spirv_cross::Compiler& _comp, const SIntrospectionParams& _ep, const IShader::E_SHADER_STAGE stage) const;
    void shaderMemBlockIntrospection(spirv_cross::Compiler& _comp, impl::SShaderMemoryBlock& _res, uint32_t _blockBaseTypeID, uint32_t _varID, const mapId2SpecConst_t& _sortedId2sconst) const;
    size_t calcBytesizeforType(spirv_cross::Compiler& _comp, const spirv_cross::SPIRType& _type) const;

private:
    const IGLSLCompiler* m_glslCompiler;

    struct Key
    {
        std::string entryPoint;
        core::smart_refctd_dynamic_array<std::string> extraDefines;
    };
    struct Comparator
    {
        using params_define_it = decltype(SIntrospectionParams::extraDefines)::const_iterator_type;
        using key_define_it = decltype(Key::extraDefines)::pointee::const_iterator;

    public:
        using is_transparent = std::true_type;

        inline bool operator()(const Key& lhs, const Key& rhs) const
        {
            if(lhs.entryPoint == rhs.entryPoint)
                less_than_extraDefines<key_define_it, key_define_it>(lhs.extraDefines->begin(), lhs.extraDefines->end(), rhs.extraDefines->begin(), rhs.extraDefines->end());
            return lhs.entryPoint < rhs.entryPoint;
        }
        inline bool operator()(const SIntrospectionParams& lhs, const Key& rhs) const
        {
            const auto cmp = strcmp(lhs.entryPoint, rhs.entryPoint.c_str());
            if(cmp == 0u)
                less_than_extraDefines<params_define_it, key_define_it>(lhs.extraDefines.begin(), lhs.extraDefines.end(), rhs.extraDefines->begin(), rhs.extraDefines->end());
            return cmp < 0;
        }
        inline bool operator()(const Key& lhs, const SIntrospectionParams& rhs) const
        {
            const auto cmp = strcmp(lhs.entryPoint.c_str(), rhs.entryPoint);
            if(cmp == 0u)
                less_than_extraDefines<key_define_it, params_define_it>(lhs.extraDefines->begin(), lhs.extraDefines->end(), rhs.extraDefines.begin(), rhs.extraDefines.end());
            return cmp < 0;
        }

    private:
        static inline int32_t flex_strcmp(const std::string& lhs, const std::string rhs)
        {
            return strcmp(lhs.c_str(), rhs.c_str());
        }
        static inline int32_t flex_strcmp(const char* lhs, const std::string rhs)
        {
            return strcmp(lhs, rhs.c_str());
        }
        static inline int32_t flex_strcmp(const std::string& lhs, const char* rhs)
        {
            return strcmp(lhs.c_str(), rhs);
        }

        template<typename LHSIterator, typename RHSIterator>
        static inline bool less_than_extraDefines(LHSIterator lhsBegin, LHSIterator lhsEnd, RHSIterator rhsBegin, RHSIterator rhsEnd)
        {
            const size_t lhsSz = std::distance(lhsBegin, lhsEnd);
            const size_t rhsSz = std::distance(rhsBegin, rhsEnd);
            if(lhsSz == rhsSz)
            {
                auto rhsIt = rhsBegin;
                for(auto lhsIt = lhsBegin; lhsIt != lhsEnd; lhsIt++, rhsIt++)
                {
                    const int cmpres = flex_strcmp(*lhsIt, *rhsIt);
                    if(cmpres == 0)
                        continue;
                    return cmpres < 0;
                }
                // if got out the loop, all extensions are equal
                return false;
            }
            return lhsSz < rhsSz;
        }
    };
    using Shader2IntrospectionMap = core::unordered_map<core::smart_refctd_ptr<const ICPUShader>, core::smart_refctd_ptr<CIntrospectionData>>;
    using Params2ShaderMap = core::map<Key, Shader2IntrospectionMap, Comparator>;
    Params2ShaderMap m_introspectionCache;
};

}  // nbl::asset

#endif
