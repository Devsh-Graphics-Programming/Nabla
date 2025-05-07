// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_CPU_PIPELINE_H_INCLUDED_


#include "nbl/asset/IAsset.h"
#include "nbl/asset/IPipeline.h"
#include "nbl/asset/ICPUPipelineLayout.h"


namespace nbl::asset
{

class ICPUPipelineBase
{
    public:
        struct SShaderSpecInfo
        {
            //! Structure specifying a specialization map entry
            /*
              Note that if specialization constant ID is used
              in a shader, \bsize\b and \boffset'b must match
              to \isuch an ID\i accordingly.

              By design the API satisfies:
              https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSpecializationInfo.html#VUID-VkSpecializationInfo-offset-00773
              https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSpecializationInfo.html#VUID-VkSpecializationInfo-pMapEntries-00774
            */
            //!< The ID of the specialization constant in SPIR-V. If it isn't used in the shader, the map entry does not affect the behavior of the pipeline.
            using spec_constant_id_t = uint32_t;

            struct SSpecConstantValue
            {
                core::vector<uint8_t> data;
                inline operator bool() const { return data.size(); }
                inline size_t size() const { return data.size(); }
            };

            inline SSpecConstantValue* getSpecializationByteValue(const spec_constant_id_t _specConstID)
            {
                const auto found = entries.find(_specConstID);
                if (found != entries.end() && bool(found->second)) return &found->second;
                else return nullptr;
            }

            static constexpr int32_t INVALID_SPEC_INFO = -1;
            inline int32_t valid() const
            {
                if (!shader) return INVALID_SPEC_INFO;

                // Impossible to check: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-pName-00707
                if (entryPoint.empty()) return INVALID_SPEC_INFO;

                // Impossible to efficiently check anything from:
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-maxClipDistances-00708
                // to:
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-stage-06686
                // and from:
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-pNext-02756
                // to:
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-module-08987

                int64_t specData = 0;
                for (const auto& entry : entries)
                {
                    if (!entry.second) return INVALID_SPEC_INFO;
                    specData += entry.second.size();
                }
                if (specData > 0x7fffffff) return INVALID_SPEC_INFO;
                return static_cast<int32_t>(specData);
            }

            core::smart_refctd_ptr<IShader> shader = nullptr;
            std::string entryPoint = "";
            IPipelineBase::SUBGROUP_SIZE requiredSubgroupSize : 3 = IPipelineBase::SUBGROUP_SIZE::UNKNOWN;	//!< Default value of 8 means no requirement
            uint8_t requireFullSubgroups : 1 = false;

            // Container choice implicitly satisfies:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSpecializationInfo.html#VUID-VkSpecializationInfo-constantID-04911
            core::unordered_map<spec_constant_id_t, SSpecConstantValue> entries;
            // By requiring Nabla Core Profile features we implicitly satisfy:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-flags-02784
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-flags-02785
            // Also because our API is sane, it satisfies the following by construction:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-pNext-02754

        };

        virtual std::span<const SShaderSpecInfo> getSpecInfo(const hlsl::ShaderStage stage) const = 0;

        virtual bool valid() const = 0;
};

// Common Base class for pipelines
template<typename PipelineNonAssetBase>
class ICPUPipeline : public IAsset, public PipelineNonAssetBase, public ICPUPipelineBase
{
        using this_t = ICPUPipeline<PipelineNonAssetBase>;

    public:

        // extras for this class
        ICPUPipelineLayout* getLayout() 
        {
            assert(isMutable());
            return const_cast<ICPUPipelineLayout*>(PipelineNonAssetBase::m_layout.get());
        }
        const ICPUPipelineLayout* getLayout() const { return PipelineNonAssetBase::m_layout.get(); }

        inline void setLayout(core::smart_refctd_ptr<const ICPUPipelineLayout>&& _layout)
        {
            assert(isMutable());
            PipelineNonAssetBase::m_layout = std::move(_layout);
        }

        inline core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override final
        {
            if (!getLayout()) return nullptr;

            core::smart_refctd_ptr<ICPUPipelineLayout> layout;
            if (_depth > 0u) 
              layout = core::smart_refctd_ptr_static_cast<ICPUPipelineLayout>(getLayout->clone(_depth-1u));

            return clone_impl(std::move(layout), _depth);
        }

        SShaderSpecInfo cloneSpecInfo(const SShaderSpecInfo& specInfo, uint32_t depth)
        inline std::span<SShaderSpecInfo> getSpecInfo(hlsl::ShaderStage stage)
        {
            if (!isMutable()) return {};
            const auto specInfo = static_cast<const this_t*>(this)->getSpecInfo(stage);
            return { const_cast<SShaderSpecInfo*>(specInfo.data()), specInfo.size() };
        }

    protected:

        using PipelineNonAssetBase::PipelineNonAssetBase;
        virtual ~ICPUPipeline() = default;
        
        virtual core::smart_refctd_ptr<this_t> clone_impl(core::smart_refctd_ptr<const ICPUPipelineLayout>&& layout, uint32_t depth) const = 0;

};

}
#endif