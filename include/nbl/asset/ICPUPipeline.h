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

            using SSpecConstantValue = core::vector<uint8_t>;

            inline SSpecConstantValue* getSpecializationByteValue(const spec_constant_id_t _specConstID)
            {
                const auto found = entries.find(_specConstID);
                if (found != entries.end() && found->second.size()) return &found->second;
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
                    if (!entry.second.size()) return INVALID_SPEC_INFO;
                    specData += entry.second.size();
                }
                if (specData > 0x7fffffff) return INVALID_SPEC_INFO;
                return static_cast<int32_t>(specData);
            }

            core::smart_refctd_ptr<IShader> shader = nullptr;
            std::string entryPoint = "";

            IPipelineBase::SUBGROUP_SIZE requiredSubgroupSize = IPipelineBase::SUBGROUP_SIZE::UNKNOWN;	//!< Default value of 8 means no requirement

            using spec_constant_map_t = core::unordered_map<spec_constant_id_t, SSpecConstantValue>;
            // Container choice implicitly satisfies:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSpecializationInfo.html#VUID-VkSpecializationInfo-constantID-04911
            spec_constant_map_t entries;
            // By requiring Nabla Core Profile features we implicitly satisfy:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-flags-02784
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-flags-02785
            // Also because our API is sane, it satisfies the following by construction:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-pNext-02754

            SShaderSpecInfo clone(uint32_t depth) const
            {
                auto newSpecInfo = *this;
                if (newSpecInfo.shader.get() != nullptr && depth > 0u)
                {
                    newSpecInfo.shader = core::smart_refctd_ptr_static_cast<IShader>(this->shader->clone(depth - 1u));
                }
                return newSpecInfo;
            }
        };

        virtual std::span<const SShaderSpecInfo> getSpecInfos(const hlsl::ShaderStage stage) const = 0;

};

// Common Base class for pipelines
template<typename PipelineNonAssetBase>
    requires (std::is_base_of_v<IPipeline<ICPUPipelineLayout>, PipelineNonAssetBase> && !std::is_base_of_v<IAsset, PipelineNonAssetBase>)
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
              layout = core::smart_refctd_ptr_static_cast<ICPUPipelineLayout>(getLayout()->clone(_depth - 1u));

            return clone_impl(std::move(layout), _depth);
        }

        // Note(kevinyu): For some reason overload resolution cannot find this function when I name id getSpecInfos. It always use the const variant. Will check on it later.
        inline std::span<SShaderSpecInfo> getSpecInfos(const hlsl::ShaderStage stage)
        {
            if (!isMutable()) return {};
            const this_t* constPipeline = const_cast<const this_t*>(this);
            const ICPUPipelineBase* basePipeline = constPipeline;
            const auto specInfo = basePipeline->getSpecInfos(stage);
            return { const_cast<SShaderSpecInfo*>(specInfo.data()), specInfo.size() };
        }

    protected:

        using PipelineNonAssetBase::PipelineNonAssetBase;
        virtual ~ICPUPipeline() = default;
        
        virtual core::smart_refctd_ptr<this_t> clone_impl(core::smart_refctd_ptr<ICPUPipelineLayout>&& layout, uint32_t depth) const = 0;

};

}
#endif