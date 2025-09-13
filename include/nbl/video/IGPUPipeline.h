

// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_GPU_PIPELINE_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_PIPELINE_H_INCLUDED_

#include "nbl/video/IGPUPipelineLayout.h"
#include "nbl/video/SPipelineCreationParams.h"
#include "nbl/asset/ICPUPipeline.h"
#include "nbl/asset/IPipeline.h"

namespace nbl::video
{

class IGPUPipelineBase {
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

            using SSpecConstantValue = std::span<const uint8_t>;

            inline SSpecConstantValue getSpecializationByteValue(const spec_constant_id_t _specConstID) const
            {
                if (!entries) return {};

                const auto found = entries->find(_specConstID);
                if (found != entries->end() && found->second.size()) return found->second;
                else return {};
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
                if (entries)
                {
                    for (const auto& entry : *entries)
                    {
                      if (!entry.second.size())
                          return INVALID_SPEC_INFO;
                      specData += entry.second.size();
                    }
                }
                if (specData>0x7fffffff)
                    return INVALID_SPEC_INFO;
                return static_cast<int32_t>(specData);
            }

            inline bool accumulateSpecializationValidationResult(SSpecializationValidationResult* retval) const
            {
                const auto dataSize = valid();
                if (dataSize < 0)
                    return false;
                if (dataSize == 0)
                    return true;

                const size_t count = entries ? entries->size() : 0x80000000ull;
                if (count > 0x7fffffff)
                    return false;
                *retval += {
                    .count = dataSize ? static_cast<uint32_t>(count) : 0,
                    .dataSize = static_cast<uint32_t>(dataSize),
                };
                return *retval;
            }

            const asset::IShader* shader = nullptr;
            std::string_view entryPoint = "";

            asset::IPipelineBase::SUBGROUP_SIZE requiredSubgroupSize = asset::IPipelineBase::SUBGROUP_SIZE::UNKNOWN;	//!< Default value of 8 means no requirement

            // Container choice implicitly satisfies:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSpecializationInfo.html#VUID-VkSpecializationInfo-constantID-04911
            using entry_map_t = core::unordered_map<spec_constant_id_t, SSpecConstantValue>;
            const entry_map_t* entries;
            // By requiring Nabla Core Profile features we implicitly satisfy:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-flags-02784
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-flags-02785
            // Also because our API is sane, it satisfies the following by construction:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-pNext-02754


            static inline SShaderSpecInfo create(const asset::ICPUPipelineBase::SShaderSpecInfo& cpuSpecInfo, entry_map_t* outEntries)  
            {
                SShaderSpecInfo specInfo;
                specInfo.shader = cpuSpecInfo.shader.get();
                specInfo.entryPoint = cpuSpecInfo.entryPoint;
                specInfo.requiredSubgroupSize = cpuSpecInfo.requiredSubgroupSize;
                outEntries->clear();
                for (const auto&[key, value] : cpuSpecInfo.entries)
                {
                    outEntries->insert({ key, { value.data(), value.size() } });
                }
                specInfo.entries = outEntries;
                return specInfo;
            };
        };

        using SShaderEntryMap = SShaderSpecInfo::entry_map_t;

};

// Common Base class for pipelines
template<typename PipelineNonBackendObjectBase>
    requires (std::is_base_of_v<asset::IPipeline<const IGPUPipelineLayout>, PipelineNonBackendObjectBase> && !std::is_base_of_v<IBackendObject, PipelineNonBackendObjectBase>)
class IGPUPipeline : public IBackendObject, public PipelineNonBackendObjectBase, public IGPUPipelineBase
{
    protected:

        template <typename... Args>
        explicit IGPUPipeline(core::smart_refctd_ptr<const ILogicalDevice>&& device, Args&&... args) :
         PipelineNonBackendObjectBase(std::forward<Args>(args)...), IBackendObject(std::move(device))
        {}
        virtual ~IGPUPipeline() = default;

};

}

#endif
