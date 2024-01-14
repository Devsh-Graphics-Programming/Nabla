#ifndef _NBL_C_VULKAN_PHYSICAL_DEVICE_H_INCLUDED_
#define _NBL_C_VULKAN_PHYSICAL_DEVICE_H_INCLUDED_

#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/CVulkanLogicalDevice.h"
#include "nbl/video/surface/CSurfaceVulkan.h"
#include "nbl/video/CVulkanCommon.h"

namespace nbl::video
{

class CVulkanPhysicalDevice final : public IPhysicalDevice
{
    public:
        static std::unique_ptr<CVulkanPhysicalDevice> create(core::smart_refctd_ptr<system::ISystem>&& sys, IAPIConnection* const api, renderdoc_api_t* const rdoc, const VkPhysicalDevice vk_physicalDevice);
            
        inline VkPhysicalDevice getInternalObject() const { return m_vkPhysicalDevice; }
            
        inline E_API_TYPE getAPIType() const override { return EAT_VULKAN; }

    protected:
        inline CVulkanPhysicalDevice(IPhysicalDevice::SInitData&& _initData, renderdoc_api_t* const rdoc, const VkPhysicalDevice vk_physicalDevice, core::unordered_set<std::string>&& _extensions)
            : IPhysicalDevice(std::move(_initData)), m_rdoc_api(rdoc), m_vkPhysicalDevice(vk_physicalDevice), m_extensions(std::move(_extensions)) {}
    
        //! This function makes sure requirements of a requested feature is also set to `true` in SPhysicalDeviceFeatures
        //! Note that this will only fix what is exposed, some may require extensions not exposed currently, that will happen later on.
        inline void resolveFeatureDependencies(SFeatures& features) const
        {
            // There is metadata in vk.xml describing some aspects of promotion, especially requires, promotedto and deprecatedby attributes of <extension> tags.
            // However, the metadata does not yet fully describe this scenario.
            // In the future, we may extend the XML schema to describe the full set of extensions and versions satisfying a dependency

            if (features.inheritedConditionalRendering)
                features.conditionalRendering = true;

            if (features.geometryShaderPassthrough)
                features.geometryShader = true;

            // VK_EXT_hdr_metadata Requires VK_KHR_swapchain to be enabled
            if (features.hdrMetadata)
            {
                features.swapchainMode |= E_SWAPCHAIN_MODE::ESM_SURFACE;
                // And VK_KHR_swapchain requires VK_KHR_surface instance extension
            }

            if (features.performanceCounterMultipleQueryPools)
                features.performanceCounterQueryPools = true;

            // `VK_EXT_shader_atomic_float2` Requires `VK_EXT_shader_atomic_float`:
            // this dependancy needs the extension to be enabled not individual features,
            // so this will be handled later on when enabling features before vkCreateDevice

            //! these RT/BVH features are in a very specific order!
            if (features.rayTracingMotionBlur || features.rayTracingMotionBlurPipelineTraceRaysIndirect)
            {
                features.rayTracingMotionBlur = true;
                features.rayTracingPipeline = true;
            }

            if (features.rayTraversalPrimitiveCulling)
            {
                features.rayQuery = true;
                features.rayTracingPipeline = true; // correct?
            }

            if (features.rayTracingPipeline || features.rayQuery)
            {
                features.accelerationStructure = true;
            }

            if (features.accelerationStructure ||
                features.accelerationStructureIndirectBuild ||
                features.accelerationStructureHostCommands)
            {
                features.accelerationStructure = true;
                // make VK_KHR_deferred_host_operations an interaction instead of a required extension (later went back on this)
                // VK_KHR_deferred_host_operations is required again
                features.deferredHostOperations = true;
            }

            // `VK_EXT_fragment_density_map2` Requires `FragmentDensityMapFeaturesEXT`
            if (features.fragmentDensityMapDeferred)
            {
                features.fragmentDensityMap = true;
            }
        
            if (features.fragmentDensityMapDynamic || features.fragmentDensityMapNonSubsampledImages)
            {
                // make sure features have their main bool enabled!
                features.fragmentDensityMap = true;
            }

            if (features.fragmentDensityMap)
            {
                // If the fragmentDensityMap feature is enabled, the pipelineFragmentShadingRate feature must not be enabled
                // If the fragmentDensityMap feature is enabled, the primitiveFragmentShadingRate feature must not be enabled
                // If the fragmentDensityMap feature is enabled, the attachmentFragmentShadingRate feature must not be enabled
            }

            // If the shadingRateImage feature is enabled, the pipelineFragmentShadingRate feature must not be enabled
            // If the shadingRateImage feature is enabled, the primitiveFragmentShadingRate feature must not be enabled
            // If the shadingRateImage feature is enabled, the attachmentFragmentShadingRate feature must not be enabled
        
            // Handle later: E_SWAPCHAIN_MODE::ESM_SURFACE: VK_KHR_swapchain requires VK_KHR_surface instance extension
            
            // [NOOP] If sparseImageInt64Atomics is enabled, shaderImageInt64Atomics must be enabled
            // [NOOP] If sparseImageFloat32Atomics is enabled, shaderImageFloat32Atomics must be enabled
            // [NOOP] If sparseImageFloat32AtomicAdd is enabled, shaderImageFloat32AtomicAdd must be enabled
            // [NOOP] If sparseImageFloat32AtomicMinMax is enabled, shaderImageFloat32AtomicMinMax must be enabled
        }

        inline static SExternalMemoryProperties mapExternalMemoryProps(VkExternalMemoryProperties const& props)
        {
            return {
                .exportableTypes = props.exportFromImportedHandleTypes,
                .compatibleTypes = props.compatibleHandleTypes,
                .dedicatedOnly = props.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT ? 1u : 0u,
                .exportable = props.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT ? 1u : 0u,
                .importable = props.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT ? 1u : 0u,
            };
        }

        SExternalMemoryProperties getExternalBufferProperties_impl(core::bitflag<IGPUBuffer::E_USAGE_FLAGS> usage, IDeviceMemoryAllocation::E_EXTERNAL_HANDLE_TYPE handleType) const override
        {
            assert(!(handleType & (handleType - 1)));
            VkPhysicalDeviceExternalBufferInfo info = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
                .usage = static_cast<VkBufferUsageFlags>(usage.value),
                .handleType = static_cast<VkExternalMemoryHandleTypeFlagBits>(handleType)
            };
            VkExternalBufferProperties externalProps = { VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES };
            vkGetPhysicalDeviceExternalBufferProperties(m_vkPhysicalDevice, &info, &externalProps);
            return mapExternalMemoryProps(externalProps.externalMemoryProperties);
        }

        SExternalImageFormatProperties getExternalImageProperties_impl(
            asset::E_FORMAT format, 
            IGPUImage::TILING tiling, 
            IGPUImage::E_TYPE type,
            core::bitflag<IGPUImage::E_USAGE_FLAGS> usage, 
            core::bitflag<IGPUImage::E_CREATE_FLAGS> flags,  
            IDeviceMemoryAllocation::E_EXTERNAL_HANDLE_TYPE handleType) const override
        {
            assert(!(handleType & (handleType - 1)));

            VkPhysicalDeviceExternalImageFormatInfo extInfo = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
                .handleType = static_cast<VkExternalMemoryHandleTypeFlagBits>(handleType),
            };

            VkPhysicalDeviceImageFormatInfo2 info = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
                .pNext = &extInfo,
                .format = getVkFormatFromFormat(format),
                .type  = static_cast<VkImageType>(type),
                .tiling = static_cast<VkImageTiling>(tiling),
                .usage = usage.value,
                .flags = flags.value,
            };

            VkExternalImageFormatProperties externalProps = { VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES };

            VkImageFormatProperties2 props = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
                .pNext = &externalProps,
            };

            VkResult re = vkGetPhysicalDeviceImageFormatProperties2(m_vkPhysicalDevice, &info, &props);
            if(VK_SUCCESS != re)
                return {};

            return 
                { 
                    {
                        .maxExtent = props.imageFormatProperties.maxExtent,
                        .maxMipLevels = props.imageFormatProperties.maxMipLevels,
                        .maxArrayLayers = props.imageFormatProperties.maxArrayLayers,
                        .sampleCounts = static_cast<IGPUImage::E_SAMPLE_COUNT_FLAGS>(props.imageFormatProperties.sampleCounts),
                        .maxResourceSize = props.imageFormatProperties.maxResourceSize,
                    }, 
                    mapExternalMemoryProps(externalProps.externalMemoryProperties) 
                };
        }

        core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice_impl(ILogicalDevice::SCreationParams&& params) override;

    private:
        renderdoc_api_t* const m_rdoc_api;
        const VkPhysicalDevice m_vkPhysicalDevice;

        const core::unordered_set<std::string> m_extensions;
};

}

#endif
