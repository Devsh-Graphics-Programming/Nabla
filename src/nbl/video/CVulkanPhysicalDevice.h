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
        static inline std::unique_ptr<CVulkanPhysicalDevice> create(core::smart_refctd_ptr<system::ISystem>&& sys, IAPIConnection* const api, renderdoc_api_t* const rdoc, const VkPhysicalDevice vk_physicalDevice);
            
        inline VkPhysicalDevice getInternalObject() const { return m_vkPhysicalDevice; }
            
        inline E_API_TYPE getAPIType() const override { return EAT_VULKAN; }

        inline IDebugCallback* getDebugCallback() const override { return m_api->getDebugCallback(); }

    protected:
        inline CVulkanPhysicalDevice(core::smart_refctd_ptr<system::ISystem>&& sys, IAPIConnection* const api, renderdoc_api_t* const rdoc, const VkPhysicalDevice vk_physicalDevice)
            : IPhysicalDevice(std::move(sys),api), m_rdoc_api(rdoc), m_vkPhysicalDevice(vk_physicalDevice) {}
    
        //! This function makes sure requirements of a requested feature is also set to `true` in SPhysicalDeviceFeatures
        //! Note that this will only fix what is exposed, some may require extensions not exposed currently, that will happen later on.
        inline void resolveFeatureDependencies(SFeatures& features) const
        {
            // `VK_EXT_shader_atomic_float2` Requires `VK_EXT_shader_atomic_float`: this dependancy needs the extension to be enabled not individual features, so this will be handled later on when enabling features before vkCreateDevice
            if (features.rayTracingMotionBlur ||
                features.rayTracingMotionBlurPipelineTraceRaysIndirect)
            {
                features.rayTracingMotionBlur = true;
                features.rayTracingPipeline = true;
            }

            if (features.rayTracingPipeline ||
                features.rayTracingPipelineTraceRaysIndirect ||
                features.rayTraversalPrimitiveCulling)
            {
                features.rayTracingPipeline = true;
                features.accelerationStructure = true;
            }

            if (features.rayQuery)
            {
                features.accelerationStructure = true;
            }

            if (features.accelerationStructure ||
                features.accelerationStructureIndirectBuild ||
                features.accelerationStructureHostCommands ||
                features.descriptorBindingAccelerationStructureUpdateAfterBind)
            {
                features.accelerationStructure = true;
                features.descriptorIndexing = true;
                features.bufferDeviceAddress = true;
                features.deferredHostOperations = true;
            }

            // VK_NV_coverage_reduction_mode requires VK_NV_framebuffer_mixed_samples
            if (features.coverageReductionMode)
                features.mixedAttachmentSamples = true;

            if (features.deviceGeneratedCommands)
                features.bufferDeviceAddress = true;
        
            if (features.bufferDeviceAddressMultiDevice)
                features.bufferDeviceAddress = true; // make sure features have their main bool enabled

            // TODO: review
            if (features.shaderInputAttachmentArrayDynamicIndexing ||
                features.shaderUniformTexelBufferArrayDynamicIndexing ||
                features.shaderStorageTexelBufferArrayDynamicIndexing ||
                features.shaderUniformBufferArrayNonUniformIndexing ||
                features.shaderSampledImageArrayNonUniformIndexing ||
                features.shaderStorageBufferArrayNonUniformIndexing ||
                features.shaderStorageImageArrayNonUniformIndexing ||
                features.shaderInputAttachmentArrayNonUniformIndexing ||
                features.shaderUniformTexelBufferArrayNonUniformIndexing ||
                features.shaderStorageTexelBufferArrayNonUniformIndexing ||
                features.descriptorBindingUniformBufferUpdateAfterBind ||
                features.descriptorBindingSampledImageUpdateAfterBind ||
                features.descriptorBindingStorageImageUpdateAfterBind ||
                features.descriptorBindingStorageBufferUpdateAfterBind ||
                features.descriptorBindingUniformTexelBufferUpdateAfterBind ||
                features.descriptorBindingStorageTexelBufferUpdateAfterBind ||
                features.descriptorBindingUpdateUnusedWhilePending ||
                features.descriptorBindingPartiallyBound ||
                features.descriptorBindingVariableDescriptorCount ||
                features.runtimeDescriptorArray)
            {
                // make sure features have their main bool enabled
                features.descriptorIndexing = true; // IMPLICIT ENABLE Because: descriptorIndexing indicates whether the implementation supports the minimum set of descriptor indexing features
            }

            // VK_EXT_hdr_metadata Requires VK_KHR_swapchain to be enabled
            if (features.hdrMetadata)
            {
                features.swapchainMode |= E_SWAPCHAIN_MODE::ESM_SURFACE;
                // And VK_KHR_swapchain requires VK_KHR_surface instance extension
            }
        
            // VK_GOOGLE_display_timing Requires VK_KHR_swapchain to be enabled
            if (features.displayTiming)
            {
                features.swapchainMode |= E_SWAPCHAIN_MODE::ESM_SURFACE;
                // And VK_KHR_swapchain requires VK_KHR_surface instance extension
            }

            // `VK_EXT_fragment_density_map2` Requires `FragmentDensityMapFeaturesEXT`
            if (features.fragmentDensityMapDeferred)
            {
                features.fragmentDensityMap = true;
            }

            if (features.workgroupMemoryExplicitLayoutScalarBlockLayout ||
                features.workgroupMemoryExplicitLayout8BitAccess ||
                features.workgroupMemoryExplicitLayout16BitAccess)
            {
                // make sure features have their main bool enabled!
                features.workgroupMemoryExplicitLayout = true;
            }
        
            if (features.cooperativeMatrixRobustBufferAccess)
            {
                // make sure features have their main bool enabled!
                features.cooperativeMatrix = true;
            }
        
            if (features.inheritedConditionalRendering)
            {
                // make sure features have their main bool enabled!
                features.conditionalRendering = true;
            }
        
            if (features.fragmentDensityMapDynamic ||
                features.fragmentDensityMapNonSubsampledImages)
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
            
            // If sparseImageInt64Atomics is enabled, shaderImageInt64Atomics must be enabled
            if (features.sparseImageInt64Atomics)
            {
                features.shaderImageInt64Atomics = true;
            }
            // If sparseImageFloat32Atomics is enabled, shaderImageFloat32Atomics must be enabled
            if (features.sparseImageFloat32Atomics)
            {
                features.shaderImageFloat32Atomics = true;
            }
            // If sparseImageFloat32AtomicAdd is enabled, shaderImageFloat32AtomicAdd must be enabled
            if (features.sparseImageFloat32AtomicAdd)
            {
                features.shaderImageFloat32AtomicAdd = true;
            }
            // If sparseImageFloat32AtomicMinMax is enabled, shaderImageFloat32AtomicMinMax must be enabled
            if (features.sparseImageFloat32AtomicMinMax)
            {
                features.shaderImageFloat32AtomicMinMax = true;
            }
        }

        core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice_impl(ILogicalDevice::SCreationParams&& params) override;

    private:
        renderdoc_api_t* const m_rdoc_api;
        const VkPhysicalDevice m_vkPhysicalDevice;
};

}

#endif
