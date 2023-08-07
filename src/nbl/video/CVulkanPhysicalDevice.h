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

        core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice_impl(ILogicalDevice::SCreationParams&& params) override
        {
            // We might alter it to account for dependancies.
            resolveFeatureDependencies(params.featuresToEnable);
            SFeatures& enabledFeatures = params.featuresToEnable;


            core::unordered_set<core::string> extensionsToEnable;
            // Extensions we REQUIRE
            extensionsToEnable.insert(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
            VkPhysicalDeviceSynchronization2FeaturesKHR synchronization2Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR, nullptr };
            VkBaseInStructure* featuresTail = reinterpret_cast<VkBaseInStructure*>(&synchronization2Features);

            //
            VkPhysicalDeviceVulkan12Features vulkan12Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &synchronization2Features };
            VkPhysicalDeviceVulkan11Features vulkan11Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &vulkan12Features };
            VkPhysicalDeviceFeatures2 vk_deviceFeatures2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &vulkan11Features };

            // Important notes on extension dependancies, both instance and device
            /*
                If an extension is supported (as queried by vkEnumerateInstanceExtensionProperties or vkEnumerateDeviceExtensionProperties), 
                then required extensions of that extension must also be supported for the same instance or physical device.

                Any device extension that has an instance extension dependency that is not enabled by vkCreateInstance is considered to be unsupported,
                hence it must not be returned by vkEnumerateDeviceExtensionProperties for any VkPhysicalDevice child of the instance. Instance extensions do not have dependencies on device extensions.

                Conclusion: We don't need to specifically check instance extension dependancies but we can do it through apiConnection->getEnableFeatures to hint the user on what might be wrong 
            */


            // Extensions promoted to 1.3 core
            VkPhysicalDeviceImageRobustnessFeaturesEXT                  imageRobustnessFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES_EXT, nullptr };
            VkPhysicalDevicePipelineCreationCacheControlFeaturesEXT     pipelineCreationCacheControlFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES_EXT, nullptr };
            VkPhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT   shaderDemoteToHelperInvocationFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES_EXT, nullptr };
            VkPhysicalDeviceShaderTerminateInvocationFeaturesKHR        shaderTerminateInvocationFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES_KHR, nullptr };
            VkPhysicalDeviceSubgroupSizeControlFeaturesEXT              subgroupSizeControlFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT, nullptr };
            VkPhysicalDeviceTextureCompressionASTCHDRFeaturesEXT        textureCompressionASTCHDRFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES_EXT, nullptr };

            // Real Extensions
            VkPhysicalDeviceTexelBufferAlignmentFeaturesEXT                 texelBufferAlignmentFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_FEATURES_EXT, nullptr };
            VkPhysicalDeviceColorWriteEnableFeaturesEXT                     colorWriteEnableFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COLOR_WRITE_ENABLE_FEATURES_EXT, nullptr };
            VkPhysicalDeviceConditionalRenderingFeaturesEXT                 conditionalRenderingFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT, nullptr };
            VkPhysicalDeviceDeviceMemoryReportFeaturesEXT                   deviceMemoryReportFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_MEMORY_REPORT_FEATURES_EXT, nullptr };
            VkPhysicalDeviceFragmentDensityMapFeaturesEXT                   fragmentDensityMapFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_FEATURES_EXT, nullptr };
            VkPhysicalDeviceFragmentDensityMap2FeaturesEXT                  fragmentDensityMap2Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_2_FEATURES_EXT, nullptr };
            VkPhysicalDeviceInlineUniformBlockFeatures                      inlineUniformBlockFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES, nullptr };
            VkPhysicalDeviceLineRasterizationFeaturesEXT                    lineRasterizationFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_EXT, nullptr };
            VkPhysicalDeviceMemoryPriorityFeaturesEXT                       memoryPriorityFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT, nullptr };
            VkPhysicalDeviceRobustness2FeaturesEXT                          robustness2Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT, nullptr };
            VkPhysicalDevicePerformanceQueryFeaturesKHR                     performanceQueryFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PERFORMANCE_QUERY_FEATURES_KHR, nullptr };
            VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR         pipelineExecutablePropertiesFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR, nullptr };
            VkPhysicalDeviceMaintenance4Features                            maintenance4Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES, nullptr };
            VkPhysicalDeviceCoherentMemoryFeaturesAMD                       coherentMemoryFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD, nullptr };
            VkPhysicalDeviceGlobalPriorityQueryFeaturesKHR                  globalPriorityFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GLOBAL_PRIORITY_QUERY_FEATURES_KHR, nullptr };
            VkPhysicalDeviceCoverageReductionModeFeaturesNV                 coverageReductionModeFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COVERAGE_REDUCTION_MODE_FEATURES_NV, nullptr };
            VkPhysicalDeviceDeviceGeneratedCommandsFeaturesNV               deviceGeneratedCommandsFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_NV, nullptr };
            VkPhysicalDeviceMeshShaderFeaturesNV                            meshShaderFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_NV, nullptr };
            VkPhysicalDeviceRepresentativeFragmentTestFeaturesNV            representativeFragmentTestFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_REPRESENTATIVE_FRAGMENT_TEST_FEATURES_NV, nullptr };
            VkPhysicalDeviceShaderImageFootprintFeaturesNV                  shaderImageFootprintFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_FOOTPRINT_FEATURES_NV, nullptr };
            VkPhysicalDeviceComputeShaderDerivativesFeaturesNV              computeShaderDerivativesFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_NV, nullptr };
            VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR        workgroupMemoryExplicitLayout = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR, nullptr };
            VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR     subgroupUniformControlFlowFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_FEATURES_KHR, nullptr };
            VkPhysicalDeviceShaderClockFeaturesKHR                          shaderClockFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR, nullptr };
            VkPhysicalDeviceShaderIntegerFunctions2FeaturesINTEL            intelShaderIntegerFunctions2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_FUNCTIONS_2_FEATURES_INTEL, nullptr };
            VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT               shaderImageAtomicInt64Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT, nullptr };
            VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT                   shaderAtomicFloat2Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT, nullptr };
            VkPhysicalDeviceShaderAtomicFloatFeaturesEXT                    shaderAtomicFloatFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT, nullptr };
            VkPhysicalDeviceIndexTypeUint8FeaturesEXT                       indexTypeUint8Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT, nullptr };
            VkPhysicalDeviceRasterizationOrderAttachmentAccessFeaturesARM   rasterizationOrderAttachmentAccessFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_FEATURES_ARM, nullptr };
            VkPhysicalDeviceShaderIntegerDotProductFeatures                 shaderIntegerDotProductFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES, nullptr };
            VkPhysicalDeviceRayTracingMotionBlurFeaturesNV                  rayTracingMotionBlurFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MOTION_BLUR_FEATURES_NV, nullptr };
            VkPhysicalDeviceASTCDecodeFeaturesEXT                           astcDecodeFeaturesEXT = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ASTC_DECODE_FEATURES_EXT, nullptr };
            VkPhysicalDeviceShaderSMBuiltinsFeaturesNV                      shaderSMBuiltinsFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_FEATURES_NV, nullptr };
            VkPhysicalDeviceCooperativeMatrixFeaturesNV                     cooperativeMatrixFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_NV, nullptr };
            VkPhysicalDeviceRayTracingPipelineFeaturesKHR                   rayTracingPipelineFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR, nullptr };
            VkPhysicalDeviceAccelerationStructureFeaturesKHR                accelerationStructureFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, nullptr };
            VkPhysicalDeviceRayQueryFeaturesKHR                             rayQueryFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR, nullptr };
            VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT              fragmentShaderInterlockFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT, nullptr };
        

            auto insertExtensionIfAvailable = [&](const char* extName) -> bool
            {
                if(isExtensionSupported(extName))
                {
                    extensionsToEnable.insert(extName);
                    return true;
                }
                else
                    return false;
            };

            auto uncondAddFeatureToChain = [&featuresTail](void* feature) -> void
            {
                VkBaseInStructure* toAdd = reinterpret_cast<VkBaseInStructure*>(feature);
                featuresTail->pNext = toAdd;
                featuresTail = toAdd;
            };

            // prime ourselves with good defaults
            {
                // special handling of texture compression/format extensions 
                if (insertExtensionIfAvailable(VK_EXT_TEXTURE_COMPRESSION_ASTC_HDR_EXTENSION_NAME))
                    uncondAddFeatureToChain(&textureCompressionASTCHDRFeatures);
                if (insertExtensionIfAvailable(VK_EXT_ASTC_DECODE_MODE_EXTENSION_NAME))
                    uncondAddFeatureToChain(&astcDecodeFeaturesEXT);

                // we actually re-query all available Vulkan <= MinimumApiVersion features so that by default they're all enabled unless we explicitly disable
                vkGetPhysicalDeviceFeatures2(m_vkPhysicalDevice,&vk_deviceFeatures2);
            }

            // Vulkan has problems with having features in the feature chain that have all values set to false.
            // For example having an empty "RayTracingPipelineFeaturesKHR" in the chain will lead to validation errors for RayQueryONLY applications.
            auto addFeatureToChain = [&featuresTail,uncondAddFeatureToChain](void* feature) -> void
            {
                VkBaseInStructure* toAdd = reinterpret_cast<VkBaseInStructure*>(feature);
                // For protecting against duplication of feature structures that may be requested to add to chain twice due to extension requirements
                if (toAdd->pNext==nullptr && toAdd!=featuresTail);
                    uncondAddFeatureToChain(toAdd);
            };

            // A. Enable by Default, exposed as limits : add names to string and structs to feature chain
            {
                if (insertExtensionIfAvailable(VK_INTEL_SHADER_INTEGER_FUNCTIONS_2_EXTENSION_NAME))
                {
                    // All Requirements Exist in Vulkan 1.1
                    intelShaderIntegerFunctions2.shaderIntegerFunctions2 = m_properties.limits.shaderIntegerFunctions2;
                    addFeatureToChain(&intelShaderIntegerFunctions2);
                }

                if (insertExtensionIfAvailable(VK_KHR_SHADER_CLOCK_EXTENSION_NAME))
                {
                    // All Requirements Exist in Vulkan 1.1
                    shaderClockFeatures.shaderSubgroupClock = m_properties.limits.shaderSubgroupClock;
                    addFeatureToChain(&shaderClockFeatures);
                }
            
                if (insertExtensionIfAvailable(VK_NV_SHADER_IMAGE_FOOTPRINT_EXTENSION_NAME))
                {
                    // All Requirements Exist in Vulkan 1.1
                    shaderImageFootprintFeatures.imageFootprint = m_properties.limits.imageFootprint;
                    addFeatureToChain(&shaderImageFootprintFeatures);
                }

                if(insertExtensionIfAvailable(VK_EXT_TEXEL_BUFFER_ALIGNMENT_EXTENSION_NAME))
                {
                    // All Requirements Exist in Vulkan 1.1
                    texelBufferAlignmentFeatures.texelBufferAlignment = m_properties.limits.texelBufferAlignment;
                    addFeatureToChain(&texelBufferAlignmentFeatures);
                }

                if(insertExtensionIfAvailable(VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME))
                {
                    // No Extension Requirements
                    shaderSMBuiltinsFeatures.shaderSMBuiltins = m_properties.limits.shaderSMBuiltins;
                    addFeatureToChain(&shaderSMBuiltinsFeatures);
                }

                if (insertExtensionIfAvailable(VK_KHR_MAINTENANCE_4_EXTENSION_NAME))
                {
                    // No Extension Requirements
                    maintenance4Features.maintenance4 = m_properties.limits.workgroupSizeFromSpecConstant;
                    addFeatureToChain(&maintenance4Features);
                }

                insertExtensionIfAvailable(VK_NV_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME); // No Extension Requirements
                insertExtensionIfAvailable(VK_AMD_GCN_SHADER_EXTENSION_NAME); // No Extension Requirements
                insertExtensionIfAvailable(VK_AMD_GPU_SHADER_HALF_FLOAT_EXTENSION_NAME); // No Extension Requirements
                insertExtensionIfAvailable(VK_AMD_SHADER_BALLOT_EXTENSION_NAME); // No Extension Requirements
                insertExtensionIfAvailable(VK_AMD_SHADER_IMAGE_LOAD_STORE_LOD_EXTENSION_NAME); // No Extension Requirements
                insertExtensionIfAvailable(VK_AMD_SHADER_TRINARY_MINMAX_EXTENSION_NAME); // No Extension Requirements
                insertExtensionIfAvailable(VK_EXT_POST_DEPTH_COVERAGE_EXTENSION_NAME); // No Extension Requirements
                insertExtensionIfAvailable(VK_EXT_SHADER_STENCIL_EXPORT_EXTENSION_NAME); // No Extension Requirements
                insertExtensionIfAvailable(VK_GOOGLE_DECORATE_STRING_EXTENSION_NAME); // No Extension Requirements

                insertExtensionIfAvailable(VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME); 
    #ifdef _NBL_WINDOWS_API_
                insertExtensionIfAvailable(VK_KHR_EXTERNAL_FENCE_WIN32_EXTENSION_NAME); // All Requirements Exist in Vulkan 1.1 (including instance extensions)
    #endif
                insertExtensionIfAvailable(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME); 
    #ifdef _NBL_WINDOWS_API_
                insertExtensionIfAvailable(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME); // All Requirements Exist in Vulkan 1.1 (including instance extensions)
    #endif
                insertExtensionIfAvailable(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME); 
    #ifdef _NBL_WINDOWS_API_
                insertExtensionIfAvailable(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME); // All Requirements Exist in Vulkan 1.1 (including instance extensions)
    #endif

                insertExtensionIfAvailable(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME); // No Extension Requirements
                insertExtensionIfAvailable(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME); // No Extension Requirements
                insertExtensionIfAvailable(VK_NV_GEOMETRY_SHADER_PASSTHROUGH_EXTENSION_NAME); // No Extension Requirements
                insertExtensionIfAvailable(VK_NV_VIEWPORT_SWIZZLE_EXTENSION_NAME); // No Extension Requirements
            }

            // B. FeaturesToEnable: add names to strings and structs to feature chain
            
        /* Vulkan 1.0 Core  */
        vk_deviceFeatures2.features.robustBufferAccess = enabledFeatures.robustBufferAccess;
        vk_deviceFeatures2.features.fullDrawIndexUint32 = true; // ROADMAP 2022
        vk_deviceFeatures2.features.imageCubeArray = true; // ROADMAP 2022
        vk_deviceFeatures2.features.independentBlend = true; // ROADMAP 2022
        vk_deviceFeatures2.features.geometryShader = enabledFeatures.geometryShader;
        vk_deviceFeatures2.features.tessellationShader = enabledFeatures.tessellationShader;
        vk_deviceFeatures2.features.sampleRateShading = true; // ROADMAP 2022
        vk_deviceFeatures2.features.dualSrcBlend = enabledFeatures.dualSrcBlend;
        vk_deviceFeatures2.features.logicOp = enabledFeatures.logicOp;
        vk_deviceFeatures2.features.multiDrawIndirect = true; // ROADMAP 2022
        vk_deviceFeatures2.features.drawIndirectFirstInstance = true; // ROADMAP 2022
        vk_deviceFeatures2.features.depthClamp = true; // ROADMAP 2022
        vk_deviceFeatures2.features.depthBiasClamp = true; // ROADMAP 2022
        vk_deviceFeatures2.features.fillModeNonSolid = enabledFeatures.fillModeNonSolid;
        vk_deviceFeatures2.features.depthBounds = enabledFeatures.depthBounds;
        vk_deviceFeatures2.features.wideLines = enabledFeatures.wideLines;
        vk_deviceFeatures2.features.largePoints = enabledFeatures.largePoints;
        vk_deviceFeatures2.features.alphaToOne = enabledFeatures.alphaToOne;
        vk_deviceFeatures2.features.multiViewport = enabledFeatures.multiViewport;
        vk_deviceFeatures2.features.samplerAnisotropy = true; // ROADMAP
        // leave defaulted
        //vk_deviceFeatures2.features.textureCompressionETC2;
        //vk_deviceFeatures2.features.textureCompressionASTC_LDR;
        //vk_deviceFeatures2.features.textureCompressionBC;
        vk_deviceFeatures2.features.occlusionQueryPrecise = true; // ROADMAP 2022
        vk_deviceFeatures2.features.pipelineStatisticsQuery = enabledFeatures.pipelineStatisticsQuery;
        vk_deviceFeatures2.features.vertexPipelineStoresAndAtomics = m_properties.limits.vertexPipelineStoresAndAtomics;
        vk_deviceFeatures2.features.fragmentStoresAndAtomics = m_properties.limits.fragmentStoresAndAtomics;
        vk_deviceFeatures2.features.shaderTessellationAndGeometryPointSize = m_properties.limits.shaderTessellationAndGeometryPointSize;
        vk_deviceFeatures2.features.shaderImageGatherExtended = m_properties.limits.shaderImageGatherExtended;
        vk_deviceFeatures2.features.shaderStorageImageExtendedFormats = true; // ROADMAP 2022
        vk_deviceFeatures2.features.shaderStorageImageMultisample = m_properties.limits.shaderStorageImageMultisample;
        vk_deviceFeatures2.features.shaderStorageImageReadWithoutFormat = enabledFeatures.shaderStorageImageReadWithoutFormat;
        vk_deviceFeatures2.features.shaderStorageImageWriteWithoutFormat = enabledFeatures.shaderStorageImageWriteWithoutFormat;
        vk_deviceFeatures2.features.shaderUniformBufferArrayDynamicIndexing = true; // ROADMAP 2022
        vk_deviceFeatures2.features.shaderSampledImageArrayDynamicIndexing = true; // ROADMAP 2022
        vk_deviceFeatures2.features.shaderStorageBufferArrayDynamicIndexing = true; // ROADMAP 2022
        vk_deviceFeatures2.features.shaderStorageImageArrayDynamicIndexing = m_properties.limits.shaderStorageImageArrayDynamicIndexing;
        vk_deviceFeatures2.features.shaderClipDistance = enabledFeatures.shaderClipDistance;
        vk_deviceFeatures2.features.shaderCullDistance = enabledFeatures.shaderCullDistance;
        vk_deviceFeatures2.features.shaderInt64 = m_properties.limits.shaderInt64;
        vk_deviceFeatures2.features.shaderInt16 = m_properties.limits.shaderInt16;
        vk_deviceFeatures2.features.shaderFloat64 = m_properties.limits.shaderFloat64;
        vk_deviceFeatures2.features.shaderResourceResidency = enabledFeatures.shaderResourceResidency;
        vk_deviceFeatures2.features.shaderResourceMinLod = enabledFeatures.shaderResourceMinLod;
        vk_deviceFeatures2.features.sparseBinding = false; // not implemented yet
        vk_deviceFeatures2.features.sparseResidencyBuffer = false; // not implemented yet
        vk_deviceFeatures2.features.sparseResidencyImage2D = false; // not implemented yet
        vk_deviceFeatures2.features.sparseResidencyImage3D = false; // not implemented yet
        vk_deviceFeatures2.features.sparseResidency2Samples = false; // not implemented yet
        vk_deviceFeatures2.features.sparseResidency4Samples = false; // not implemented yet
        vk_deviceFeatures2.features.sparseResidency8Samples = false; // not implemented yet
        vk_deviceFeatures2.features.sparseResidency16Samples = false; // not implemented yet
        vk_deviceFeatures2.features.sparseResidencyAliased = false; // not implemented yet
        vk_deviceFeatures2.features.variableMultisampleRate = enabledFeatures.variableMultisampleRate;
        vk_deviceFeatures2.features.inheritedQueries = enabledFeatures.inheritedQueries;

        /* Vulkan 1.1 Core */
        vulkan11Features.storageBuffer16BitAccess = m_properties.limits.storageBuffer16BitAccess;
        vulkan11Features.uniformAndStorageBuffer16BitAccess = m_properties.limits.uniformAndStorageBuffer16BitAccess;
        vulkan11Features.storagePushConstant16 = m_properties.limits.storagePushConstant16;
        vulkan11Features.storageInputOutput16 = m_properties.limits.storageInputOutput16;
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#features-requirements
        vulkan11Features.multiview = true;
        vulkan11Features.multiviewGeometryShader = false;// = TODO;
        vulkan11Features.multiviewTessellationShader = false;// = TODO;
        vulkan11Features.variablePointers = m_properties.limits.variablePointers;
        vulkan11Features.variablePointersStorageBuffer = vulkan11Features.variablePointers;
        // not yet
        vulkan11Features.protectedMemory = false;
        vulkan11Features.samplerYcbcrConversion = false;
        vulkan11Features.shaderDrawParameters = enabledFeatures.shaderDrawParameters;
            
        /* Vulkan 1.2 Core */
        vulkan12Features.samplerMirrorClampToEdge = true; // ubiquitous
        vulkan12Features.drawIndirectCount = m_properties.limits.drawIndirectCount;
        vulkan12Features.storageBuffer8BitAccess = m_properties.limits.storageBuffer8BitAccess;
        vulkan12Features.uniformAndStorageBuffer8BitAccess = m_properties.limits.uniformAndStorageBuffer8BitAccess;
        vulkan12Features.storagePushConstant8 = m_properties.limits.storagePushConstant8;
        vulkan12Features.shaderBufferInt64Atomics = m_properties.limits.shaderBufferInt64Atomics;
        vulkan12Features.shaderSharedInt64Atomics = m_properties.limits.shaderSharedInt64Atomics;
        vulkan12Features.shaderFloat16 = m_properties.limits.shaderFloat16;
        vulkan12Features.shaderInt8 = m_properties.limits.shaderInt8;
        vulkan12Features.descriptorIndexing = enabledFeatures.descriptorIndexing;
        vulkan12Features.shaderInputAttachmentArrayDynamicIndexing = enabledFeatures.shaderInputAttachmentArrayDynamicIndexing;
        vulkan12Features.shaderUniformTexelBufferArrayDynamicIndexing = enabledFeatures.shaderUniformTexelBufferArrayDynamicIndexing;
        vulkan12Features.shaderStorageTexelBufferArrayDynamicIndexing = enabledFeatures.shaderStorageTexelBufferArrayDynamicIndexing;
        vulkan12Features.shaderUniformBufferArrayNonUniformIndexing = enabledFeatures.shaderUniformBufferArrayNonUniformIndexing;
        vulkan12Features.shaderSampledImageArrayNonUniformIndexing = enabledFeatures.shaderSampledImageArrayNonUniformIndexing;
        vulkan12Features.shaderStorageBufferArrayNonUniformIndexing = enabledFeatures.shaderStorageBufferArrayNonUniformIndexing;
        vulkan12Features.shaderStorageImageArrayNonUniformIndexing = enabledFeatures.shaderStorageImageArrayNonUniformIndexing;
        vulkan12Features.shaderInputAttachmentArrayNonUniformIndexing = enabledFeatures.shaderInputAttachmentArrayNonUniformIndexing;
        vulkan12Features.shaderUniformTexelBufferArrayNonUniformIndexing = enabledFeatures.shaderUniformTexelBufferArrayNonUniformIndexing;
        vulkan12Features.shaderStorageTexelBufferArrayNonUniformIndexing = enabledFeatures.shaderStorageTexelBufferArrayNonUniformIndexing;
        vulkan12Features.descriptorBindingUniformBufferUpdateAfterBind = enabledFeatures.descriptorBindingUniformBufferUpdateAfterBind;
        vulkan12Features.descriptorBindingSampledImageUpdateAfterBind = enabledFeatures.descriptorBindingSampledImageUpdateAfterBind;
        vulkan12Features.descriptorBindingStorageImageUpdateAfterBind = enabledFeatures.descriptorBindingStorageImageUpdateAfterBind;
        vulkan12Features.descriptorBindingStorageBufferUpdateAfterBind = enabledFeatures.descriptorBindingStorageBufferUpdateAfterBind;
        vulkan12Features.descriptorBindingUniformTexelBufferUpdateAfterBind = enabledFeatures.descriptorBindingUniformTexelBufferUpdateAfterBind;
        vulkan12Features.descriptorBindingStorageTexelBufferUpdateAfterBind = enabledFeatures.descriptorBindingStorageTexelBufferUpdateAfterBind;
        vulkan12Features.descriptorBindingUpdateUnusedWhilePending = enabledFeatures.descriptorBindingUpdateUnusedWhilePending;
        vulkan12Features.descriptorBindingPartiallyBound = enabledFeatures.descriptorBindingPartiallyBound;
        vulkan12Features.descriptorBindingVariableDescriptorCount = enabledFeatures.descriptorBindingVariableDescriptorCount;
        vulkan12Features.runtimeDescriptorArray = enabledFeatures.runtimeDescriptorArray;
        vulkan12Features.samplerFilterMinmax = enabledFeatures.samplerFilterMinmax;
        vulkan12Features.scalarBlockLayout = true; // ROADMAP 2022
        vulkan12Features.imagelessFramebuffer = false; // decided against
        vulkan12Features.uniformBufferStandardLayout = true; // required anyway
        vulkan12Features.shaderSubgroupExtendedTypes = true; // required anyway
        vulkan12Features.separateDepthStencilLayouts = true; // required anyway
        vulkan12Features.hostQueryReset = true; // required anyway
        vulkan12Features.timelineSemaphore = true; // required anyway
        vulkan12Features.bufferDeviceAddress = enabledFeatures.bufferDeviceAddress || enabledFeatures.bufferDeviceAddressMultiDevice;
        // Some capture tools need this but can't enable this when you set this to false (they're buggy probably, We shouldn't worry about this)
        vulkan12Features.bufferDeviceAddressCaptureReplay = bool(vulkan12Features.bufferDeviceAddress) && (m_rdoc_api!=nullptr);
        vulkan12Features.bufferDeviceAddressMultiDevice = enabledFeatures.bufferDeviceAddressMultiDevice;
        vulkan12Features.vulkanMemoryModel = m_properties.limits.vulkanMemoryModel;
        vulkan12Features.vulkanMemoryModelDeviceScope = m_properties.limits.vulkanMemoryModelDeviceScope;
        vulkan12Features.vulkanMemoryModelAvailabilityVisibilityChains = m_properties.limits.vulkanMemoryModelAvailabilityVisibilityChains;
        vulkan12Features.shaderOutputViewportIndex = m_properties.limits.shaderOutputViewportIndex;
        vulkan12Features.shaderOutputLayer = m_properties.limits.shaderOutputLayer;
        vulkan12Features.subgroupBroadcastDynamicId = true; // ubiquitous

            
#define CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(VAR_NAME, EXT_NAME, FEATURE_STRUCT)           \
        if(enabledFeatures.VAR_NAME)                                                    \
        {                                                                               \
            insertExtensionIfAvailable(EXT_NAME);                                       \
            FEATURE_STRUCT.VAR_NAME = enabledFeatures.VAR_NAME;                         \
            addFeatureToChain(&FEATURE_STRUCT);                                         \
        }

        /* Vulkan 1.3 Core */
// TODO: robustImageAccess
// TODO: pipelineCreationCacheControl
        CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(shaderDemoteToHelperInvocation, VK_EXT_SHADER_DEMOTE_TO_HELPER_INVOCATION_EXTENSION_NAME, shaderDemoteToHelperInvocationFeatures);
        CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(shaderTerminateInvocation, VK_KHR_SHADER_TERMINATE_INVOCATION_EXTENSION_NAME, shaderTerminateInvocationFeatures);         
        // Instead of checking and enabling individual features like below, I can do awesome things like this:
        /*
            CHECK_VULKAN_EXTENTION(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME, subgroupSizeControlFeatures,
                    subgroupSizeControl,
                    computeFullSubgroups);
        */
        // But I would need to enable /Zc:preprocessor in compiler So I could use __VA_OPT__ :D
        if (enabledFeatures.subgroupSizeControl ||
            enabledFeatures.computeFullSubgroups)
        {
            // All Requirements Exist in Vulkan 1.1
            insertExtensionIfAvailable(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
            subgroupSizeControlFeatures.subgroupSizeControl = enabledFeatures.subgroupSizeControl;
            subgroupSizeControlFeatures.computeFullSubgroups = enabledFeatures.computeFullSubgroups;
            addFeatureToChain(&subgroupSizeControlFeatures);
        }
        //leave defaulted
        //textureCompressionASTCHDRFeatures;
// TODO: shaderZeroInitializeWorkgroupMemory
            
        CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(shaderIntegerDotProduct, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME, shaderIntegerDotProductFeatures);
            
        if (enabledFeatures.rasterizationOrderColorAttachmentAccess ||
            enabledFeatures.rasterizationOrderDepthAttachmentAccess ||
            enabledFeatures.rasterizationOrderStencilAttachmentAccess)
        {
            // All Requirements Exist in Vulkan 1.1
            insertExtensionIfAvailable(VK_ARM_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_EXTENSION_NAME);
            rasterizationOrderAttachmentAccessFeatures.rasterizationOrderColorAttachmentAccess = enabledFeatures.rasterizationOrderColorAttachmentAccess;
            rasterizationOrderAttachmentAccessFeatures.rasterizationOrderDepthAttachmentAccess = enabledFeatures.rasterizationOrderDepthAttachmentAccess;
            rasterizationOrderAttachmentAccessFeatures.rasterizationOrderStencilAttachmentAccess = enabledFeatures.rasterizationOrderStencilAttachmentAccess;
            addFeatureToChain(&rasterizationOrderAttachmentAccessFeatures);
        }

        if (enabledFeatures.fragmentShaderSampleInterlock ||
            enabledFeatures.fragmentShaderPixelInterlock ||
            enabledFeatures.fragmentShaderShadingRateInterlock)
        {
            // All Requirements Exist in Vulkan 1.1
            insertExtensionIfAvailable(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME);
            fragmentShaderInterlockFeatures.fragmentShaderSampleInterlock = enabledFeatures.fragmentShaderSampleInterlock;
            fragmentShaderInterlockFeatures.fragmentShaderPixelInterlock = enabledFeatures.fragmentShaderPixelInterlock;
            fragmentShaderInterlockFeatures.fragmentShaderShadingRateInterlock = enabledFeatures.fragmentShaderShadingRateInterlock;
            addFeatureToChain(&fragmentShaderInterlockFeatures);
        }
            
        CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(indexTypeUint8, VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME, indexTypeUint8Features);

        if (enabledFeatures.shaderBufferFloat32Atomics ||
            enabledFeatures.shaderBufferFloat32AtomicAdd ||
            enabledFeatures.shaderBufferFloat64Atomics ||
            enabledFeatures.shaderBufferFloat64AtomicAdd ||
            enabledFeatures.shaderSharedFloat32Atomics ||
            enabledFeatures.shaderSharedFloat32AtomicAdd ||
            enabledFeatures.shaderSharedFloat64Atomics ||
            enabledFeatures.shaderSharedFloat64AtomicAdd ||
            enabledFeatures.shaderImageFloat32Atomics ||
            enabledFeatures.shaderImageFloat32AtomicAdd ||
            enabledFeatures.sparseImageFloat32Atomics ||
            enabledFeatures.sparseImageFloat32AtomicAdd)
        {
            insertExtensionIfAvailable(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
            shaderAtomicFloatFeatures.shaderBufferFloat32Atomics   = enabledFeatures.shaderBufferFloat32Atomics;
            shaderAtomicFloatFeatures.shaderBufferFloat32AtomicAdd = enabledFeatures.shaderBufferFloat32AtomicAdd;
            shaderAtomicFloatFeatures.shaderBufferFloat64Atomics   = enabledFeatures.shaderBufferFloat64Atomics;
            shaderAtomicFloatFeatures.shaderBufferFloat64AtomicAdd = enabledFeatures.shaderBufferFloat64AtomicAdd;
            shaderAtomicFloatFeatures.shaderSharedFloat32Atomics   = enabledFeatures.shaderSharedFloat32Atomics;
            shaderAtomicFloatFeatures.shaderSharedFloat32AtomicAdd = enabledFeatures.shaderSharedFloat32AtomicAdd;
            shaderAtomicFloatFeatures.shaderSharedFloat64Atomics   = enabledFeatures.shaderSharedFloat64Atomics;
            shaderAtomicFloatFeatures.shaderSharedFloat64AtomicAdd = enabledFeatures.shaderSharedFloat64AtomicAdd;
            shaderAtomicFloatFeatures.shaderImageFloat32Atomics    = enabledFeatures.shaderImageFloat32Atomics;
            shaderAtomicFloatFeatures.shaderImageFloat32AtomicAdd  = enabledFeatures.shaderImageFloat32AtomicAdd;
            shaderAtomicFloatFeatures.sparseImageFloat32Atomics    = enabledFeatures.sparseImageFloat32Atomics;
            shaderAtomicFloatFeatures.sparseImageFloat32AtomicAdd  = enabledFeatures.sparseImageFloat32AtomicAdd;
            addFeatureToChain(&shaderAtomicFloatFeatures);
        }
            
        if (enabledFeatures.shaderBufferFloat16Atomics ||
            enabledFeatures.shaderBufferFloat16AtomicAdd ||
            enabledFeatures.shaderBufferFloat16AtomicMinMax ||
            enabledFeatures.shaderBufferFloat32AtomicMinMax ||
            enabledFeatures.shaderBufferFloat64AtomicMinMax ||
            enabledFeatures.shaderSharedFloat16Atomics ||
            enabledFeatures.shaderSharedFloat16AtomicAdd ||
            enabledFeatures.shaderSharedFloat16AtomicMinMax ||
            enabledFeatures.shaderSharedFloat32AtomicMinMax ||
            enabledFeatures.shaderSharedFloat64AtomicMinMax ||
            enabledFeatures.shaderImageFloat32AtomicMinMax ||
            enabledFeatures.sparseImageFloat32AtomicMinMax)
        {
            insertExtensionIfAvailable(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME); // Requirement
            insertExtensionIfAvailable(VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME);
            shaderAtomicFloat2Features.shaderBufferFloat16Atomics       = enabledFeatures.shaderBufferFloat16Atomics;
            shaderAtomicFloat2Features.shaderBufferFloat16AtomicAdd     = enabledFeatures.shaderBufferFloat16AtomicAdd;
            shaderAtomicFloat2Features.shaderBufferFloat16AtomicMinMax  = enabledFeatures.shaderBufferFloat16AtomicMinMax;
            shaderAtomicFloat2Features.shaderBufferFloat32AtomicMinMax  = enabledFeatures.shaderBufferFloat32AtomicMinMax;
            shaderAtomicFloat2Features.shaderBufferFloat64AtomicMinMax  = enabledFeatures.shaderBufferFloat64AtomicMinMax;
            shaderAtomicFloat2Features.shaderSharedFloat16Atomics       = enabledFeatures.shaderSharedFloat16Atomics;
            shaderAtomicFloat2Features.shaderSharedFloat16AtomicAdd     = enabledFeatures.shaderSharedFloat16AtomicAdd;
            shaderAtomicFloat2Features.shaderSharedFloat16AtomicMinMax  = enabledFeatures.shaderSharedFloat16AtomicMinMax;
            shaderAtomicFloat2Features.shaderSharedFloat32AtomicMinMax  = enabledFeatures.shaderSharedFloat32AtomicMinMax;
            shaderAtomicFloat2Features.shaderSharedFloat64AtomicMinMax  = enabledFeatures.shaderSharedFloat64AtomicMinMax;
            shaderAtomicFloat2Features.shaderImageFloat32AtomicMinMax   = enabledFeatures.shaderImageFloat32AtomicMinMax;
            shaderAtomicFloat2Features.sparseImageFloat32AtomicMinMax   = enabledFeatures.sparseImageFloat32AtomicMinMax;
            addFeatureToChain(&shaderAtomicFloat2Features);
        }
            
        if (enabledFeatures.shaderImageInt64Atomics ||
            enabledFeatures.sparseImageInt64Atomics)
        {
            // All Requirements Exist in Vulkan 1.1
            insertExtensionIfAvailable(VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME);
            shaderImageAtomicInt64Features.shaderImageInt64Atomics = enabledFeatures.shaderImageInt64Atomics;
            shaderImageAtomicInt64Features.sparseImageInt64Atomics = enabledFeatures.sparseImageInt64Atomics;
            addFeatureToChain(&shaderImageAtomicInt64Features);
        }
            
        if (enabledFeatures.accelerationStructure ||
            enabledFeatures.accelerationStructureIndirectBuild ||
            enabledFeatures.accelerationStructureHostCommands ||
            enabledFeatures.descriptorBindingAccelerationStructureUpdateAfterBind)
        {
            // IMPLICIT ENABLE: descriptorIndexing -> Already handled because of resolveFeatureDependencies(featuresToEnable);
            // IMPLICIT ENABLE: bufferDeviceAddress -> Already handled because of resolveFeatureDependencies(featuresToEnable);
            // IMPLICIT ENABLE: VK_KHR_DEFERRED_HOST_OPERATIONS -> Already handled because of resolveFeatureDependencies(featuresToEnable);

            insertExtensionIfAvailable(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
            accelerationStructureFeatures.accelerationStructure = enabledFeatures.accelerationStructure;
            accelerationStructureFeatures.accelerationStructureIndirectBuild = enabledFeatures.accelerationStructureIndirectBuild;
            accelerationStructureFeatures.accelerationStructureHostCommands = enabledFeatures.accelerationStructureHostCommands;
            accelerationStructureFeatures.descriptorBindingAccelerationStructureUpdateAfterBind = enabledFeatures.descriptorBindingAccelerationStructureUpdateAfterBind;
            addFeatureToChain(&accelerationStructureFeatures);
        }
            
        if (enabledFeatures.rayQuery)
        {
            // IMPLICIT ENABLE: accelerationStructure -> Already handled because of resolveFeatureDependencies(featuresToEnable);
            insertExtensionIfAvailable(VK_KHR_SPIRV_1_4_EXTENSION_NAME); // Requires VK_KHR_spirv_1_4
            insertExtensionIfAvailable(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME); // VK_KHR_spirv_1_4 requires VK_KHR_shader_float_controls

            insertExtensionIfAvailable(VK_KHR_RAY_QUERY_EXTENSION_NAME);
            rayQueryFeatures.rayQuery = enabledFeatures.rayQuery;
            addFeatureToChain(&rayQueryFeatures);
        }
            
        if (enabledFeatures.rayTracingPipeline ||
            enabledFeatures.rayTracingPipelineTraceRaysIndirect ||
            enabledFeatures.rayTraversalPrimitiveCulling)
        {
            // IMPLICIT ENABLE: accelerationStructure -> Already handled because of resolveFeatureDependencies(featuresToEnable);
            insertExtensionIfAvailable(VK_KHR_SPIRV_1_4_EXTENSION_NAME); // Requires VK_KHR_spirv_1_4
            insertExtensionIfAvailable(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME); // VK_KHR_spirv_1_4 requires VK_KHR_shader_float_controls
                
            insertExtensionIfAvailable(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
            rayTracingPipelineFeatures.rayTracingPipeline = enabledFeatures.rayTracingPipeline;
            rayTracingPipelineFeatures.rayTracingPipelineTraceRaysIndirect = enabledFeatures.rayTracingPipelineTraceRaysIndirect;
            rayTracingPipelineFeatures.rayTraversalPrimitiveCulling = enabledFeatures.rayTraversalPrimitiveCulling;
            addFeatureToChain(&rayTracingPipelineFeatures);
        }

        if (enabledFeatures.rayTracingMotionBlur ||
            enabledFeatures.rayTracingMotionBlurPipelineTraceRaysIndirect)
        {
            // IMPLICIT ENABLE: rayTracingPipeline -> Already handled because of resolveFeatureDependencies(featuresToEnable);
            insertExtensionIfAvailable(VK_NV_RAY_TRACING_MOTION_BLUR_EXTENSION_NAME);
            rayTracingMotionBlurFeatures.rayTracingMotionBlur = enabledFeatures.rayTracingMotionBlur;
            rayTracingMotionBlurFeatures.rayTracingMotionBlurPipelineTraceRaysIndirect = enabledFeatures.rayTracingMotionBlurPipelineTraceRaysIndirect;
            addFeatureToChain(&rayTracingMotionBlurFeatures);
        }
            
        if(enabledFeatures.shaderDeviceClock)
        {
            // shaderClockFeatures.shaderSubgroupClock will be enabled by defaul and the extension name and feature struct should've been added by now, but the functions implicitly protect against duplication
            insertExtensionIfAvailable(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
            shaderClockFeatures.shaderDeviceClock = enabledFeatures.shaderDeviceClock;
            addFeatureToChain(&shaderClockFeatures);
        }

        CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(shaderSubgroupUniformControlFlow, VK_KHR_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_EXTENSION_NAME, subgroupUniformControlFlowFeatures);
            
        if (enabledFeatures.workgroupMemoryExplicitLayout ||
            enabledFeatures.workgroupMemoryExplicitLayoutScalarBlockLayout ||
            enabledFeatures.workgroupMemoryExplicitLayout8BitAccess ||
            enabledFeatures.workgroupMemoryExplicitLayout16BitAccess)
        {
            insertExtensionIfAvailable(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME);
            workgroupMemoryExplicitLayout.workgroupMemoryExplicitLayout = enabledFeatures.workgroupMemoryExplicitLayout;
            workgroupMemoryExplicitLayout.workgroupMemoryExplicitLayoutScalarBlockLayout = enabledFeatures.workgroupMemoryExplicitLayoutScalarBlockLayout;
            workgroupMemoryExplicitLayout.workgroupMemoryExplicitLayout8BitAccess = enabledFeatures.workgroupMemoryExplicitLayout8BitAccess;
            workgroupMemoryExplicitLayout.workgroupMemoryExplicitLayout16BitAccess = enabledFeatures.workgroupMemoryExplicitLayout16BitAccess;
            addFeatureToChain(&workgroupMemoryExplicitLayout);
        }
            
        if (enabledFeatures.computeDerivativeGroupQuads ||
            enabledFeatures.computeDerivativeGroupLinear)
        {
            // All Requirements Exist in Vulkan 1.1
            insertExtensionIfAvailable(VK_NV_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME);
            computeShaderDerivativesFeatures.computeDerivativeGroupQuads = enabledFeatures.computeDerivativeGroupQuads;
            computeShaderDerivativesFeatures.computeDerivativeGroupLinear = enabledFeatures.computeDerivativeGroupLinear;
            addFeatureToChain(&computeShaderDerivativesFeatures);
        }
            
        if (enabledFeatures.cooperativeMatrix ||
            enabledFeatures.cooperativeMatrixRobustBufferAccess)
        {
            // All Requirements Exist in Vulkan 1.1
            insertExtensionIfAvailable(VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME);
            cooperativeMatrixFeatures.cooperativeMatrix = enabledFeatures.cooperativeMatrix;
            cooperativeMatrixFeatures.cooperativeMatrixRobustBufferAccess = enabledFeatures.cooperativeMatrixRobustBufferAccess;
            addFeatureToChain(&cooperativeMatrixFeatures);
        }
            
        CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(coverageReductionMode, VK_NV_COVERAGE_REDUCTION_MODE_EXTENSION_NAME, coverageReductionModeFeatures);
            
        // IMPLICIT ENABLE: deviceGeneratedCommands requires bufferDeviceAddress -> Already handled because of resolveFeatureDependencies(featuresToEnable); 
        CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(deviceGeneratedCommands, VK_NV_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME, deviceGeneratedCommandsFeatures);
            
        if (enabledFeatures.taskShader ||
            enabledFeatures.meshShader)
        {
            // All Requirements Exist in Vulkan 1.1
            insertExtensionIfAvailable(VK_NV_MESH_SHADER_EXTENSION_NAME);
            meshShaderFeatures.taskShader = enabledFeatures.taskShader;
            meshShaderFeatures.meshShader = enabledFeatures.meshShader;
            addFeatureToChain(&meshShaderFeatures);
        }

        if (enabledFeatures.mixedAttachmentSamples)
        {
            // All Requirements Exist in Vulkan 1.1
            insertExtensionIfAvailable(VK_AMD_MIXED_ATTACHMENT_SAMPLES_EXTENSION_NAME);
            insertExtensionIfAvailable(VK_NV_FRAMEBUFFER_MIXED_SAMPLES_EXTENSION_NAME);
        }

        if (enabledFeatures.hdrMetadata)
        {
            // IMPLICIT ENABLE: VK_KHR_swapchain -> Already handled because of resolveFeatureDependencies(featuresToEnable); 
            insertExtensionIfAvailable(VK_EXT_HDR_METADATA_EXTENSION_NAME);
        }
            
        if (enabledFeatures.displayTiming)
        {
            // IMPLICIT ENABLE: VK_KHR_swapchain -> Already handled because of resolveFeatureDependencies(featuresToEnable); 
            insertExtensionIfAvailable(VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME);
        }
            
        if (enabledFeatures.rasterizationOrder)
            insertExtensionIfAvailable(VK_AMD_RASTERIZATION_ORDER_EXTENSION_NAME);
            
        if (enabledFeatures.shaderExplicitVertexParameter)
            insertExtensionIfAvailable(VK_AMD_SHADER_EXPLICIT_VERTEX_PARAMETER_EXTENSION_NAME);
            
        if (enabledFeatures.shaderInfoAMD)
            insertExtensionIfAvailable(VK_AMD_SHADER_INFO_EXTENSION_NAME);
            
        if (enabledFeatures.pipelineCreationCacheControl)
            insertExtensionIfAvailable(VK_EXT_PIPELINE_CREATION_CACHE_CONTROL_EXTENSION_NAME);

        CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(colorWriteEnable, VK_EXT_COLOR_WRITE_ENABLE_EXTENSION_NAME, colorWriteEnableFeatures);
            
        if (enabledFeatures.conditionalRendering ||
            enabledFeatures.inheritedConditionalRendering)
        {
            // All Requirements Exist in Vulkan 1.1
            insertExtensionIfAvailable(VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME);
            conditionalRenderingFeatures.conditionalRendering = enabledFeatures.conditionalRendering;
            conditionalRenderingFeatures.inheritedConditionalRendering = enabledFeatures.inheritedConditionalRendering;
            addFeatureToChain(&conditionalRenderingFeatures);
        }

        CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(deviceMemoryReport, VK_EXT_DEVICE_MEMORY_REPORT_EXTENSION_NAME, deviceMemoryReportFeatures);
            
        if (enabledFeatures.fragmentDensityMap ||
            enabledFeatures.fragmentDensityMapDynamic ||
            enabledFeatures.fragmentDensityMapNonSubsampledImages)
        {
            insertExtensionIfAvailable(VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME);
            fragmentDensityMapFeatures.fragmentDensityMap = enabledFeatures.fragmentDensityMap;
            fragmentDensityMapFeatures.fragmentDensityMapDynamic = enabledFeatures.fragmentDensityMapDynamic;
            fragmentDensityMapFeatures.fragmentDensityMapNonSubsampledImages = enabledFeatures.fragmentDensityMapNonSubsampledImages;
            addFeatureToChain(&fragmentDensityMapFeatures);
        }

        if (enabledFeatures.fragmentDensityMapDeferred)
        {
            // IMPLICIT ENABLE: fragmentDensityMap -> Already handled because of resolveFeatureDependencies(featuresToEnable);
            insertExtensionIfAvailable(VK_EXT_FRAGMENT_DENSITY_MAP_2_EXTENSION_NAME);
            fragmentDensityMap2Features.fragmentDensityMapDeferred = enabledFeatures.fragmentDensityMapDeferred;
            addFeatureToChain(&fragmentDensityMap2Features);
        }
            
        if (enabledFeatures.robustImageAccess)
            insertExtensionIfAvailable(VK_EXT_IMAGE_ROBUSTNESS_EXTENSION_NAME);
    
        if (enabledFeatures.inlineUniformBlock ||
            enabledFeatures.descriptorBindingInlineUniformBlockUpdateAfterBind)
        {
            // All Requirements Exist in Vulkan 1.1
            insertExtensionIfAvailable(VK_EXT_INLINE_UNIFORM_BLOCK_EXTENSION_NAME);
            inlineUniformBlockFeatures.inlineUniformBlock = enabledFeatures.inlineUniformBlock;
            inlineUniformBlockFeatures.descriptorBindingInlineUniformBlockUpdateAfterBind = enabledFeatures.descriptorBindingInlineUniformBlockUpdateAfterBind;
            addFeatureToChain(&inlineUniformBlockFeatures);
        }
            
        if (enabledFeatures.rectangularLines ||
            enabledFeatures.bresenhamLines ||
            enabledFeatures.smoothLines ||
            enabledFeatures.stippledRectangularLines ||
            enabledFeatures.stippledBresenhamLines ||
            enabledFeatures.stippledSmoothLines)
        {
            insertExtensionIfAvailable(VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME);
            lineRasterizationFeatures.rectangularLines = enabledFeatures.rectangularLines;
            lineRasterizationFeatures.bresenhamLines = enabledFeatures.bresenhamLines;
            lineRasterizationFeatures.smoothLines = enabledFeatures.smoothLines;
            lineRasterizationFeatures.stippledRectangularLines = enabledFeatures.stippledRectangularLines;
            lineRasterizationFeatures.stippledBresenhamLines = enabledFeatures.stippledBresenhamLines;
            lineRasterizationFeatures.stippledSmoothLines = enabledFeatures.stippledSmoothLines;
            addFeatureToChain(&lineRasterizationFeatures);
        }

        CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(memoryPriority, VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME, memoryPriorityFeatures);
            
        if (enabledFeatures.robustBufferAccess2 ||
            enabledFeatures.robustImageAccess2 ||
            enabledFeatures.nullDescriptor)
        {
            insertExtensionIfAvailable(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);
            robustness2Features.robustBufferAccess2 = enabledFeatures.robustBufferAccess2;
            robustness2Features.robustImageAccess2 = enabledFeatures.robustImageAccess2;
            robustness2Features.nullDescriptor = enabledFeatures.nullDescriptor;
            addFeatureToChain(&robustness2Features);
        }
            
        if (enabledFeatures.performanceCounterQueryPools ||
            enabledFeatures.performanceCounterMultipleQueryPools)
        {
            insertExtensionIfAvailable(VK_INTEL_PERFORMANCE_QUERY_EXTENSION_NAME);
            performanceQueryFeatures.performanceCounterQueryPools = enabledFeatures.performanceCounterQueryPools;
            performanceQueryFeatures.performanceCounterMultipleQueryPools = enabledFeatures.performanceCounterMultipleQueryPools;
            addFeatureToChain(&performanceQueryFeatures);
        }

        CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(pipelineExecutableInfo, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME, pipelineExecutablePropertiesFeatures);
        
        CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE(deviceCoherentMemory, VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME, coherentMemoryFeatures);
            
        if (enabledFeatures.bufferMarkerAMD)
            insertExtensionIfAvailable(VK_AMD_BUFFER_MARKER_EXTENSION_NAME);

        if (enabledFeatures.swapchainMode.hasFlags(E_SWAPCHAIN_MODE::ESM_SURFACE))
        {
            // If we reach here then the instance extension VK_KHR_Surface was definitely enabled otherwise the extension wouldn't be reported by physical device
            insertExtensionIfAvailable(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
            insertExtensionIfAvailable(VK_KHR_SWAPCHAIN_MUTABLE_FORMAT_EXTENSION_NAME);
            // TODO: https://github.com/Devsh-Graphics-Programming/Nabla/issues/508
        }
        
        if (enabledFeatures.deferredHostOperations)
            insertExtensionIfAvailable(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

#undef CHECK_VULKAN_1_2_FEATURE_FOR_SINGLE_VAR
#undef CHECK_VULKAN_1_2_FEATURE_FOR_EXT_ALIAS
#undef CHECK_VULKAN_EXTENTION_FOR_SINGLE_VAR_FEATURE
        
        core::vector<const char*> extensionStrings(extensionsToEnable.size());
        {
            uint32_t i = 0u;
            for (const auto& feature : extensionsToEnable)
                extensionStrings[i++] = feature.c_str();
        }

        // Create Device
        VkDeviceCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        vk_createInfo.pNext = &vk_deviceFeatures2;
        // Vulkan >= 1.1 Device uses createInfo.pNext to use features
        vk_createInfo.pEnabledFeatures = nullptr;
        vk_createInfo.flags = static_cast<VkDeviceCreateFlags>(0); // reserved for future use, by Vulkan

        vk_createInfo.queueCreateInfoCount = params.queueParamsCount;
        core::vector<VkDeviceQueueCreateInfo> queueCreateInfos(vk_createInfo.queueCreateInfoCount);
        for (uint32_t i = 0u; i < queueCreateInfos.size(); ++i)
        {
            const auto& qparams = params.queueParams[i];
            auto& qci = queueCreateInfos[i];
                    
            qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            qci.pNext = nullptr;
            qci.queueCount = qparams.count;
            qci.queueFamilyIndex = qparams.familyIndex;
            qci.flags = static_cast<VkDeviceQueueCreateFlags>(qparams.flags);
            qci.pQueuePriorities = qparams.priorities;
        }
        vk_createInfo.pQueueCreateInfos = queueCreateInfos.data();

        vk_createInfo.enabledExtensionCount = static_cast<uint32_t>(extensionStrings.size());
        vk_createInfo.ppEnabledExtensionNames = extensionStrings.data();

        if (!params.compilerSet)
            params.compilerSet = core::make_smart_refctd_ptr<asset::CCompilerSet>(core::smart_refctd_ptr(m_system));

        VkDevice vk_device = VK_NULL_HANDLE;
        if (vkCreateDevice(m_vkPhysicalDevice, &vk_createInfo, nullptr, &vk_device) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<CVulkanLogicalDevice>(core::smart_refctd_ptr<IAPIConnection>(m_api),m_rdoc_api,this,vk_device,m_vkInstance,params);
        }
        else
        {
            return nullptr;
        }    
    }

        inline bool isExtensionSupported(const char* name) const
        {
            return m_availableFeatureSet.find(name)!=m_availableFeatureSet.end();
        }

    private:
        renderdoc_api_t* const m_rdoc_api;
        const VkPhysicalDevice m_vkPhysicalDevice;
        const core::unordered_set<std::string> m_availableFeatureSet;
};

}

#endif
