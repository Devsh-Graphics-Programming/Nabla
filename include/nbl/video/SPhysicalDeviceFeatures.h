#ifndef _NBL_VIDEO_S_PHYSICAL_DEVICE_FEATURES_H_INCLUDED_
#define _NBL_VIDEO_S_PHYSICAL_DEVICE_FEATURES_H_INCLUDED_

namespace nbl::video
{

struct SPhysicalDeviceFeatures
{
    /* Vulkan 1.0 Core  */
    bool robustBufferAccess = false;
    //VkBool32    fullDrawIndexUint32;
    bool imageCubeArray = false;
    //VkBool32    independentBlend;
    bool geometryShader    = false;
    //VkBool32    tessellationShader;
    //VkBool32    sampleRateShading;
    //VkBool32    dualSrcBlend;
    bool logicOp = false;
    bool multiDrawIndirect = false;
    //VkBool32    drawIndirectFirstInstance;
    //VkBool32    depthClamp;
    //VkBool32    depthBiasClamp;
    //VkBool32    fillModeNonSolid;
    //VkBool32    depthBounds;
    //VkBool32    wideLines;
    //VkBool32    largePoints;
    //VkBool32    alphaToOne;
    bool multiViewport = false;
    bool samplerAnisotropy = false;
    //VkBool32    textureCompressionETC2;
    //VkBool32    textureCompressionASTC_LDR;
    //VkBool32    textureCompressionBC;
    //VkBool32    occlusionQueryPrecise;
    //VkBool32    pipelineStatisticsQuery;
    //VkBool32    vertexPipelineStoresAndAtomics;
    //VkBool32    fragmentStoresAndAtomics;
    //VkBool32    shaderTessellationAndGeometryPointSize;
    //VkBool32    shaderImageGatherExtended;
    //VkBool32    shaderStorageImageExtendedFormats;
    //VkBool32    shaderStorageImageMultisample;
    //VkBool32    shaderStorageImageReadWithoutFormat;
    //VkBool32    shaderStorageImageWriteWithoutFormat;
    //VkBool32    shaderUniformBufferArrayDynamicIndexing;
    //VkBool32    shaderSampledImageArrayDynamicIndexing;
    //VkBool32    shaderStorageBufferArrayDynamicIndexing;
    //VkBool32    shaderStorageImageArrayDynamicIndexing;
    //VkBool32    shaderClipDistance;
    //VkBool32    shaderCullDistance;
    bool vertexAttributeDouble = false; // shaderFloat64
    //VkBool32    shaderInt64;
    //VkBool32    shaderInt16;
    //VkBool32    shaderResourceResidency;
    //VkBool32    shaderResourceMinLod;
    //VkBool32    sparseBinding;
    //VkBool32    sparseResidencyBuffer;
    //VkBool32    sparseResidencyImage2D;
    //VkBool32    sparseResidencyImage3D;
    //VkBool32    sparseResidency2Samples;
    //VkBool32    sparseResidency4Samples;
    //VkBool32    sparseResidency8Samples;
    //VkBool32    sparseResidency16Samples;
    //VkBool32    sparseResidencyAliased;
    //VkBool32    variableMultisampleRate;
    bool inheritedQueries = false;
                
    /* FragmentDensityMapFeaturesEXT */
    //VkBool32           fragmentDensityMap;
    //VkBool32           fragmentDensityMapDynamic;
    //VkBool32           fragmentDensityMapNonSubsampledImages;

    /* FragmentDensityMap2FeaturesEXT */
    //VkBool32           fragmentDensityMapDeferred;

    /* RayQueryFeaturesKHR */
    bool rayQuery = false;
            
    /* AccelerationStructureFeaturesKHR */
    bool accelerationStructure = false;
    bool accelerationStructureCaptureReplay = false;
    bool accelerationStructureIndirectBuild = false;
    bool accelerationStructureHostCommands = false;
    bool descriptorBindingAccelerationStructureUpdateAfterBind = false;
            
    /* RayTracingPipelineFeaturesKHR */
    bool rayTracingPipeline = false;
    bool rayTracingPipelineShaderGroupHandleCaptureReplay = false;
    bool rayTracingPipelineShaderGroupHandleCaptureReplayMixed = false;
    bool rayTracingPipelineTraceRaysIndirect = false;
    bool rayTraversalPrimitiveCulling = false;
            
    /* FragmentShaderInterlockFeaturesEXT */
    bool fragmentShaderSampleInterlock = false;
    bool fragmentShaderPixelInterlock = false;
    bool fragmentShaderShadingRateInterlock = false;
            
    /* BufferDeviceAddressFeaturesKHR */
    bool bufferDeviceAddress = false;
    //VkBool32           bufferDeviceAddressCaptureReplay;
    //VkBool32           bufferDeviceAddressMultiDevice;
            
    bool drawIndirectCount = false; // TODO(Erfan): Move in 1.2 features
            
    /* Nabla */
    bool dispatchBase = false; // true in Vk, false in GL
    bool allowCommandBufferQueryCopies = false;
};

} // nbl::video

#endif
