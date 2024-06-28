   // VK 1.0 Core
    res.robustBufferAccess &= _rhs.robustBufferAccess;

    res.geometryShader &= _rhs.geometryShader;
    res.tessellationShader &= _rhs.tessellationShader;

    res.depthBounds &= _rhs.depthBounds;

    res.wideLines &= _rhs.wideLines;
    res.largePoints &= _rhs.largePoints;

    res.alphaToOne &= _rhs.alphaToOne;

    res.pipelineStatisticsQuery &= _rhs.pipelineStatisticsQuery;

    res.shaderCullDistance &= _rhs.shaderCullDistance;

    res.shaderResourceResidency &= _rhs.shaderResourceResidency;
    res.shaderResourceMinLod &= _rhs.shaderResourceMinLod;

   // VK 1.1 Everything is either a Limit or Required
   // VK 1.2
    res.bufferDeviceAddressMultiDevice &= _rhs.bufferDeviceAddressMultiDevice;

   // VK 1.3
    res.robustImageAccess &= _rhs.robustImageAccess;

   // Nabla Core Extensions
    res.robustBufferAccess2 &= _rhs.robustBufferAccess2;
    res.robustImageAccess2 &= _rhs.robustImageAccess2;
    res.nullDescriptor &= _rhs.nullDescriptor;

   // Extensions
    res.swapchainMode &= _rhs.swapchainMode;

    res.shaderInfoAMD &= _rhs.shaderInfoAMD;

    res.conditionalRendering &= _rhs.conditionalRendering;
    res.inheritedConditionalRendering &= _rhs.inheritedConditionalRendering;

    res.geometryShaderPassthrough &= _rhs.geometryShaderPassthrough;

    res.hdrMetadata &= _rhs.hdrMetadata;

    res.performanceCounterQueryPools &= _rhs.performanceCounterQueryPools;
    res.performanceCounterMultipleQueryPools &= _rhs.performanceCounterMultipleQueryPools;

    res.mixedAttachmentSamples &= _rhs.mixedAttachmentSamples;

    res.accelerationStructure &= _rhs.accelerationStructure;
    res.accelerationStructureIndirectBuild &= _rhs.accelerationStructureIndirectBuild;
    res.accelerationStructureHostCommands &= _rhs.accelerationStructureHostCommands;

    res.rayTracingPipeline &= _rhs.rayTracingPipeline;
    res.rayTraversalPrimitiveCulling &= _rhs.rayTraversalPrimitiveCulling;

    res.rayQuery &= _rhs.rayQuery;

    res.representativeFragmentTest &= _rhs.representativeFragmentTest;

    res.bufferMarkerAMD &= _rhs.bufferMarkerAMD;

    res.fragmentDensityMap &= _rhs.fragmentDensityMap;
    res.fragmentDensityMapDynamic &= _rhs.fragmentDensityMapDynamic;
    res.fragmentDensityMapNonSubsampledImages &= _rhs.fragmentDensityMapNonSubsampledImages;

    res.deviceCoherentMemory &= _rhs.deviceCoherentMemory;

    res.memoryPriority &= _rhs.memoryPriority;

    res.fragmentShaderSampleInterlock &= _rhs.fragmentShaderSampleInterlock;
    res.fragmentShaderPixelInterlock &= _rhs.fragmentShaderPixelInterlock;
    res.fragmentShaderShadingRateInterlock &= _rhs.fragmentShaderShadingRateInterlock;

    res.rectangularLines &= _rhs.rectangularLines;
    res.bresenhamLines &= _rhs.bresenhamLines;
    res.smoothLines &= _rhs.smoothLines;
    res.stippledRectangularLines &= _rhs.stippledRectangularLines;
    res.stippledBresenhamLines &= _rhs.stippledBresenhamLines;
    res.stippledSmoothLines &= _rhs.stippledSmoothLines;

    res.indexTypeUint8 &= _rhs.indexTypeUint8;

    res.deferredHostOperations &= _rhs.deferredHostOperations;

    res.pipelineExecutableInfo &= _rhs.pipelineExecutableInfo;

    res.deviceGeneratedCommands &= _rhs.deviceGeneratedCommands;

    res.rayTracingMotionBlur &= _rhs.rayTracingMotionBlur;
    res.rayTracingMotionBlurPipelineTraceRaysIndirect &= _rhs.rayTracingMotionBlurPipelineTraceRaysIndirect;

    res.fragmentDensityMapDeferred &= _rhs.fragmentDensityMapDeferred;

    res.rasterizationOrderColorAttachmentAccess &= _rhs.rasterizationOrderColorAttachmentAccess;
    res.rasterizationOrderDepthAttachmentAccess &= _rhs.rasterizationOrderDepthAttachmentAccess;
    res.rasterizationOrderStencilAttachmentAccess &= _rhs.rasterizationOrderStencilAttachmentAccess;

    res.cooperativeMatrixRobustBufferAccess &= _rhs.cooperativeMatrixRobustBufferAccess;

