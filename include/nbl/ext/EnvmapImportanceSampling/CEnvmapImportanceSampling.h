#ifndef _NBL_EXT_ENVMAP_IMPORTANCE_SAMPLING_INCLUDED_
#define _NBL_EXT_ENVMAP_IMPORTANCE_SAMPLING_INCLUDED_

#include "nbl/asset/IPipelineLayout.h"
#include "nbl/video/declarations.h"

namespace nbl::ext::envmap_importance_sampling
{

class EnvmapImportanceSampling final : public core::IReferenceCounted
{
  public:

		static constexpr uint32_t MaxMipCountLuminance = 13u;
		static constexpr uint32_t DefaultLumaMipMapGenWorkgroupDimension = 16u;
		static constexpr uint32_t DefaultWarpMapGenWorkgroupDimension = 16u;

    struct SCachedCreationParameters
    {
        core::smart_refctd_ptr<video::IUtilities> utilities;
        uint32_t genLumaMapWorkgroupDimension = DefaultLumaMipMapGenWorkgroupDimension;
        uint32_t genWarpMapWorkgroupDimension = DefaultWarpMapGenWorkgroupDimension;
    };

    struct SCreationParameters : public SCachedCreationParameters
    {
        core::smart_refctd_ptr<asset::IAssetManager> assetManager = nullptr;
        core::smart_refctd_ptr<video::IGPUImageView> envMap = nullptr;

        inline bool validate() const
        {
            const auto validation = std::to_array
            ({
                std::make_pair(bool(assetManager), "Invalid `creationParams.assetManager` is nullptr!"),
                std::make_pair(bool(utilities), "Invalid `creationParams.utilities` is nullptr!"),
                std::make_pair(bool(envMap), "Invalid `creationParams.envMap` is nullptr!"),
            });

            system::logger_opt_ptr logger = utilities->getLogger();
            for (const auto& [ok, error] : validation)
                if (!ok)
                {
                    logger.log(error, system::ILogger::ELL_ERROR);
                    return false;
                }

            assert(bool(assetManager->getSystem()));

            return true;
        }

    };

    static core::smart_refctd_ptr<EnvmapImportanceSampling> create(SCreationParameters&& params);

    static core::smart_refctd_ptr<video::IGPUPipelineLayout> createGenLumaPipelineLayout(video::ILogicalDevice* device);

    static core::smart_refctd_ptr<video::IGPUPipelineLayout> createMeasureLumaPipelineLayout(video::ILogicalDevice* device);

    static core::smart_refctd_ptr<video::IGPUPipelineLayout> createGenWarpMapPipelineLayout(video::ILogicalDevice* device);

    //! mounts the extension's archive to given system - useful if you want to create your own shaders with common header included
    static core::smart_refctd_ptr<system::IFileArchive> mount(core::smart_refctd_ptr<system::ILogger> logger, system::ISystem* system, video::ILogicalDevice* device, const std::string_view archiveAlias = "");

    static core::smart_refctd_ptr<video::IGPUComputePipeline> createGenLumaPipeline(const SCreationParameters& params, const video::IGPUPipelineLayout* pipelineLayout);

    static core::smart_refctd_ptr<video::IGPUComputePipeline> createMeasureLumaPipeline(const SCreationParameters& params, const video::IGPUPipelineLayout* pipelineLayout);

    static core::smart_refctd_ptr<video::IGPUImageView> createLumaMap(video::ILogicalDevice* device, asset::VkExtent3D extent, uint32_t mipCount, std::string_view debugName = "");

    static core::smart_refctd_ptr<video::IGPUImageView> createWarpMap(video::ILogicalDevice* device, asset::VkExtent3D extent, std::string_view debugName = "");

    bool computeWarpMap(video::IGPUCommandBuffer* cmdBuf, float envMapRegularizationFactor, float& pdfNormalizationFactor, float& maxEmittanceLuma);

		// returns if RIS should be enabled based on variance calculations
    inline bool computeWarpMap(video::IGPUCommandBuffer* cmdBuf, float envMapRegularizationFactor, float& pdfNormalizationFactor)
    {
      [[maybe_unused]] float dummy;
      return computeWarpMap(cmdBuf, envMapRegularizationFactor, pdfNormalizationFactor, dummy);
    }


    inline core::smart_refctd_ptr<video::IGPUImageView> getLumaMapView()
    {
      return m_lumaMap;
    }

  protected:
    struct ConstructorParams
    {
      SCachedCreationParameters creationParams;
      hlsl::uint32_t2 lumaWorkgroupSize;
      hlsl::uint32_t2 warpWorkgroupSize;
      core::smart_refctd_ptr<video::IGPUImageView> lumaMap;
      core::smart_refctd_ptr<video::IGPUImageView> warpMap;
      core::smart_refctd_ptr<video::IGPUComputePipeline> genLumaPipeline;
      core::smart_refctd_ptr<video::IGPUDescriptorSet> genLumaDescriptorSet;
    };

    explicit EnvmapImportanceSampling(ConstructorParams&& params) : 
      m_cachedCreationParams(std::move(params.creationParams)),
      m_lumaWorkgroupSize(params.lumaWorkgroupSize),
      m_warpWorkgroupSize(params.warpWorkgroupSize),
      m_lumaMap(std::move(params.lumaMap)),
      m_warpMap(std::move(params.warpMap)),
      m_genLumaPipeline(std::move(params.genLumaPipeline)), 
      m_genLumaDescriptorSet(std::move(params.genLumaDescriptorSet))
    {}

    ~EnvmapImportanceSampling() override {}

  private:

    SCachedCreationParameters m_cachedCreationParams;

    hlsl::uint32_t2 m_lumaWorkgroupSize;
    hlsl::uint32_t2 m_warpWorkgroupSize;

    core::smart_refctd_ptr<video::IGPUImageView> m_lumaMap;
    core::smart_refctd_ptr<video::IGPUImageView> m_warpMap;

    core::smart_refctd_ptr<video::IGPUComputePipeline> m_genLumaPipeline;
    core::smart_refctd_ptr<video::IGPUDescriptorSet> m_genLumaDescriptorSet;
  
};

}
#endif
