#ifndef _NBL_VIDEO_ENVMAP_WARP_GENERATOR_INCLUDED_
#define _NBL_VIDEO_ENVMAP_WARP_GENERATOR_INCLUDED_

#include "nbl/video/declarations.h"

namespace nbl::video
{

class NBL_API2 CEnvmapWarpGenerator final : public core::IReferenceCounted
{
	public:

		static constexpr uint32_t MaxMipCountLuminance = 13u;

		struct SCachedCreationParameters
		{
				core::smart_refctd_ptr<video::IUtilities> utilities;
		};

		struct SCreationParameters : public SCachedCreationParameters
		{
				core::smart_refctd_ptr<asset::IAssetManager> assetManager = nullptr;

				inline bool validate() const
				{
						const auto validation = std::to_array
						({
								std::make_pair(bool(assetManager), "Invalid `creationParams.assetManager` is nullptr!"),
								std::make_pair(bool(utilities), "Invalid `creationParams.utilities` is nullptr!"),
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


		static core::smart_refctd_ptr<CEnvmapWarpGenerator> create(SCreationParameters&& params);

		static core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createDescriptorSetLayout(video::ILogicalDevice* device);
		static core::smart_refctd_ptr<video::IGPUPipelineLayout> createPipelineLayout(video::ILogicalDevice* device);

		static core::smart_refctd_ptr<video::IGPUComputePipeline> createPipeline(const SCreationParameters& params, const video::IGPUPipelineLayout* pipelineLayout, std::string_view shaderPath);

		
		static core::smart_refctd_ptr<video::IGPUImageView> createLumaMap(video::ILogicalDevice* device, asset::VkExtent3D extent, uint32_t mipCount, uint32_t layerCount, const char* debugName = "");

		static core::smart_refctd_ptr<video::IGPUImageView> createWarpMap(video::ILogicalDevice* device, asset::VkExtent3D extent, uint32_t layerCount, const char* debugName = "");

    inline video::IGPUComputePipeline* getGenLumaPipeline() const
    {
			return m_genLumaPipeline.get();
    } 

    inline video::IGPUComputePipeline* getGenWarpPipeline() const
    {
			return m_genWarpPipeline.get();
    } 

    class NBL_API2 SSession : public core::IReferenceCounted
    {
        public:

					  // ASK(kevin): Should this and constructor be private and we use friend class?
            struct SCachedCreationParams
            {
								core::smart_refctd_ptr<video::IGPUImageView> envMap;
                core::smart_refctd_ptr<video::IGPUImageView> lumaMap;
                core::smart_refctd_ptr<video::IGPUImageView> warpMap;
                core::smart_refctd_ptr<video::IGPUDescriptorSet> descriptorSet;
                core::smart_refctd_ptr<CEnvmapWarpGenerator> generator;
                hlsl::uint32_t2 genLumaWorkgroupCount;
                hlsl::uint32_t2 genWarpWorkgroupCount;
								uint16_t layerCount;
            };

						explicit SSession(SCachedCreationParams&& params) : m_params(std::move(params)) {}

            void computeWarpMap(video::IGPUCommandBuffer* cmdBuf);

            inline core::smart_refctd_ptr<video::IGPUImageView> getLumaMapView() const
            {
              return m_params.lumaMap;
            }

            inline core::smart_refctd_ptr<video::IGPUImageView> getWarpMapView() const
            {
              return m_params.warpMap;
            }

            using image_barrier_t = IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t;
            // barrier against previous uses of the envmap. Don't access luma map and warp map before calling computeWarpMap
            image_barrier_t getEnvMapPrevBarrier(core::bitflag<asset::PIPELINE_STAGE_FLAGS> srcStageMask, core::bitflag<asset::ACCESS_FLAGS> srcAccessMask, IGPUImage::LAYOUT oldLayout);

						image_barrier_t getEnvMapNextBarrier(core::bitflag<asset::PIPELINE_STAGE_FLAGS> dstStageMask, core::bitflag<asset::ACCESS_FLAGS> dstAccessMask, IGPUImage::LAYOUT newLayout);

            // barrier against future uses for luma map and warp map.
            std::array<image_barrier_t, 2> getOutputMapNextBarrier(core::bitflag<asset::PIPELINE_STAGE_FLAGS> dstStageMask, core::bitflag<asset::ACCESS_FLAGS> dstAccessMask, IGPUImage::LAYOUT newLayout);

        private:
					SCachedCreationParams m_params;
    };

		core::smart_refctd_ptr<SSession> createSession(core::smart_refctd_ptr<IGPUImageView>&& envMap, uint16_t upscaleLog2 = 0);

	protected:
		struct ConstructorParams
		{
			SCachedCreationParameters creationParams;
			core::smart_refctd_ptr<video::IGPUComputePipeline> genLumaPipeline;
			core::smart_refctd_ptr<video::IGPUComputePipeline> genWarpPipeline;
		};

		explicit CEnvmapWarpGenerator(ConstructorParams&& params) : 
			m_params(std::move(params.creationParams)),
			m_genLumaPipeline(std::move(params.genLumaPipeline)), 
			m_genWarpPipeline(std::move(params.genWarpPipeline))
		{}

		~CEnvmapWarpGenerator() override {}

	private:

		SCachedCreationParameters m_params;

		core::smart_refctd_ptr<video::IGPUComputePipeline> m_genLumaPipeline;
		core::smart_refctd_ptr<video::IGPUComputePipeline> m_genWarpPipeline;
	
};

}
#endif
