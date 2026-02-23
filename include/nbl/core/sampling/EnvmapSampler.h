#ifndef _NBL_CORE_ENVMAP_SAMPLER_INCLUDED_
#define _NBL_CORE_ENVMAP_SAMPLER_INCLUDED_

#include "nbl/video/declarations.h"

namespace nbl::core
{

class NBL_API2 EnvmapSampler final : public core::IReferenceCounted
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

		static core::smart_refctd_ptr<EnvmapSampler> create(SCreationParameters&& params);

		static core::smart_refctd_ptr<video::IGPUPipelineLayout> createGenLumaPipelineLayout(video::ILogicalDevice* device);

		static core::smart_refctd_ptr<video::IGPUPipelineLayout> createGenWarpPipelineLayout(video::ILogicalDevice* device);

		//! mounts the extension's archive to given system - useful if you want to create your own shaders with common header included
		static core::smart_refctd_ptr<system::IFileArchive> mount(core::smart_refctd_ptr<system::ILogger> logger, system::ISystem* system, video::ILogicalDevice* device, const std::string_view archiveAlias = "");

		static core::smart_refctd_ptr<video::IGPUComputePipeline> createGenLumaPipeline(const SCreationParameters& params, const video::IGPUPipelineLayout* pipelineLayout);

		static core::smart_refctd_ptr<video::IGPUComputePipeline> createGenWarpPipeline(const SCreationParameters& params, const video::IGPUPipelineLayout* pipelineLayout);

		static core::smart_refctd_ptr<video::IGPUImageView> createLumaMap(video::ILogicalDevice* device, asset::VkExtent3D extent, uint32_t mipCount, std::string_view debugName = "");

		static core::smart_refctd_ptr<video::IGPUImageView> createWarpMap(video::ILogicalDevice* device, asset::VkExtent3D extent, std::string_view debugName = "");

		void computeWarpMap(video::IQueue* queue);

		// use this to synchronize warp map after computeWarpMap call
		nbl::video::IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t getWarpMapBarrier(
			core::bitflag<nbl::asset::PIPELINE_STAGE_FLAGS> dstStageMask,
			core::bitflag<nbl::asset::ACCESS_FLAGS> dstAccessMask,
			nbl::video::IGPUImage::LAYOUT oldLayout);

		// use this to synchronize luma map after computeWarpMap call
		nbl::video::IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t getLumaMapBarrier(
			core::bitflag<nbl::asset::PIPELINE_STAGE_FLAGS> dstStageMask,
			core::bitflag<nbl::asset::ACCESS_FLAGS> dstAccessMask,
			nbl::video::IGPUImage::LAYOUT oldLayout);

		inline core::smart_refctd_ptr<video::IGPUImageView> getLumaMapView() const
		{
			return m_lumaMap;
		}

		inline core::smart_refctd_ptr<video::IGPUImageView> getWarpMapView() const
		{
			return m_warpMap;
		}

		inline hlsl::float32_t getAvgLuma() const
		{
			return m_avgLuma;
		}

	protected:
		struct ConstructorParams
		{
			SCachedCreationParameters creationParams;
			hlsl::uint32_t2 lumaWorkgroupCount;
			hlsl::uint32_t2 warpWorkgroupCount;
			core::smart_refctd_ptr<video::IGPUImageView> lumaMap;
			core::smart_refctd_ptr<video::IGPUImageView> warpMap;
			core::smart_refctd_ptr<video::IGPUComputePipeline> genLumaPipeline;
			core::smart_refctd_ptr<video::IGPUDescriptorSet> genLumaDescriptorSet;
			core::smart_refctd_ptr<video::IGPUComputePipeline> genWarpPipeline;
			core::smart_refctd_ptr<video::IGPUDescriptorSet> genWarpDescriptorSet;
		};

		explicit EnvmapSampler(ConstructorParams&& params) : 
			m_cachedCreationParams(std::move(params.creationParams)),
			m_lumaWorkgroupCount(params.lumaWorkgroupCount),
			m_warpWorkgroupCount(params.warpWorkgroupCount),
			m_lumaMap(std::move(params.lumaMap)),
			m_warpMap(std::move(params.warpMap)),
			m_genLumaPipeline(std::move(params.genLumaPipeline)), 
			m_genLumaDescriptorSet(std::move(params.genLumaDescriptorSet)),
			m_genWarpPipeline(std::move(params.genWarpPipeline)), 
			m_genWarpDescriptorSet(std::move(params.genWarpDescriptorSet))
		{}

		~EnvmapSampler() override {}

	private:

		SCachedCreationParameters m_cachedCreationParams;

		hlsl::uint32_t2 m_lumaWorkgroupCount;
		hlsl::uint32_t2 m_warpWorkgroupCount;

		hlsl::float32_t m_avgLuma;

		core::smart_refctd_ptr<video::IGPUImageView> m_lumaMap;
		core::smart_refctd_ptr<video::IGPUImageView> m_warpMap;

		core::smart_refctd_ptr<video::IGPUComputePipeline> m_genLumaPipeline;
		core::smart_refctd_ptr<video::IGPUDescriptorSet> m_genLumaDescriptorSet;

		core::smart_refctd_ptr<video::IGPUComputePipeline> m_genWarpPipeline;
		core::smart_refctd_ptr<video::IGPUDescriptorSet> m_genWarpDescriptorSet;
	
};

}
#endif
