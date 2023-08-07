// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_ENVMAP_IMPORTANCE_SAMPLING_INCLUDED_
#define _NBL_EXT_ENVMAP_IMPORTANCE_SAMPLING_INCLUDED_

#include "nabla.h"
#include "nbl/video/IGPUShader.h"
#include "nbl/asset/ICPUShader.h"

namespace nbl
{
namespace ext
{
namespace EnvmapImportanceSampling
{

#include "nbl/builtin/glsl/ext/EnvmapImportanceSampling/parameters.glsl"
#undef uint
#undef vec4
#undef mat4
#undef mat4x3

class EnvmapImportanceSampling final : public core::IReferenceCounted
{
	public:
		EnvmapImportanceSampling(nbl::video::IVideoDriver* _driver);
		
		~EnvmapImportanceSampling() {}
		
		static constexpr uint32_t MaxMipCountLuminance = 13u;
		static constexpr uint32_t DefaultLumaMipMapGenWorkgroupDimension = 16u;
		static constexpr uint32_t DefaultWarpMapGenWorkgroupDimension = 16u;

		void initResources(
			nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> envmap, 
			uint32_t lumaMipMapGenWorkgroupDimension = DefaultLumaMipMapGenWorkgroupDimension,
			uint32_t warpMapGenWorkgroupDimension = DefaultWarpMapGenWorkgroupDimension);

		void deinitResources();

		// returns if RIS should be enabled based on variance calculations
		bool computeWarpMap(float envMapRegularizationFactor);

		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> getLuminanceImageView() { return m_luminanceBaseImageView; }
		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> getWarpMapImageView() { return m_warpMap; }
		
		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> m_luminanceBaseImageView;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> m_warpMap; // Warps Sample based on EnvMap Luminance
	private:
		
		nbl::core::smart_refctd_ptr<IGPUImageView> createTexture(uint32_t width, uint32_t height, nbl::asset::E_FORMAT format, uint32_t mipLevels = 1u, uint32_t layers = 1u);

		uint32_t m_mipCountEnvmap;
		uint32_t m_mipCountLuminance;
		uint32_t m_lumaMipMapGenWorkgroupDimension;
		uint32_t m_warpMapGenWorkgroupDimension;

		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> m_luminanceMipMaps[MaxMipCountLuminance];
		uint32_t m_lumaWorkGroups[2];
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> m_lumaDSLayout;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_lumaDS[MaxMipCountLuminance - 1];
		nbl::core::smart_refctd_ptr<nbl::video::IGPUPipelineLayout> m_lumaPipelineLayout;
		nbl::core::smart_refctd_ptr<IGPUSpecializedShader> m_lumaGPUShader;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUComputePipeline> m_lumaPipeline;

		// Shader and Resources for EnvironmentalMap Sample Warping

		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> m_warpDSLayout;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_warpDS;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUPipelineLayout> m_warpPipelineLayout;
		nbl::core::smart_refctd_ptr<IGPUSpecializedShader> m_warpGPUShader;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUComputePipeline> m_warpPipeline;
		
		nbl::video::IVideoDriver* m_driver;
};


}
}
}

#endif
