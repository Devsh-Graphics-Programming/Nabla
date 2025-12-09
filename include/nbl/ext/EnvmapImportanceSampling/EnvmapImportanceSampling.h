// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_ENVMAP_IMPORTANCE_SAMPLING_INCLUDED_
#define _NBL_EXT_ENVMAP_IMPORTANCE_SAMPLING_INCLUDED_

#include "nabla.h"
#include "nbl/video/IGPUShader.h"
#include "nbl/asset/ICPUShader.h"

namespace nbl::ext::EnvmapImportanceSampling
{

class EnvmapImportanceSampling
{
	public:
		EnvmapImportanceSampling(video::IVideoDriver* _driver) : m_driver(_driver) 
		{}
		~EnvmapImportanceSampling() = default;

		// Shader and Resources for Generating Luminance MipMaps from EnvMap
		static constexpr uint32_t MaxMipCountLuminance = 13u;
		static constexpr uint32_t DefaultLumaMipMapGenWorkgroupDimension = 16u;
		static constexpr uint32_t DefaultWarpMapGenWorkgroupDimension = 16u;
	
		void initResources(
			core::smart_refctd_ptr<video::IGPUImageView> envmap, 
			uint32_t lumaGenWorkgroupDimension = DefaultLumaMipMapGenWorkgroupDimension,
			uint32_t warpMapGenWorkgroupDimension = DefaultWarpMapGenWorkgroupDimension);
		void deinitResources();

		// returns if RIS should be enabled based on variance calculations
		inline bool computeWarpMap(const float envMapRegularizationFactor, float& pdfNormalizationFactor)
		{
			[[maybe_unused]] float dummy;
			return computeWarpMap(envMapRegularizationFactor,pdfNormalizationFactor,dummy);
		}
		bool computeWarpMap(const float envMapRegularizationFactor, float& pdfNormalizationFactor, float& maxEmittanceLuma);
	
		core::smart_refctd_ptr<video::IGPUImageView> getLuminanceImageView() { return m_luminance; }
		core::smart_refctd_ptr<video::IGPUImageView> getWarpMapImageView() { return m_warpMap; }

	private:
		#define uint uint32_t
		struct uvec2
		{
			uint x,y;
		};
		struct vec2
		{
			float x,y;
		};
		struct vec3
		{
			float x,y,z;
		};
		#define vec4 core::vectorSIMDf
		#define mat4 core::matrix4SIMD
		#define mat4x3 core::matrix3x4SIMD
		#include "nbl/builtin/glsl/ext/EnvmapImportanceSampling/structs.glsl"
		#undef uint
		#undef vec4
		#undef mat4
		#undef mat4x3
		inline uint32_t calcMeasurementBufferSize() const
		{
			return sizeof(nbl_glsl_ext_EnvmapSampling_LumaMeasurement_t)*m_lumaWorkgroups[0]*m_lumaWorkgroups[1];
		}
		#undef NBL_GLSL_EXT_ENVMAP_SAMPLING_LUMA_MEASUREMENTS

		uint32_t m_lumaWorkgroups[2];
		uint32_t m_warpWorkgroups[2];

		core::smart_refctd_ptr<video::IGPUImageView> m_luminance;
		core::smart_refctd_ptr<video::IGPUImageView> m_warpMap; // Warps Sample based on EnvMap Luminance

		core::smart_refctd_ptr<video::IGPUDescriptorSet> m_lumaDS;
		core::smart_refctd_ptr<video::IGPUComputePipeline> m_lumaMeasurePipeline;
		core::smart_refctd_ptr<video::IGPUComputePipeline> m_lumaGenPipeline;

		// Shader and Resources for EnvironmentalMap Sample Warping
		core::smart_refctd_ptr<video::IGPUDescriptorSet> m_warpDS;
		core::smart_refctd_ptr<video::IGPUSpecializedShader> m_warpGPUShader;
		core::smart_refctd_ptr<video::IGPUComputePipeline> m_warpPipeline;
	
		video::IVideoDriver* m_driver;
};

}

#endif
