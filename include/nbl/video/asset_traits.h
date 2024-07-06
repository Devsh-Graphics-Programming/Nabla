// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_ASSET_TRAITS_H_INCLUDED_
#define _NBL_VIDEO_ASSET_TRAITS_H_INCLUDED_

#include "nbl/asset/ICPUShader.h"
#include "nbl/video/IGPUShader.h"
#include "nbl/asset/ICPUBufferView.h"
#include "nbl/video/IGPUBufferView.h"
#include "nbl/asset/ICPUDescriptorSet.h"
#include "nbl/video/IGPUDescriptorSet.h"
#include "nbl/asset/ICPUComputePipeline.h"
#include "nbl/video/IGPUComputePipeline.h"
#include "nbl/asset/ICPUGraphicsPipeline.h"
#include "nbl/video/IGPUGraphicsPipeline.h"
#include "nbl/asset/ICPUSampler.h"
#include "nbl/video/IGPUSampler.h"
#include "nbl/asset/ICPUImageView.h"
#include "nbl/video/IGPUImageView.h"
#include "nbl/asset/ICPUAccelerationStructure.h"
#include "nbl/video/IGPUAccelerationStructure.h"


namespace nbl::video
{
template<asset::Asset AssetType>
struct asset_traits;

//! Pipelines
template<>
struct asset_traits<asset::ICPUShader>
{
	// the asset type
	using asset_t = asset::ICPUShader;
	// the video type
	using video_t = IGPUShader;
};

template<>
struct asset_traits<asset::ICPUDescriptorSetLayout>
{
	// the asset type
	using asset_t = asset::ICPUDescriptorSetLayout;
	// the video type
	using video_t = IGPUDescriptorSetLayout;
};

template<>
struct asset_traits<asset::ICPUPipelineLayout>
{
	// the asset type
	using asset_t = asset::ICPUPipelineLayout;
	// the video type
	using video_t = IGPUPipelineLayout;
};
/*
/*
template<>
struct asset_traits<asset::ICPUDescriptorSetLayout> { using GPUObjectType = IGPUDescriptorSetLayout; };

template<>
struct asset_traits<asset::ICPUComputePipeline> { using GPUObjectType = IGPUComputePipeline; };
*/

template<>
struct asset_traits<asset::ICPUBuffer>
{
	using asset_t = asset::ICPUBuffer;
	using video_t = asset::SBufferRange<IGPUBuffer>;
};

template<>
struct asset_traits<asset::ICPUBufferView>
{
	// the asset type
	using asset_t = asset::ICPUBufferView;
	// the video type
	using video_t = IGPUBufferView;
};


/*
template<>
struct asset_traits<asset::ICPUImage> { using GPUObjectType = IGPUImage; };

template<>
struct asset_traits<asset::ICPUImageView> { using GPUObjectType = IGPUImageView; };
*/

template<>
struct asset_traits<asset::ICPUSampler>
{
	// the asset type
	using asset_t = asset::ICPUSampler;
	// the video type
	using video_t = IGPUSampler;
};

/*
template<>
struct asset_traits<asset::ICPUShader> { using GPUObjectType = IGPUShader; };

template<>
struct asset_traits<asset::ICPUDescriptorSet> { using GPUObjectType = IGPUDescriptorSet; };

template<>
struct asset_traits<asset::ICPUAccelerationStructure> { using GPUObjectType = IGPUAccelerationStructure; };
*/
}
#endif