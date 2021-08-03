// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_DEPTH_PYRAMID_GENERATOR_H_INCLUDED_
#define _NBL_EXT_DEPTH_PYRAMID_GENERATOR_H_INCLUDED_

#include "nabla.h"

#include "nbl/video/IGPUImageView.h"
#include "nbl/asset/format/EFormat.h"

using namespace nbl;
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::video;

namespace nbl
{
namespace ext
{
namespace DepthPyramidGenerator
{

class DepthPyramidGenerator final
{
public:

	enum class E_MIPMAP_GENERATION_OPERATOR
	{
		MAX, 
		MIN,
		BOTH // min goes to r, min to g
	};
	
	struct Config
	{
		asset::E_FORMAT outputFormat = asset::E_FORMAT::EF_R32_SFLOAT;
		E_MIPMAP_GENERATION_OPERATOR op = E_MIPMAP_GENERATION_OPERATOR::MAX;
		uint32_t lvlLimit = 0u; //no limit when set to 0 (full mip chain)
		bool roundUpToPoTWithPadding = false;
	};

	// inputDepthImageView - input texture
	//outputDepthPyramidMips - array of created mipMaps
	DepthPyramidGenerator(IVideoDriver* driver, IAssetManager* am, core::smart_refctd_ptr<IGPUImageView> inputDepthImageView,
		core::smart_refctd_ptr<IGPUImageView>* outputDepthPyramidMips,
		const Config& config = Config());

	inline uint32_t getMaxMipCntFromImage(const core::smart_refctd_ptr<IGPUImageView>& image, bool roundUpToPoTWithPadding = false)
	{
		const VkExtent3D lvl0MipExtent = calcLvl0MipExtent(image->getCreationParameters().image->getCreationParameters().extent, roundUpToPoTWithPadding);

		uint32_t minVal = std::min(lvl0MipExtent.width, lvl0MipExtent.height);

		uint32_t mipLvlCnt = 1u;
		while (minVal >>= 1u)
			mipLvlCnt++;

		return mipLvlCnt;
	}

	void generateMipMaps();

private:
	inline VkExtent3D calcLvl0MipExtent(const VkExtent3D& sourceImageExtent, bool roundUpToPoTWithPadding)
	{
		VkExtent3D lvl0MipExtent;

		lvl0MipExtent.width = core::roundUpToPoT(sourceImageExtent.width);
		lvl0MipExtent.height = core::roundUpToPoT(sourceImageExtent.height);

		if (!roundUpToPoTWithPadding)
		{
			lvl0MipExtent.width >>= 1u;
			lvl0MipExtent.height >>= 1u;
		}

		return lvl0MipExtent;
	}

	void configureMipImages(core::smart_refctd_ptr<IGPUImageView> inputDepthImageView, core::smart_refctd_ptr<IGPUImageView>* outputDepthPyramidMips, const Config& config);

private:
	IVideoDriver* m_driver;

	core::smart_refctd_ptr<IGPUDescriptorSet> m_ds = nullptr;
	core::smart_refctd_ptr<IGPUComputePipeline> m_ppln = nullptr;
	vector2df m_globalWorkGroupSize;
};

}
}
}

#endif