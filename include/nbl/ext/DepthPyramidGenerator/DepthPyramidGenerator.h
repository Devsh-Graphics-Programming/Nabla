// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_DEPTH_PYRAMID_GENERATOR_H_INCLUDED_
#define _NBL_EXT_DEPTH_PYRAMID_GENERATOR_H_INCLUDED_

#include "nabla.h"

#include "nbl/video/IGPUImageView.h"
#include "nbl/asset/format/EFormat.h"
#include "../../../../Nabla/source/Nabla/COpenGLExtensionHandler.h"
#include "nbl/builtin/glsl/ext/DepthPyramidGenerator/push_constants_struct_common.h"

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

// TODO: test: `E_MIPMAP_GENERATION_OPERATOR::BOTH`, `reundUpToPoTWithPadding = true`

class DepthPyramidGenerator final
{
public:

	enum class E_MIPMAP_GENERATION_OPERATOR
	{
		EMGO_MAX, 
		EMGO_MIN,
		EMGO_BOTH // min goes to r, max to g
	};
	
	enum class E_WORK_GROUP_SIZE
	{
		EWGS_16x16x1 = 16u,
		EWGS_32x32x1 = 32u
	};

	// TODO: more formats
	enum class E_IMAGE_FORMAT
	{
		EIF_R16_FLOAT = EF_R16_SFLOAT,
		EIF_R32_FLOAT = EF_R32_SFLOAT,
		EIF_R16G16_FLOAT = EF_R16G16_SFLOAT,
		EIF_R32G32_FLOAT = EF_R32G32_SFLOAT
	};

	struct Config
	{
		E_WORK_GROUP_SIZE workGroupSize = E_WORK_GROUP_SIZE::EWGS_32x32x1;
		E_IMAGE_FORMAT outputFormat = E_IMAGE_FORMAT::EIF_R32_FLOAT;
		E_MIPMAP_GENERATION_OPERATOR op = E_MIPMAP_GENERATION_OPERATOR::EMGO_MAX;
		uint32_t lvlLimit = 0u; //no limit when set to 0 (full mip chain)
		bool roundUpToPoTWithPadding = false;
	};

	struct DispatchData
	{
		nbl_glsl_depthPyramid_PushConstantsData pcData;
		core::vector2d<uint32_t> globalWorkGroupSize;
	};

	// inputDepthImageView - input texture
	//outputDepthPyramidMips - array of created mipMaps
	DepthPyramidGenerator(IVideoDriver* driver, IAssetManager* am, core::smart_refctd_ptr<IGPUImageView> inputDepthImageView,
		const Config& config = Config());

	static inline uint32_t getMaxMipCntFromImage(const core::smart_refctd_ptr<IGPUImageView>& image, bool roundUpToPoTWithPadding = false)
	{
		const VkExtent3D lvl0MipExtent = calcLvl0MipExtent(image->getCreationParameters().image->getCreationParameters().extent, roundUpToPoTWithPadding);

		// TODO: take `roundUpToPoTWithPadding` into account
		return core::findMSB(std::min(lvl0MipExtent.width, lvl0MipExtent.height)) + 1u;
	}

	static uint32_t createMipMapImageViews(IVideoDriver* driver, core::smart_refctd_ptr<IGPUImageView> inputDepthImageView, core::smart_refctd_ptr<IGPUImageView>* outputDepthPyramidMips, const Config& config = Config());

	static uint32_t createDescriptorSets(IVideoDriver* driver, core::smart_refctd_ptr<IGPUImageView> inputDepthImageView, core::smart_refctd_ptr<IGPUImageView>* inputDepthPyramidMips, 
		core::smart_refctd_ptr<IGPUDescriptorSetLayout>& outputDsLayout, core::smart_refctd_ptr<IGPUDescriptorSet>* outputDs, DispatchData* outputDispatchData, const Config& config = Config());

	void createPipeline(IVideoDriver* driver, core::smart_refctd_ptr<IGPUDescriptorSetLayout>& dsLayout, core::smart_refctd_ptr<IGPUComputePipeline>& outputPpln);

	void generateMipMaps(const core::smart_refctd_ptr<IGPUImageView>& inputImage, core::smart_refctd_ptr<IGPUComputePipeline>& ppln, core::smart_refctd_ptr<IGPUDescriptorSet>& ds, const DispatchData& dispatchData, bool issueDefaultBarrier = true);

	static inline void defaultBarrier()
	{
		COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT); // GL_FRAMEBUFFER_BARRIER_BIT ?
	}

private:
	static inline VkExtent3D calcLvl0MipExtent(const VkExtent3D& sourceImageExtent, bool roundUpToPoTWithPadding)
	{
		VkExtent3D lvl0MipExtent;

		lvl0MipExtent.width = core::roundUpToPoT(sourceImageExtent.width);
		lvl0MipExtent.height = core::roundUpToPoT(sourceImageExtent.height);

		if (!roundUpToPoTWithPadding)
		{
			if (!core::isPoT(sourceImageExtent.width))
				lvl0MipExtent.width >>= 1u;
			if (!core::isPoT(sourceImageExtent.height))
				lvl0MipExtent.height >>= 1u;
		}

		return lvl0MipExtent;
	}

private:
	IVideoDriver* m_driver;

	const Config m_config;
	core::smart_refctd_ptr<IGPUSpecializedShader> m_shader = nullptr;

	static constexpr uint32_t maxMipLvlsPerDispatch = 8u;

};

}
}
}

#endif