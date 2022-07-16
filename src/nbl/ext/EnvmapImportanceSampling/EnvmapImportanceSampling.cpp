// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/EnvmapImportanceSampling/EnvmapImportanceSampling.h"

#include <cstdio>

using namespace nbl;
using namespace nbl::asset;
using namespace nbl::video;
using namespace ext::EnvmapImportanceSampling;

void getEnvmapResolutionFromMipLevel(uint32_t level, uint32_t& outWidth, uint32_t& outHeight)
{
	const uint32_t resolution = 0x1u<<(level);
	outWidth = std::max(resolution, 1u);
	outHeight = std::max(resolution/2u, 1u);
}

static core::smart_refctd_ptr<IGPUImageView> createTexture(nbl::video::IVideoDriver* _driver, uint32_t width, uint32_t height, E_FORMAT format, uint32_t mipLevels=1u, uint32_t layers=0u)
{
	const auto real_layers = layers ? layers:1u;

	IGPUImage::SCreationParams imgparams;
	imgparams.extent = {width, height, 1u};
	imgparams.arrayLayers = real_layers;
	imgparams.flags = static_cast<IImage::E_CREATE_FLAGS>(0);
	imgparams.format = format;
	imgparams.mipLevels = mipLevels;
	imgparams.samples = IImage::ESCF_1_BIT;
	imgparams.type = IImage::ET_2D;

	IGPUImageView::SCreationParams viewparams;
	viewparams.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0);
	viewparams.format = format;
	viewparams.image = _driver->createDeviceLocalGPUImageOnDedMem(std::move(imgparams));
	viewparams.viewType = layers ? IGPUImageView::ET_2D_ARRAY:IGPUImageView::ET_2D;
	viewparams.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0);
	viewparams.subresourceRange.baseArrayLayer = 0u;
	viewparams.subresourceRange.layerCount = real_layers;
	viewparams.subresourceRange.baseMipLevel = 0u;
	viewparams.subresourceRange.levelCount = mipLevels;

	return _driver->createGPUImageView(std::move(viewparams));
}

void EnvmapImportanceSampling::initResources(core::smart_refctd_ptr<IGPUImageView> envmap, uint32_t lumaMipMapGenWorkgroupDimension, uint32_t warpMapGenWorkgroupDimension)
{
	const uint32_t MipCountEnvMap = envmap->getCreationParameters().subresourceRange.levelCount;
	const uint32_t MipCountLuminance = MipCountEnvMap;
	
	m_lumaMipMapGenWorkgroupDimension = lumaMipMapGenWorkgroupDimension;
	m_warpMapGenWorkgroupDimension = warpMapGenWorkgroupDimension;
	m_mipCountLuminance = MipCountLuminance;
	m_mipCountEnvmap = MipCountEnvMap;

	{
		uint32_t width, height = 0u;
		getEnvmapResolutionFromMipLevel(MipCountLuminance - 1, width, height);
		m_luminanceBaseImageView = createTexture(m_driver, width, height, EF_R32_SFLOAT, MipCountLuminance);
		assert(m_luminanceBaseImageView);

		m_luminanceMipMaps[0] = m_luminanceBaseImageView;
		for(uint32_t i = 1; i < MipCountLuminance; ++i)
		{
			IGPUImageView::SCreationParams viewCreateParams = m_luminanceBaseImageView->getCreationParameters();
			viewCreateParams.subresourceRange.baseMipLevel = i;
			viewCreateParams.subresourceRange.levelCount = 1u;

			m_luminanceMipMaps[i] = m_driver->createGPUImageView(std::move(viewCreateParams));
		}
	}

	{
		uint32_t width, height = 0u;
		getEnvmapResolutionFromMipLevel(m_mipCountEnvmap- 1, width, height);
		m_warpMap = createTexture(m_driver, width, height, EF_R32G32_SFLOAT);
	}

	ISampler::SParams samplerParams;
	samplerParams.TextureWrapU = samplerParams.TextureWrapV = samplerParams.TextureWrapW = ISampler::ETC_CLAMP_TO_EDGE;
	samplerParams.MinFilter = samplerParams.MaxFilter = ISampler::ETF_NEAREST;
	samplerParams.MipmapMode = ISampler::ESMM_NEAREST;
	samplerParams.AnisotropicFilter = 0u;
	samplerParams.CompareEnable = false;
	auto nearestSampler = m_driver->createGPUSampler(samplerParams);

	// Create DescriptorLayout
	{
		{
			constexpr auto lumaDescriptorCount = 3u;
			IGPUDescriptorSetLayout::SBinding bindings[lumaDescriptorCount];
			bindings[0].binding = 0u;
			bindings[0].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
			bindings[0].stageFlags = ISpecializedShader::ESS_COMPUTE;
			bindings[0].count = 1u;
			bindings[0].samplers = &nearestSampler;
		
			bindings[1].binding = 1u;
			bindings[1].type = asset::EDT_STORAGE_IMAGE;
			bindings[1].stageFlags = ISpecializedShader::ESS_COMPUTE;
			bindings[1].count = 1u;

			bindings[2].binding = 2u;
			bindings[2].type = asset::EDT_STORAGE_IMAGE;
			bindings[2].stageFlags = ISpecializedShader::ESS_COMPUTE;
			bindings[2].count = 1u;

			m_lumaDSLayout = m_driver->createGPUDescriptorSetLayout(bindings,bindings+lumaDescriptorCount);
		}

		{
			
			ISampler::SParams lumaSamplerParams;
			lumaSamplerParams.TextureWrapU = lumaSamplerParams.TextureWrapV = lumaSamplerParams.TextureWrapW = ISampler::ETC_CLAMP_TO_BORDER;
			lumaSamplerParams.BorderColor = ISampler::ETBC_FLOAT_OPAQUE_BLACK;
			lumaSamplerParams.MinFilter = samplerParams.MaxFilter = ISampler::ETF_NEAREST;
			lumaSamplerParams.MipmapMode = ISampler::ESMM_NEAREST;
			lumaSamplerParams.AnisotropicFilter = 0u;
			lumaSamplerParams.CompareEnable = false;
			auto lumaSampler = m_driver->createGPUSampler(lumaSamplerParams);

			core::smart_refctd_ptr<IGPUSampler> samplers[MaxMipCountLuminance];
			for(uint32_t i = 0u; i < MipCountLuminance; ++i)
				samplers[i] = lumaSampler;

			constexpr auto warpDescriptorCount = 2u;
			IGPUDescriptorSetLayout::SBinding bindings[warpDescriptorCount];
			bindings[0].binding = 0u;
			bindings[0].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
			bindings[0].stageFlags = ISpecializedShader::ESS_COMPUTE;
			bindings[0].count = MipCountLuminance;
			bindings[0].samplers = samplers;
		
			bindings[1].binding = 1u;
			bindings[1].type = asset::EDT_STORAGE_IMAGE;
			bindings[1].stageFlags = ISpecializedShader::ESS_COMPUTE;
			bindings[1].count = 1u;

			m_warpDSLayout = m_driver->createGPUDescriptorSetLayout(bindings,bindings+warpDescriptorCount);
		}
	}

	{
		{
			SPushConstantRange range{ISpecializedShader::ESS_COMPUTE,0u,sizeof(LumaMipMapGenShaderData_t)};
			m_lumaPipelineLayout = m_driver->createGPUPipelineLayout(&range,&range+1u,core::smart_refctd_ptr(m_lumaDSLayout));
		
			for(uint32_t i = 0u; i < MipCountLuminance - 1; ++i)
		 		m_lumaDS[i] = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr(m_lumaDSLayout));
	
			for(uint32_t i = 0u; i < MipCountLuminance - 1; ++i)
			{
				const uint32_t src = i;
				const uint32_t dst = i + 1;
			
				IGPUDescriptorSet::SDescriptorInfo envMapDescriptorInfo = {};
				envMapDescriptorInfo.desc = envmap;
				envMapDescriptorInfo.image.sampler = nullptr;
				envMapDescriptorInfo.image.imageLayout = asset::EIL_SHADER_READ_ONLY_OPTIMAL;
		
				IGPUDescriptorSet::SDescriptorInfo srcMipDescriptorInfo = {};
				srcMipDescriptorInfo.desc = m_luminanceMipMaps[src];
				srcMipDescriptorInfo.image.sampler = nullptr;
				srcMipDescriptorInfo.image.imageLayout = asset::EIL_GENERAL;

				IGPUDescriptorSet::SDescriptorInfo dstMipDescriptorInfo = {};
				dstMipDescriptorInfo.desc = m_luminanceMipMaps[dst];
				dstMipDescriptorInfo.image.sampler = nullptr;
				dstMipDescriptorInfo.image.imageLayout = asset::EIL_GENERAL;

				IGPUDescriptorSet::SWriteDescriptorSet writes[3u];
				writes[0].binding = 0u;
				writes[0].arrayElement = 0u;
				writes[0].count = 1u;
				writes[0].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
				writes[0].dstSet = m_lumaDS[i].get();
				writes[0].info = &envMapDescriptorInfo;
		
				writes[1].binding = 1u;
				writes[1].arrayElement = 0u;
				writes[1].count = 1u;
				writes[1].descriptorType = EDT_STORAGE_IMAGE;
				writes[1].dstSet = m_lumaDS[i].get();
				writes[1].info = &srcMipDescriptorInfo;

				writes[2].binding = 2u;
				writes[2].arrayElement = 0u;
				writes[2].count = 1u;
				writes[2].descriptorType = EDT_STORAGE_IMAGE;
				writes[2].dstSet = m_lumaDS[i].get();
				writes[2].info = &dstMipDescriptorInfo;

				m_driver->updateDescriptorSets(3u,writes,0u,nullptr);
			}
		}

		{
			m_warpPipelineLayout = m_driver->createGPUPipelineLayout(nullptr,nullptr,core::smart_refctd_ptr(m_warpDSLayout));
		
		 	m_warpDS = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr(m_warpDSLayout));
			
			IGPUDescriptorSet::SDescriptorInfo luminanceDescriptorInfo = {};
			luminanceDescriptorInfo.desc = m_luminanceBaseImageView;
			luminanceDescriptorInfo.image.sampler = nullptr;
			luminanceDescriptorInfo.image.imageLayout = asset::EIL_SHADER_READ_ONLY_OPTIMAL;
		
			IGPUDescriptorSet::SDescriptorInfo warpMapDescriptorInfo = {};
			warpMapDescriptorInfo.desc = m_warpMap;
			warpMapDescriptorInfo.image.sampler = nullptr;
			warpMapDescriptorInfo.image.imageLayout = asset::EIL_GENERAL;

			IGPUDescriptorSet::SWriteDescriptorSet writes[2u];
			writes[0].binding = 0u;
			writes[0].arrayElement = 0u;
			writes[0].count = 1u;
			writes[0].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
			writes[0].dstSet = m_warpDS.get();
			writes[0].info = &luminanceDescriptorInfo;
		
			writes[1].binding = 1u;
			writes[1].arrayElement = 0u;
			writes[1].count = 1u;
			writes[1].descriptorType = EDT_STORAGE_IMAGE;
			writes[1].dstSet = m_warpDS.get();
			writes[1].info = &warpMapDescriptorInfo;

			m_driver->updateDescriptorSets(2u,writes,0u,nullptr);
		}
	}

	{

		const char* sourceFmt =
R"===(#version 430 core

#define LUMA_MIP_MAP_GEN_WORKGROUP_DIM %u
#define WARP_MAP_GEN_WORKGROUP_DIM %u
#define MAX_LUMINANCE_LEVELS %u

#include "%s"

)===";

		{
			const size_t extraSize = 3u*8u+128u;
			auto lumaShader = core::make_smart_refctd_ptr<ICPUBuffer>(strlen(sourceFmt)+extraSize+1u);
			snprintf(
				reinterpret_cast<char*>(lumaShader->getPointer()),lumaShader->getSize(), sourceFmt,
				lumaMipMapGenWorkgroupDimension,
				warpMapGenWorkgroupDimension,
				MipCountLuminance,
				"nbl/builtin/glsl/ext/EnvmapImportanceSampling/gen_luma_mipmap.comp"
			);

			auto cpuSpecializedShader = core::make_smart_refctd_ptr<ICPUSpecializedShader>(
				core::make_smart_refctd_ptr<ICPUShader>(std::move(lumaShader),ICPUShader::buffer_contains_glsl),
				ISpecializedShader::SInfo{nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE}
			);
	
			auto gpuShader = m_driver->createGPUShader(nbl::core::smart_refctd_ptr<const ICPUShader>(cpuSpecializedShader->getUnspecialized()));
			
			m_lumaGPUShader = m_driver->createGPUSpecializedShader(gpuShader.get(), cpuSpecializedShader->getSpecializationInfo());
			assert(m_lumaGPUShader);
		}

		m_lumaPipeline = m_driver->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(m_lumaPipelineLayout), core::smart_refctd_ptr(m_lumaGPUShader));
		assert(m_lumaPipeline);
		
		{
			const size_t extraSize = 3u*8u+128u;
			auto warpGenShader = core::make_smart_refctd_ptr<ICPUBuffer>(strlen(sourceFmt)+extraSize+1u);
			snprintf(
				reinterpret_cast<char*>(warpGenShader->getPointer()),warpGenShader->getSize(), sourceFmt,
				lumaMipMapGenWorkgroupDimension,
				warpMapGenWorkgroupDimension,
				MipCountLuminance,
				"nbl/builtin/glsl/ext/EnvmapImportanceSampling/gen_warpmap.comp"
			);

			auto cpuSpecializedShader = core::make_smart_refctd_ptr<ICPUSpecializedShader>(
				core::make_smart_refctd_ptr<ICPUShader>(std::move(warpGenShader),ICPUShader::buffer_contains_glsl),
				ISpecializedShader::SInfo{nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE}
			);
	
			auto gpuShader = m_driver->createGPUShader(nbl::core::smart_refctd_ptr<const ICPUShader>(cpuSpecializedShader->getUnspecialized()));
			
			m_warpGPUShader = m_driver->createGPUSpecializedShader(gpuShader.get(), cpuSpecializedShader->getSpecializationInfo());
			assert(m_warpGPUShader);
		}

		m_warpPipeline = m_driver->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(m_warpPipelineLayout), core::smart_refctd_ptr(m_warpGPUShader));
		assert(m_warpPipeline);
	}
}

void EnvmapImportanceSampling::deinitResources()
{
	m_lumaPipeline = nullptr;
	m_lumaGPUShader = nullptr;
	for(uint32_t i = 0u; i < MaxMipCountLuminance - 1; ++i)
		m_lumaDS[i] = nullptr;
	m_lumaPipelineLayout = nullptr;
	m_lumaDSLayout = nullptr;

	for(uint32_t i = 0; i < MaxMipCountLuminance; ++i)
		m_luminanceMipMaps[i] = nullptr;

	m_warpPipeline = nullptr;
	m_warpGPUShader = nullptr;
	m_warpDS = nullptr;
	m_warpPipelineLayout = nullptr;
	m_warpDSLayout = nullptr;
	m_warpMap = nullptr;
}

bool EnvmapImportanceSampling::computeWarpMap(float envMapRegularizationFactor)
{
	bool enableRIS = false;

	LumaMipMapGenShaderData_t pcData = {};
	const nbl::core::vectorSIMDf lumaScales = nbl::core::vectorSIMDf(0.2126729f, 0.7151522f, 0.0721750f, 1.0f);
	
	m_driver->bindComputePipeline(m_lumaPipeline.get());

	// Calc Luma without regularization factor
	{
		pcData.luminanceScales = nbl::core::vectorSIMDf(lumaScales[0], lumaScales[1], lumaScales[2], 0.f);
		pcData.calcLuma = 1;
		m_driver->bindDescriptorSets(EPBP_COMPUTE,m_lumaPipeline->getLayout(),0u,1u,&m_lumaDS[0].get(),nullptr);

		
		uint32_t sourceMipWidth, sourceMipHeight = 0u;
		getEnvmapResolutionFromMipLevel(m_mipCountLuminance - 1, sourceMipWidth, sourceMipHeight);
		
		uint32_t workGroups[2] = {
			(sourceMipWidth-1u)/m_lumaMipMapGenWorkgroupDimension+1u,
			(sourceMipHeight-1u)/m_lumaMipMapGenWorkgroupDimension+1u
		};

		m_driver->pushConstants(m_lumaPipeline->getLayout(),ICPUSpecializedShader::ESS_COMPUTE,0u,sizeof(pcData),&pcData);
		m_driver->dispatch(workGroups[0],workGroups[1],1);
		COpenGLExtensionHandler::pGlMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT|GL_SHADER_IMAGE_ACCESS_BARRIER_BIT|GL_TEXTURE_UPDATE_BARRIER_BIT);
	}

	// Download Luma Image and caclulate directionality metric (0 uniform, 1 totally unidirectional) and new Regularization Factor
	// ideally would want a better metric of how "concentrated" the energy is in one direction rather than variance,
	// it turns out that the first order spherical harmonic band and weighted (by luma) average of directions are the same thing.
	float directionalityMetric = 0.0f;
	{
		uint32_t width, height = 0u;
		getEnvmapResolutionFromMipLevel(m_mipCountLuminance - 1, width, height);

		const uint32_t colorBufferBytesize = width * height * asset::getTexelOrBlockBytesize(EF_R32_SFLOAT);

		auto downloadStagingArea = m_driver->getDefaultDownStreamingBuffer();

		constexpr uint64_t timeoutInNanoSeconds = 300000000000u;
		const auto waitPoint = std::chrono::high_resolution_clock::now()+std::chrono::nanoseconds(timeoutInNanoSeconds);

		uint32_t address = std::remove_pointer<decltype(downloadStagingArea)>::type::invalid_address; // remember without initializing the address to be allocated to invalid_address you won't get an allocation!
		const uint32_t alignment = 4096u; // common page size
		auto unallocatedSize = downloadStagingArea->multi_alloc(waitPoint, 1u, &address, &colorBufferBytesize, &alignment);
		if (unallocatedSize)
		{
			os::Printer::log("Could not download the buffer from the GPU!", ELL_ERROR);
		}

		IImage::SBufferCopy copyRegion = {};
		copyRegion.bufferOffset = address;
		copyRegion.bufferRowLength = 0u;
		copyRegion.bufferImageHeight = 0u;
		//copyRegion.imageSubresource.aspectMask = wait for Vulkan;
		copyRegion.imageSubresource.mipLevel = 0u;
		copyRegion.imageSubresource.baseArrayLayer = 0u;
		copyRegion.imageSubresource.layerCount = 1u;
		copyRegion.imageOffset = { 0u,0u,0u };
		copyRegion.imageExtent = { width, height, 1u };

		auto luminanceGPUImage = m_luminanceMipMaps[0].get()->getCreationParameters().image.get();
		m_driver->copyImageToBuffer(luminanceGPUImage, downloadStagingArea->getBuffer(), 1, &copyRegion);
		
		auto downloadFence = m_driver->placeFence(true);
		
		auto* data = reinterpret_cast<uint8_t*>(downloadStagingArea->getBufferPointer()) + address;
		
		// wait for download fence and then invalidate the CPU cache
		{
			auto result = downloadFence->waitCPU(timeoutInNanoSeconds,true);
			if (result==E_DRIVER_FENCE_RETVAL::EDFR_TIMEOUT_EXPIRED||result==E_DRIVER_FENCE_RETVAL::EDFR_FAIL)
			{
				os::Printer::log("Could not download the buffer from the GPU, fence not signalled!", ELL_ERROR);
				downloadStagingArea->multi_free(1u, &address, &colorBufferBytesize, nullptr);
			}
			if (downloadStagingArea->needsManualFlushOrInvalidate())
				m_driver->invalidateMappedMemoryRanges({{downloadStagingArea->getBuffer()->getBoundMemory(),address,colorBufferBytesize}});
		}

		float* fltData = reinterpret_cast<float*>(data);
		core::vectorSIMDf avgDir;
		{
			core::vector<core::vectorSIMDf> texelAccumulator(width);
			core::vector<core::vectorSIMDf> scanlineAccumulator(height);

			const float toTheta = core::PI<double>()/double(height);
			const float toPhi = 2.0*core::PI<double>()/double(width);
			for (uint32_t j=0; j<height; ++j)
			{
				const float cosTheta = cos((float(j)+0.5f)*toTheta);
				const float sinTheta = sqrt(1.f-cosTheta*cosTheta);

				for (uint32_t i=0; i<width; ++i)
				{
					const float luma = fltData[j*width+i];

					const float phi = (float(i)+0.5f)*toPhi;

					texelAccumulator[i] = core::vectorSIMDf(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta, 1.f) * luma;
				}
				scanlineAccumulator[j] = std::reduce(texelAccumulator.begin(),texelAccumulator.end());
			}
			avgDir = std::reduce(scanlineAccumulator.begin(),scanlineAccumulator.end());
			avgDir /= avgDir.wwww();
			avgDir.w = 0.f;
		}
		// should it be length or length squared?
		directionalityMetric = core::length(avgDir)[0];

		std::cout << "Final Luminance Directionality = " << directionalityMetric << std::endl;
		// the only reason why we'd get a NaN would be because there's literally 0 luminance in the image
		if (core::isnan(directionalityMetric))
			directionalityMetric = 0.f;
		
		downloadStagingArea->multi_free(1u, &address, &colorBufferBytesize, nullptr);
	}

	float regularizationFactor = core::min(envMapRegularizationFactor*directionalityMetric,envMapRegularizationFactor);
	std::cout << "New Regularization Factor based on Directionality = " << regularizationFactor << std::endl;

	constexpr float regularizationThreshold = 0.00001f;
	enableRIS = regularizationFactor>=regularizationThreshold;

	// Calc Luma again with Sin Factor and new Regularization Factor
	{
		pcData.luminanceScales = nbl::core::vectorSIMDf(lumaScales[0] * regularizationFactor, lumaScales[1] * regularizationFactor, lumaScales[2] * regularizationFactor, (1.0f-regularizationFactor));
		pcData.calcLuma = 1;

		m_driver->bindDescriptorSets(EPBP_COMPUTE,m_lumaPipeline->getLayout(),0u,1u,&m_lumaDS[0].get(),nullptr);
		
		uint32_t sourceMipWidth, sourceMipHeight = 0u;
		getEnvmapResolutionFromMipLevel(m_mipCountLuminance - 1, sourceMipWidth, sourceMipHeight);
		
		uint32_t workGroups[2] = {
			(sourceMipWidth-1u)/m_lumaMipMapGenWorkgroupDimension+1u,
			(sourceMipHeight-1u)/m_lumaMipMapGenWorkgroupDimension+1u
		};

		m_driver->pushConstants(m_lumaPipeline->getLayout(),ICPUSpecializedShader::ESS_COMPUTE,0u,sizeof(pcData),&pcData);
		m_driver->dispatch(workGroups[0],workGroups[1],1);
		COpenGLExtensionHandler::pGlMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT|GL_SHADER_IMAGE_ACCESS_BARRIER_BIT|GL_TEXTURE_UPDATE_BARRIER_BIT|GL_SHADER_STORAGE_BARRIER_BIT);
	}

	// Calc Mipmaps
	for(uint32_t s = 0; s < m_mipCountLuminance - 1; ++s)
	{
		m_driver->bindDescriptorSets(EPBP_COMPUTE,m_lumaPipeline->getLayout(),0u,1u,&m_lumaDS[s].get(),nullptr);
		
		uint32_t sourceMipWidth, sourceMipHeight = 0u;
		getEnvmapResolutionFromMipLevel(m_mipCountLuminance - 1 - s, sourceMipWidth, sourceMipHeight);
		
		uint32_t workGroups[2] = {
			(sourceMipWidth-1u)/m_lumaMipMapGenWorkgroupDimension+1u,
			(sourceMipHeight-1u)/m_lumaMipMapGenWorkgroupDimension+1u
		};

		pcData.calcLuma = 0;
		m_driver->pushConstants(m_lumaPipeline->getLayout(),ICPUSpecializedShader::ESS_COMPUTE,0u,sizeof(pcData),&pcData);
		m_driver->dispatch(workGroups[0],workGroups[1],1);
		COpenGLExtensionHandler::pGlMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT|GL_SHADER_IMAGE_ACCESS_BARRIER_BIT|GL_TEXTURE_UPDATE_BARRIER_BIT);
	}

	// Generate WarpMap
	{
		m_driver->bindComputePipeline(m_warpPipeline.get());
	
		m_driver->bindDescriptorSets(EPBP_COMPUTE,m_warpPipeline->getLayout(),0u,1u,&m_warpDS.get(),nullptr);
		
		uint32_t warpMapWidth, warpMapHeight = 0u;
		getEnvmapResolutionFromMipLevel(m_mipCountEnvmap - 1, warpMapWidth, warpMapHeight);

		uint32_t workGroups[2] = {
			(warpMapWidth-1u)/m_warpMapGenWorkgroupDimension+1u,
			(warpMapHeight-1u)/m_warpMapGenWorkgroupDimension+1u
		};
		m_driver->dispatch(workGroups[0],workGroups[1],1);
		COpenGLExtensionHandler::pGlMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT|GL_SHADER_IMAGE_ACCESS_BARRIER_BIT|GL_TEXTURE_UPDATE_BARRIER_BIT);
	}

	return enableRIS;
}


