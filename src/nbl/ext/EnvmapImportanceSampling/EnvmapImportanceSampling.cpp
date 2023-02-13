// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/EnvmapImportanceSampling/EnvmapImportanceSampling.h"

#include <cstdio>

using namespace nbl;
using namespace nbl::asset;
using namespace nbl::video;
using namespace ext::EnvmapImportanceSampling;


static core::smart_refctd_ptr<IGPUImageView> createTexture(nbl::video::IVideoDriver* _driver, const VkExtent3D extent, E_FORMAT format, uint32_t mipLevels=1u, uint32_t layers=0u)
{
	const auto real_layers = layers ? layers:1u;

	IGPUImage::SCreationParams imgparams;
	imgparams.extent = extent;
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

void EnvmapImportanceSampling::initResources(core::smart_refctd_ptr<IGPUImageView> envmap, uint32_t lumaGenWorkgroupDimension, uint32_t warpMapGenWorkgroupDimension)
{
	const auto EnvmapExtent = envmap->getCreationParameters().image->getCreationParameters().extent;
	// we don't need the 1x1 mip for anything
	const uint32_t MipCountLuminance = IImage::calculateFullMipPyramidLevelCount(EnvmapExtent,IImage::ET_2D)-1;
	const auto EnvMapPoTExtent = [MipCountLuminance]() -> VkExtent3D
	{
		const uint32_t width = 0x1u<<MipCountLuminance;
		return { width,width>>1u,1u };
	}();
	auto calcWorkgroups = [](uint32_t* workGroups, const VkExtent3D extent, const uint32_t workgroupDimension)
	{
		for (auto i=0; i<2; i++)
			workGroups[i] = ((&extent.width)[i]-1u)/workgroupDimension+1u;
	};

	// TODO: Can we get away with R16_SFLOAT for the probabilities?
	m_luminance = createTexture(m_driver,EnvMapPoTExtent,EF_R32_SFLOAT,MipCountLuminance);
	calcWorkgroups(m_lumaWorkgroups,EnvMapPoTExtent,lumaGenWorkgroupDimension);

	// default make the warp-map same resolution as input envmap
	// Format needs to be 32bit full precision float, because the Jacobian needs to accurately match PDF
	const uint32_t upscale = 0;
	const VkExtent3D WarpMapExtent = {EnvMapPoTExtent.width<<upscale,EnvMapPoTExtent.height<<upscale,EnvMapPoTExtent.depth};
	m_warpMap = createTexture(m_driver,WarpMapExtent,EF_R32G32_SFLOAT);
	calcWorkgroups(m_warpWorkgroups,WarpMapExtent,warpMapGenWorkgroupDimension);
	
	//
	auto genPipeline = [=](const char* shaderPath, core::smart_refctd_ptr<IGPUPipelineLayout>&& pipelineLayout) -> core::smart_refctd_ptr<video::IGPUComputePipeline>
	{
		const char* sourceFmt =
			R"===(#version 430 core

#define LUMA_MIP_MAP_GEN_WORKGROUP_DIM %u
#define WARP_MAP_GEN_WORKGROUP_DIM %u

#include "%s"

)===";

		const size_t extraSize = 2u * 8u + 128u;
		auto shader = core::make_smart_refctd_ptr<ICPUBuffer>(strlen(sourceFmt) + extraSize + 1u);
		snprintf(
			reinterpret_cast<char*>(shader->getPointer()), shader->getSize(), sourceFmt,
			lumaGenWorkgroupDimension,
			warpMapGenWorkgroupDimension,
			shaderPath
		);
		auto gpuShader = m_driver->createGPUShader(core::make_smart_refctd_ptr<ICPUShader>(std::move(shader), ICPUShader::buffer_contains_glsl));
		if (!gpuShader)
			return nullptr;

		auto specializedShader = m_driver->createGPUSpecializedShader(gpuShader.get(), ISpecializedShader::SInfo{ nullptr,nullptr,"main",asset::ISpecializedShader::ESS_COMPUTE });
		if (!specializedShader)
			return nullptr;

		return m_driver->createGPUComputePipeline(nullptr,std::move(pipelineLayout),std::move(specializedShader));
	};

	// Create Everything
	{
		ISampler::SParams samplerParams;
		samplerParams.TextureWrapU = samplerParams.TextureWrapV = samplerParams.TextureWrapW = ISampler::ETC_CLAMP_TO_EDGE;
		samplerParams.MinFilter = ISampler::ETF_NEAREST;
		samplerParams.MaxFilter = ISampler::ETF_LINEAR;
		samplerParams.MipmapMode = ISampler::ESMM_NEAREST;
		samplerParams.AnisotropicFilter = 0u;
		samplerParams.CompareEnable = false;

		IGPUDescriptorSet::SDescriptorInfo lumaDescriptorInfo = {};
		lumaDescriptorInfo.desc = m_luminance;
		lumaDescriptorInfo.image.sampler = nullptr;

		{
			auto upscaleSampler = m_driver->createGPUSampler(samplerParams);

			constexpr auto lumaDescriptorCount = 3u;
			IGPUDescriptorSetLayout::SBinding bindings[lumaDescriptorCount];
			bindings[0].binding = 0u;
			bindings[0].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
			bindings[0].stageFlags = ISpecializedShader::ESS_COMPUTE;
			bindings[0].count = 1u;
			bindings[0].samplers = &upscaleSampler;
		
			bindings[1].binding = 1u;
			bindings[1].type = asset::EDT_STORAGE_BUFFER_DYNAMIC;
			bindings[1].stageFlags = ISpecializedShader::ESS_COMPUTE;
			bindings[1].count = 1u;

			bindings[2].binding = 2u;
			bindings[2].type = asset::EDT_STORAGE_IMAGE;
			bindings[2].stageFlags = ISpecializedShader::ESS_COMPUTE;
			bindings[2].count = 1u;

			auto lumaDSLayout = m_driver->createGPUDescriptorSetLayout(bindings,bindings+lumaDescriptorCount);
			{
				SPushConstantRange range{ ISpecializedShader::ESS_COMPUTE,0u,sizeof(nbl_glsl_ext_EnvmapSampling_LumaGenShaderData_t) };
				auto lumaPipelineLayout = m_driver->createGPUPipelineLayout(&range,&range+1u,core::smart_refctd_ptr(lumaDSLayout));
				m_lumaMeasurePipeline = genPipeline("nbl/builtin/glsl/ext/EnvmapImportanceSampling/measure_luma.comp",core::smart_refctd_ptr(lumaPipelineLayout));
				m_lumaGenPipeline = genPipeline("nbl/builtin/glsl/ext/EnvmapImportanceSampling/gen_luma.comp",std::move(lumaPipelineLayout));
			}
			m_lumaDS = m_driver->createGPUDescriptorSet(std::move(lumaDSLayout));

			{
				IGPUDescriptorSet::SDescriptorInfo envMapDescriptorInfo = {};
				envMapDescriptorInfo.desc = envmap;
				envMapDescriptorInfo.image.sampler = nullptr;
				envMapDescriptorInfo.image.imageLayout = asset::EIL_SHADER_READ_ONLY_OPTIMAL;

				IGPUDescriptorSet::SDescriptorInfo lumaMeasurementInfo = {};
				lumaMeasurementInfo.desc = core::smart_refctd_ptr<IGPUBuffer>(m_driver->getDefaultDownStreamingBuffer()->getBuffer());
				lumaMeasurementInfo.buffer = {0,calcMeasurementBufferSize()};

				IGPUDescriptorSet::SWriteDescriptorSet writes[lumaDescriptorCount];
				for (auto i=0u; i<lumaDescriptorCount; i++)
				{
					writes[i].binding = bindings[i].binding;
					writes[i].arrayElement = 0u;
					writes[i].count = bindings[i].count;
					writes[i].descriptorType = bindings[i].type;
					writes[i].dstSet = m_lumaDS.get();
				}
				writes[0].info = &envMapDescriptorInfo;
				writes[1].info = &lumaMeasurementInfo;
				writes[2].info = &lumaDescriptorInfo;
				lumaDescriptorInfo.image.imageLayout = asset::EIL_GENERAL;
				m_driver->updateDescriptorSets(lumaDescriptorCount,writes,0u,nullptr);
			}
		}

		{
			samplerParams.TextureWrapU = samplerParams.TextureWrapV = samplerParams.TextureWrapW = ISampler::ETC_CLAMP_TO_BORDER;
			samplerParams.BorderColor = ISampler::ETBC_FLOAT_OPAQUE_BLACK;
			samplerParams.MaxFilter = ISampler::ETF_NEAREST;
			auto lumaSampler = m_driver->createGPUSampler(samplerParams);

			constexpr auto warpDescriptorCount = 2u;
			IGPUDescriptorSetLayout::SBinding bindings[warpDescriptorCount];
			bindings[0].binding = 0u;
			bindings[0].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
			bindings[0].stageFlags = ISpecializedShader::ESS_COMPUTE;
			bindings[0].count = 1;
			bindings[0].samplers = &lumaSampler;
		
			bindings[1].binding = 1u;
			bindings[1].type = asset::EDT_STORAGE_IMAGE;
			bindings[1].stageFlags = ISpecializedShader::ESS_COMPUTE;
			bindings[1].count = 1u;

			auto warpDSLayout = m_driver->createGPUDescriptorSetLayout(bindings,bindings+warpDescriptorCount);

			m_warpPipeline = genPipeline(
				"nbl/builtin/glsl/ext/EnvmapImportanceSampling/gen_warpmap.comp",
				m_driver->createGPUPipelineLayout(nullptr,nullptr,core::smart_refctd_ptr(warpDSLayout))
			);

		 	m_warpDS = m_driver->createGPUDescriptorSet(std::move(warpDSLayout));
			{		
				IGPUDescriptorSet::SDescriptorInfo warpMapDescriptorInfo = {};
				warpMapDescriptorInfo.desc = m_warpMap;
				warpMapDescriptorInfo.image.sampler = nullptr;
				warpMapDescriptorInfo.image.imageLayout = asset::EIL_GENERAL;

				IGPUDescriptorSet::SWriteDescriptorSet writes[warpDescriptorCount];
				for (auto i=0u; i<warpDescriptorCount; i++)
				{
					writes[i].binding = bindings[i].binding;
					writes[i].arrayElement = 0u;
					writes[i].count = bindings[i].count;
					writes[i].descriptorType = bindings[i].type;
					writes[i].dstSet = m_warpDS.get();
				}
				writes[0].info = &lumaDescriptorInfo;
				writes[1].info = &warpMapDescriptorInfo;
				lumaDescriptorInfo.image.imageLayout = asset::EIL_SHADER_READ_ONLY_OPTIMAL;
				m_driver->updateDescriptorSets(warpDescriptorCount,writes,0u,nullptr);
			}
		}
	}
}

void EnvmapImportanceSampling::deinitResources()
{
	m_lumaMeasurePipeline = nullptr;
	m_lumaGenPipeline = nullptr;
	m_lumaDS = nullptr;

	m_warpPipeline = nullptr;
	m_warpDS = nullptr;

	m_warpMap = nullptr;
	m_luminance = nullptr;
}

bool EnvmapImportanceSampling::computeWarpMap(const float envMapRegularizationFactor, float& pdfNormalizationFactor, float& maxEmittanceLuma)
{
	bool enableRIS = false;
	//
	nbl_glsl_ext_EnvmapSampling_LumaGenShaderData_t pcData = {};
	pcData.luminanceScales.set(0.2126729f, 0.7151522f, 0.0721750f, 0.0f);
	{
		const auto imageExtent = m_luminance->getCreationParameters().image->getCreationParameters().extent;
		pcData.lumaMapResolution = {imageExtent.width,imageExtent.height};
	}

	auto dynamicOffsets = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<uint32_t>>(1u);
	auto lumaDispatch = [&](core::smart_refctd_ptr<IGPUComputePipeline>& pipeline,core::smart_refctd_dynamic_array<uint32_t>* dynamicOffsets)
	{
		m_driver->bindComputePipeline(pipeline.get());
		m_driver->bindDescriptorSets(EPBP_COMPUTE,pipeline->getLayout(),0u,1u,&m_lumaDS.get(),dynamicOffsets);
		m_driver->pushConstants(pipeline->getLayout(),ICPUSpecializedShader::ESS_COMPUTE,0u,sizeof(pcData),&pcData);
		m_driver->dispatch(m_lumaWorkgroups[0],m_lumaWorkgroups[1],1);
	};

	// Calculate directionality metric (0 uniform, 1 totally unidirectional) and new Regularization Factor.
	// Ideally would want a better metric of how "concentrated" the energy is in one direction rather than variance, so it
	// turns out that the first order spherical harmonic band and weighted (by luma) average of directions are the same thing.
	float directionalityMetric = [&]()
	{
		maxEmittanceLuma = 0.f;
		// 3 seconds is a long time
		constexpr uint64_t timeoutInNanoSeconds = 300000000000u;

		auto downloadStagingArea = m_driver->getDefaultDownStreamingBuffer();
		const uint32_t size = calcMeasurementBufferSize();
		// remember that without initializing the address to be allocated to invalid_address you won't get an allocation!
		const auto& address = dynamicOffsets->operator[](0) = std::remove_pointer<decltype(downloadStagingArea)>::type::invalid_address;
		// allocate
		{
			// common page size
			const uint32_t alignment = 4096u;
			const auto waitPoint = std::chrono::high_resolution_clock::now()+std::chrono::nanoseconds(timeoutInNanoSeconds);
			auto unallocatedSize = downloadStagingArea->multi_alloc(waitPoint,1u,dynamicOffsets->data(),&size,&alignment);
			if (unallocatedSize)
			{
				os::Printer::log("Could not download the buffer from the GPU!", ELL_ERROR);
				return 0.f;
			}
		}
		auto* data = reinterpret_cast<nbl_glsl_ext_EnvmapSampling_LumaMeasurement_t*>(reinterpret_cast<uint8_t*>(downloadStagingArea->getBufferPointer())+address);

		// measure into buffer
		lumaDispatch(m_lumaMeasurePipeline,&dynamicOffsets);
		COpenGLExtensionHandler::pGlMemoryBarrier(GL_ALL_BARRIER_BITS); // TODO: rethink when reimplementing in Vulkan
		{
			// place and wait for download fence
			auto downloadFence = m_driver->placeFence(true);
			auto result = downloadFence->waitCPU(timeoutInNanoSeconds,true);
			//
			if (result==E_DRIVER_FENCE_RETVAL::EDFR_TIMEOUT_EXPIRED || result==E_DRIVER_FENCE_RETVAL::EDFR_FAIL)
			{
				os::Printer::log("Could not download the buffer from the GPU, fence not signalled!", ELL_ERROR);
				downloadStagingArea->multi_free(1u,&address,&size,nullptr);
				return 0.f;
			}
			// then invalidate the CPU cache of the mapping
			if (downloadStagingArea->needsManualFlushOrInvalidate())
				m_driver->invalidateMappedMemoryRanges({ {downloadStagingArea->getBuffer()->getBoundMemory(),address,size} });
		}

		// reduce
		core::vectorSIMDf avgDir;
		{
			const auto reduction = std::reduce(
				data,data+size/sizeof(nbl_glsl_ext_EnvmapSampling_LumaMeasurement_t),
				nbl_glsl_ext_EnvmapSampling_LumaMeasurement_t{0.f,0.f,0.f,0.f,0.f},
				[](nbl_glsl_ext_EnvmapSampling_LumaMeasurement_t lhs, const nbl_glsl_ext_EnvmapSampling_LumaMeasurement_t& rhs){
					lhs.xDirSum += rhs.xDirSum;
					lhs.yDirSum += rhs.yDirSum;
					lhs.zDirSum += rhs.zDirSum;
					lhs.weightSum += rhs.weightSum;
					if (lhs.maxLuma<rhs.maxLuma)
						lhs.maxLuma = rhs.maxLuma;
					return lhs;
				}
			);
			pdfNormalizationFactor = double(pcData.lumaMapResolution.x*pcData.lumaMapResolution.y)/(2.0*core::PI<double>()*core::PI<double>()*reduction.weightSum);
			avgDir.set(&reduction.xDirSum);
			maxEmittanceLuma = reduction.maxLuma;
			downloadStagingArea->multi_free(1u,&address,&size,nullptr);
		}

		avgDir /= avgDir.wwww();
		avgDir.w = 0.f;
		// should it be length or length squared?
		const float directionality = core::length(avgDir)[0];
		std::cout << "Final Luminance Directionality = " << directionality << std::endl;
		// the only reason why we'd get a NaN would be because there's literally 0 luminance in the image
		return core::isnan(directionality) ? 0.f:directionality;
	}();

	const float regularizationFactor = core::min(envMapRegularizationFactor*directionalityMetric,envMapRegularizationFactor);
	std::cout << "New Regularization Factor based on Directionality = " << regularizationFactor << std::endl;

	constexpr float regularizationThreshold = 0.00001f;
	enableRIS = regularizationFactor>=regularizationThreshold;

	// Calc Luma again with new Regularization Factor
	{
		pcData.luminanceScales *= regularizationFactor;
		pcData.luminanceScales.w = 1.f-regularizationFactor;
		lumaDispatch(m_lumaGenPipeline,&dynamicOffsets);
		COpenGLExtensionHandler::pGlMemoryBarrier(GL_ALL_BARRIER_BITS); // TODO: rethink when reimplementing in Vulkan
	}

	// Calc Mipmaps
	m_luminance->regenerateMipMapLevels();

	// Generate WarpMap
	{
		m_driver->bindComputePipeline(m_warpPipeline.get());
		m_driver->bindDescriptorSets(EPBP_COMPUTE,m_warpPipeline->getLayout(),0u,1u,&m_warpDS.get(),nullptr);
		m_driver->dispatch(m_warpWorkgroups[0],m_warpWorkgroups[1],1);
		COpenGLExtensionHandler::pGlMemoryBarrier(GL_ALL_BARRIER_BITS); // TODO: rethink when reimplementing in Vulkan
	}

	return enableRIS;
}


