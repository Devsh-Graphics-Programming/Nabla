#include <chrono> 

#include "WaveSimApp.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/ext/FFT/FFT.h"

using namespace nbl;
using namespace core;
using namespace asset;
using namespace video;

bool WaveSimApp::Init()
{
	nbl::SIrrlichtCreationParameters params;
	params.Bits = 24;
	params.ZBufferBits = 24;
	params.DriverType = video::EDT_OPENGL;
	params.WindowSize = dimension2d<uint32_t>(900, 900);
	params.Fullscreen = false;

	m_device = createDeviceEx(params);
	if (!m_device)
		return false;

	m_driver = m_device->getVideoDriver();
	m_filesystem = m_device->getFileSystem();
	m_asset_manager = m_device->getAssetManager();
	m_device->setWindowCaption(L"Tessendorf Waves Simulation");

	IGPUDescriptorSetLayout::SBinding binding{ 0u, EDT_COMBINED_IMAGE_SAMPLER, 1u, IGPUSpecializedShader::ESS_FRAGMENT, nullptr };
	m_gpu_descriptor_set_layout = m_driver->createGPUDescriptorSetLayout(&binding, &binding + 1u);
	return true;
}

bool WaveSimApp::CreatePresentingPipeline()
{
	const char* fragment_shader_path = "../waves_display.frag";
	auto full_screen_triangle = ext::FullScreenTriangle::createFullScreenTriangle(m_device->getAssetManager(), m_device->getVideoDriver());

	auto createGPUPipeline = [&](IImageView<ICPUImage>::E_TYPE type) -> graphicsPipeline
	{
		auto getPathToFragmentShader = [&]()
		{
			switch (type)
			{
			case IImageView<ICPUImage>::E_TYPE::ET_2D:
				return "../waves_display.frag";
			default:
			{
				os::Printer::log("Not supported image view in the example!", ELL_ERROR);
				return "";
			}
			}
		};

		IAssetLoader::SAssetLoadParams lp;
		auto fs_bundle = m_device->getAssetManager()->getAsset(getPathToFragmentShader(), lp);
		auto fs_contents = fs_bundle.getContents();
		if (fs_contents.begin() == fs_contents.end())
			return false;

		ICPUSpecializedShader* fs = static_cast<ICPUSpecializedShader*>(fs_contents.begin()->get());

		auto frag_shader = m_driver->getGPUObjectsFromAssets(&fs, &fs + 1)->front();
		if (!frag_shader)
			return {};

		IGPUSpecializedShader* shaders[2] = { std::get<0>(full_screen_triangle).get(),frag_shader.get() };
		SBlendParams blend_params;
		blend_params.logicOpEnable = false;
		blend_params.logicOp = ELO_NO_OP;
		for (size_t i = 0ull; i < SBlendParams::MAX_COLOR_ATTACHMENT_COUNT; i++)
			blend_params.blendParams[i].attachmentEnabled = (i == 0ull);
		SRasterizationParams raster_params;
		raster_params.faceCullingMode = EFCM_NONE;
		raster_params.depthCompareOp = ECO_ALWAYS;
		raster_params.minSampleShading = 1.f;
		raster_params.depthWriteEnable = false;
		raster_params.depthTestEnable = false;

		auto gpu_pipeline_layout = m_driver->createGPUPipelineLayout(nullptr, nullptr, nullptr, nullptr, nullptr, core::smart_refctd_ptr(m_gpu_descriptor_set_layout));

		return m_driver->createGPURenderpassIndependentPipeline(nullptr, std::move(gpu_pipeline_layout), shaders, shaders + sizeof(shaders) / sizeof(IGPUSpecializedShader*),
			std::get<SVertexInputParams>(full_screen_triangle), blend_params,
			std::get<SPrimitiveAssemblyParams>(full_screen_triangle), raster_params);
	};

	m_presenting_pipeline = createGPUPipeline(IImageView<ICPUImage>::E_TYPE::ET_2D);
	assert(m_presenting_pipeline.get());
	{
		SBufferBinding<IGPUBuffer> idxBinding{ 0ull, nullptr };
		m_current_gpu_mesh_buffer = core::make_smart_refctd_ptr<IGPUMeshBuffer>(nullptr, nullptr, nullptr, std::move(idxBinding));
		m_current_gpu_mesh_buffer->setIndexCount(3u);
		m_current_gpu_mesh_buffer->setInstanceCount(1u);
	}
	return true;
}

bool WaveSimApp::CreateComputePipelines()
{
	enum class EPipeline
	{
		RANDOMIZE_SPECTRUM,
		ANIMATE_SPECTRUM,
		IFFT_STAGE_1,
		IFFT_STAGE_2,
		GENERATE_NORMALMAP
	};
	auto getFilePath = [](EPipeline type)
	{
		switch (type)
		{
		case EPipeline::RANDOMIZE_SPECTRUM:
			return "../initial_spectrum.comp";
		case EPipeline::ANIMATE_SPECTRUM:
			return "../spectrum_animation.comp";
		case EPipeline::GENERATE_NORMALMAP:
			return "../normalmap_generation.comp";
		case EPipeline::IFFT_STAGE_1:
			return "../ifft_x.comp";
		case EPipeline::IFFT_STAGE_2:
			return "../ifft_y.comp";
		}
	};
	auto createShader = [&](EPipeline type)
	{
		switch (type)
		{
		case EPipeline::GENERATE_NORMALMAP:
		case EPipeline::ANIMATE_SPECTRUM:
		case EPipeline::RANDOMIZE_SPECTRUM:
		{
			std::string filepath = getFilePath(type);
			auto f = core::smart_refctd_ptr<io::IReadFile>(m_filesystem->createAndOpenFile(filepath.c_str()));

			asset::IAssetLoader::SAssetLoadParams lp;
			auto cs_bundle = m_asset_manager->getAsset(filepath.c_str(), lp);
			auto cs = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*cs_bundle.getContents().begin());

			auto cs_rawptr = cs.get();
			return m_driver->getGPUObjectsFromAssets(&cs_rawptr, &cs_rawptr + 1)->front();
		}
		case EPipeline::IFFT_STAGE_1:
		{
			const char* sourceFmt =
				R"===(#version 450

#define USE_SSBO_FOR_INPUT 1
#define _NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_ %u
#define _NBL_GLSL_EXT_FFT_MAX_DIM_SIZE_ %u
#define _NBL_GLSL_EXT_FFT_MAX_ITEMS_PER_THREAD %u
 
#include "../ifft_x.comp"

)===";
			const size_t extraSize = 128;
			constexpr uint32_t DEFAULT_WORK_GROUP_SIZE = 256u;
			uint32_t maxPaddedDimensionSize = std::max(m_params.height, m_params.width);
			const uint32_t maxItemsPerThread = ((maxPaddedDimensionSize >> 1) - 1u) / (DEFAULT_WORK_GROUP_SIZE)+1u;
			auto shader = core::make_smart_refctd_ptr<ICPUBuffer>(strlen(sourceFmt) + extraSize + 1u);
			snprintf(
				reinterpret_cast<char*>(shader->getPointer()), shader->getSize(), sourceFmt,
				DEFAULT_WORK_GROUP_SIZE,
				maxPaddedDimensionSize,
				maxItemsPerThread
			);

			auto cpuSpecializedShader = core::make_smart_refctd_ptr<ICPUSpecializedShader>(
				core::make_smart_refctd_ptr<ICPUShader>(std::move(shader), ICPUShader::buffer_contains_glsl),
				ISpecializedShader::SInfo{ nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE }
			);

			auto gpuShader = m_driver->createGPUShader(nbl::core::smart_refctd_ptr<const ICPUShader>(cpuSpecializedShader->getUnspecialized()));

			auto gpuSpecializedShader = m_driver->createGPUSpecializedShader(gpuShader.get(), cpuSpecializedShader->getSpecializationInfo());

			return gpuSpecializedShader;
		}
		case EPipeline::IFFT_STAGE_2:
		{
			const char* sourceFmt =
				R"===(#version 450

#define USE_SSBO_FOR_INPUT 1
#define _NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_ %u
#define _NBL_GLSL_EXT_FFT_MAX_DIM_SIZE_ %u
#define _NBL_GLSL_EXT_FFT_MAX_ITEMS_PER_THREAD %u
 
#include "../ifft_y.comp"

)===";
			const size_t extraSize = 128;
			constexpr uint32_t DEFAULT_WORK_GROUP_SIZE = 256u;
			uint32_t maxPaddedDimensionSize = std::max(m_params.height, m_params.width);
			const uint32_t maxItemsPerThread = ((maxPaddedDimensionSize >> 1) - 1u) / (DEFAULT_WORK_GROUP_SIZE)+1u;
			auto shader = core::make_smart_refctd_ptr<ICPUBuffer>(strlen(sourceFmt) + extraSize + 1u);
			snprintf(
				reinterpret_cast<char*>(shader->getPointer()), shader->getSize(), sourceFmt,
				DEFAULT_WORK_GROUP_SIZE,
				maxPaddedDimensionSize,
				maxItemsPerThread
			);

			auto cpuSpecializedShader = core::make_smart_refctd_ptr<ICPUSpecializedShader>(
				core::make_smart_refctd_ptr<ICPUShader>(std::move(shader), ICPUShader::buffer_contains_glsl),
				ISpecializedShader::SInfo{ nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE }
			);

			auto gpuShader = m_driver->createGPUShader(nbl::core::smart_refctd_ptr<const ICPUShader>(cpuSpecializedShader->getUnspecialized()));

			auto gpuSpecializedShader = m_driver->createGPUSpecializedShader(gpuShader.get(), cpuSpecializedShader->getSpecializationInfo());

			return gpuSpecializedShader;
		}
		
		}
	};

	nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> init_ds_layout;
	nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> ift_x_ds_layout;
	nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> ift_y_ds_layout;
	nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> normalmap_ds_layout;
	nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> animating_ds_layout;
	{
		IGPUDescriptorSetLayout::SBinding texture_bindings[1];
		texture_bindings[0].binding = 0;
		texture_bindings[0].type = EDT_STORAGE_BUFFER;
		texture_bindings[0].count = 1u;
		texture_bindings[0].stageFlags = static_cast<IGPUSpecializedShader::E_SHADER_STAGE>(IGPUSpecializedShader::ESS_COMPUTE);
		texture_bindings[0].samplers = nullptr;

		init_ds_layout = m_driver->createGPUDescriptorSetLayout(texture_bindings, texture_bindings + 1);
		m_randomizer_descriptor_set = m_driver->createGPUDescriptorSet(init_ds_layout);
	}
	{
		IGPUDescriptorSetLayout::SBinding texture_bindings[2];
		texture_bindings[0].binding = 0;
		texture_bindings[0].type = EDT_STORAGE_BUFFER;
		texture_bindings[0].count = 1u;
		texture_bindings[0].stageFlags = static_cast<IGPUSpecializedShader::E_SHADER_STAGE>(IGPUSpecializedShader::ESS_COMPUTE);
		texture_bindings[0].samplers = nullptr;

		texture_bindings[1].binding = 1;
		texture_bindings[1].type = EDT_STORAGE_BUFFER;
		texture_bindings[1].count = 1u;
		texture_bindings[1].stageFlags = static_cast<IGPUSpecializedShader::E_SHADER_STAGE>(IGPUSpecializedShader::ESS_COMPUTE);
		texture_bindings[1].samplers = nullptr;

		ift_x_ds_layout = m_driver->createGPUDescriptorSetLayout(texture_bindings, texture_bindings + 2);
		m_ifft_1_descriptor_set = m_driver->createGPUDescriptorSet(ift_x_ds_layout);
	}
	{
		IGPUDescriptorSetLayout::SBinding texture_bindings[2];
		texture_bindings[0].binding = 0;
		texture_bindings[0].type = EDT_STORAGE_BUFFER;
		texture_bindings[0].count = 1u;
		texture_bindings[0].stageFlags = static_cast<IGPUSpecializedShader::E_SHADER_STAGE>(IGPUSpecializedShader::ESS_COMPUTE);
		texture_bindings[0].samplers = nullptr;

		texture_bindings[1].binding = 1;
		texture_bindings[1].type = EDT_STORAGE_IMAGE;
		texture_bindings[1].count = 1u;
		texture_bindings[1].stageFlags = static_cast<IGPUSpecializedShader::E_SHADER_STAGE>(IGPUSpecializedShader::ESS_COMPUTE);
		texture_bindings[1].samplers = nullptr;

		ift_y_ds_layout = m_driver->createGPUDescriptorSetLayout(texture_bindings, texture_bindings + 2);
		m_ifft_2_descriptor_set = m_driver->createGPUDescriptorSet(ift_y_ds_layout);
	}
	{
		IGPUDescriptorSetLayout::SBinding texture_bindings[2];
		texture_bindings[0].binding = 0;
		texture_bindings[0].type = EDT_COMBINED_IMAGE_SAMPLER;
		texture_bindings[0].count = 1u;
		texture_bindings[0].stageFlags = static_cast<IGPUSpecializedShader::E_SHADER_STAGE>(IGPUSpecializedShader::ESS_COMPUTE);
		texture_bindings[0].samplers = nullptr;

		texture_bindings[1].binding = 1;
		texture_bindings[1].type = EDT_STORAGE_IMAGE;
		texture_bindings[1].count = 1u;
		texture_bindings[1].stageFlags = static_cast<IGPUSpecializedShader::E_SHADER_STAGE>(IGPUSpecializedShader::ESS_COMPUTE);
		texture_bindings[1].samplers = nullptr;

		normalmap_ds_layout = m_driver->createGPUDescriptorSetLayout(texture_bindings, texture_bindings + 2);
		m_normalmap_descriptor_set = m_driver->createGPUDescriptorSet(normalmap_ds_layout);
	}
	{
		IGPUDescriptorSetLayout::SBinding texture_bindings[2];
		texture_bindings[0].binding = 0;
		texture_bindings[0].type = EDT_STORAGE_BUFFER;
		texture_bindings[0].count = 1u;
		texture_bindings[0].stageFlags = static_cast<IGPUSpecializedShader::E_SHADER_STAGE>(IGPUSpecializedShader::ESS_COMPUTE);
		texture_bindings[0].samplers = nullptr;

		texture_bindings[1].binding = 1;
		texture_bindings[1].type = EDT_STORAGE_BUFFER;
		texture_bindings[1].count = 1u;
		texture_bindings[1].stageFlags = static_cast<IGPUSpecializedShader::E_SHADER_STAGE>(IGPUSpecializedShader::ESS_COMPUTE);
		texture_bindings[1].samplers = nullptr;

		animating_ds_layout = m_driver->createGPUDescriptorSetLayout(texture_bindings, texture_bindings + 2);
		m_spectrum_animating_descriptor_set = m_driver->createGPUDescriptorSet(animating_ds_layout);
	}

	auto createComputePipeline = [&](EPipeline pipeline_type)
	{
		core::smart_refctd_ptr<video::IGPUComputePipeline> comp_pipeline;
		{
			core::smart_refctd_ptr<video::IGPUPipelineLayout> layout;
			{

				smart_refctd_ptr<IGPUDescriptorSetLayout> ds_layout;
				switch (pipeline_type)
				{
				case EPipeline::RANDOMIZE_SPECTRUM:
				{
					ds_layout = init_ds_layout;
					asset::SPushConstantRange range;
					range.size = sizeof(WaveSimParams);
					range.offset = 0u;
					range.stageFlags = asset::ISpecializedShader::ESS_COMPUTE;
					layout = m_driver->createGPUPipelineLayout(&range,
						&range + 1,
						std::move(ds_layout),
						nullptr,
						nullptr,
						nullptr);
				}
				break;
				case EPipeline::IFFT_STAGE_1:
				{
					ds_layout = ift_x_ds_layout;
					asset::SPushConstantRange range;
					range.size = sizeof(float) + sizeof(uint32_t) * 2 + sizeof(ext::FFT::FFT::Parameters_t);
					range.offset = 0u;
					range.stageFlags = asset::ISpecializedShader::ESS_COMPUTE;
					layout = m_driver->createGPUPipelineLayout(&range,
						&range + 1,
						std::move(ds_layout),
						nullptr,
						nullptr,
						nullptr);
				}
				break;
				case EPipeline::IFFT_STAGE_2:
				{
					ds_layout = ift_y_ds_layout;
					asset::SPushConstantRange range;
					range.size = sizeof(ext::FFT::FFT::Parameters_t);
					range.offset = 0u;
					range.stageFlags = asset::ISpecializedShader::ESS_COMPUTE;
					layout = m_driver->createGPUPipelineLayout(&range,
						&range + 1,
						std::move(ds_layout),
						nullptr,
						nullptr,
						nullptr);
				}
				break;
				case EPipeline::GENERATE_NORMALMAP:
				{
					ds_layout = normalmap_ds_layout;
					asset::SPushConstantRange range;
					range.size = sizeof(dimension2du);
					range.offset = 0u;
					range.stageFlags = asset::ISpecializedShader::ESS_COMPUTE;
					layout = m_driver->createGPUPipelineLayout(&range,
						&range + 1,
						std::move(ds_layout),
						nullptr,
						nullptr,
						nullptr);
				}
				break;
				case EPipeline::ANIMATE_SPECTRUM:
				{
					ds_layout = animating_ds_layout;
					asset::SPushConstantRange range;
					range.size = sizeof(dimension2du) + sizeof(float) + sizeof(m_params.length_unit);
					range.offset = 0u;
					range.stageFlags = asset::ISpecializedShader::ESS_COMPUTE;
					layout = m_driver->createGPUPipelineLayout(&range,
						&range + 1,
						std::move(ds_layout),
						nullptr,
						nullptr,
						nullptr);
				}
				break;
				}

			}
			auto shader = createShader(pipeline_type);


			comp_pipeline = m_driver->createGPUComputePipeline(nullptr, std::move(layout), std::move(shader));
		}

		return comp_pipeline;
	};
	m_normalmap_generating_pipeline = createComputePipeline(EPipeline::GENERATE_NORMALMAP);
	m_spectrum_animating_pipeline = createComputePipeline(EPipeline::ANIMATE_SPECTRUM); 
	m_spectrum_randomizing_pipeline = createComputePipeline(EPipeline::RANDOMIZE_SPECTRUM);
	m_ifft_pipeline_1 = createComputePipeline(EPipeline::IFFT_STAGE_1);
	m_ifft_pipeline_2 = createComputePipeline(EPipeline::IFFT_STAGE_2);
	return m_spectrum_randomizing_pipeline.get() != nullptr &&
		m_ifft_pipeline_1.get() != nullptr &&
		m_ifft_pipeline_2.get() != nullptr &&
		m_normalmap_generating_pipeline.get() != nullptr &&
		m_spectrum_animating_pipeline.get() != nullptr;
}

WaveSimApp::textureView WaveSimApp::CreateTexture(nbl::core::dimension2du size, nbl::asset::E_FORMAT format) const
{
	nbl::video::IGPUImage::SCreationParams gpu_image_params;
	gpu_image_params.mipLevels = 1;
	gpu_image_params.extent = { size.Width, size.Height, 1 };
	gpu_image_params.format = format;
	gpu_image_params.arrayLayers = 1u;
	gpu_image_params.type = nbl::asset::IImage::ET_2D;
	gpu_image_params.samples = nbl::asset::IImage::ESCF_1_BIT;
	gpu_image_params.flags = static_cast<nbl::asset::IImage::E_CREATE_FLAGS>(0u);
	nbl::core::smart_refctd_ptr<nbl::video::IGPUImage> tex = m_driver->createGPUImageOnDedMem(std::move(gpu_image_params), m_driver->getDeviceLocalGPUMemoryReqs());

	nbl::video::IGPUImageView::SCreationParams creation_params;
	creation_params.format = tex->getCreationParameters().format;
	creation_params.image = tex;
	creation_params.viewType = nbl::video::IGPUImageView::ET_2D;
	creation_params.subresourceRange = { static_cast<nbl::asset::IImage::E_ASPECT_FLAGS>(0u), 0, 1, 0, 1 };
	creation_params.flags = static_cast<nbl::video::IGPUImageView::E_CREATE_FLAGS>(0u);
	nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> image_view = m_driver->createGPUImageView(std::move(creation_params));
	return image_view;
}

void WaveSimApp::PresentWaves2D(const textureView& tex)
{
	auto sampler_descriptor_set = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr(m_gpu_descriptor_set_layout));

	IGPUDescriptorSet::SDescriptorInfo info;
	{
		info.desc = tex;
		ISampler::SParams samplerParams = { ISampler::ETC_REPEAT, ISampler::ETC_REPEAT, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
		info.image.sampler = m_driver->createGPUSampler(samplerParams);
		info.image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
	}
	{
		IGPUDescriptorSet::SWriteDescriptorSet write;
		write.dstSet = sampler_descriptor_set.get();
		write.binding = 0u;
		write.arrayElement = 0u;
		write.count = 1u;
		write.descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
		write.info = &info;

		m_driver->updateDescriptorSets(1u, &write, 0u, nullptr);
	}

	{
		m_driver->bindGraphicsPipeline(m_presenting_pipeline.get());
		m_driver->bindDescriptorSets(EPBP_GRAPHICS, m_presenting_pipeline->getLayout(), 3u, 1u, &sampler_descriptor_set.get(), nullptr);
		m_driver->drawMeshBuffer(m_current_gpu_mesh_buffer.get());
		COpenGLExtensionHandler::pGlMemoryBarrier(GL_ALL_BARRIER_BITS);

	}
}

smart_refctd_ptr<nbl::video::IGPUBuffer> WaveSimApp::RandomizeWaveSpectrum()
{
	os::Printer::log("Randomizing waves spectrum");
	const uint32_t SSBO_SIZE = m_params.width * m_params.height * 6 * sizeof(float);

	auto initial_buffer = m_driver->createDeviceLocalGPUBufferOnDedMem(SSBO_SIZE);

	{
		video::IGPUDescriptorSet::SWriteDescriptorSet write;
		video::IGPUDescriptorSet::SDescriptorInfo info;
		write.dstSet = m_randomizer_descriptor_set.get();
		write.binding = 0u;
		write.count = 1;
		write.arrayElement = 0u;
		write.descriptorType = asset::EDT_STORAGE_BUFFER;
		info.desc = initial_buffer;
		info.buffer = { 0, SSBO_SIZE };

		write.info = &info;
		m_driver->updateDescriptorSets(1u, &write, 0u, nullptr);
	}
	{
		auto ds = m_randomizer_descriptor_set.get();
		m_driver->bindDescriptorSets(video::EPBP_COMPUTE, m_spectrum_randomizing_pipeline->getLayout(), 0u, 1u, &ds, nullptr);
		m_driver->bindComputePipeline(m_spectrum_randomizing_pipeline.get());
		m_driver->pushConstants(m_spectrum_randomizing_pipeline->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(m_params), &m_params);
		m_driver->dispatch((m_params.width + 15u) / 16u, (m_params.height + 15u) / 16u, 1u);
	}
	return initial_buffer;
}

void WaveSimApp::AnimateSpectrum(const nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer>& h0, nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer>& animated_spectrum, float time)
{
	const uint32_t IN_SSBO_SIZE = m_params.width * m_params.height * 4 * sizeof(float);
	const uint32_t OUT_SSBO_SIZE = m_params.width * m_params.height * 2 * sizeof(float);
	{
		video::IGPUDescriptorSet::SWriteDescriptorSet write[3];
		video::IGPUDescriptorSet::SDescriptorInfo info[3];
		write[0].dstSet = m_spectrum_animating_descriptor_set.get();
		write[0].binding = 0u;
		write[0].count = 1;
		write[0].arrayElement = 0u;
		write[0].descriptorType = asset::EDT_STORAGE_BUFFER;
		write[0].info = info;
		info[0].desc = h0;
		info[0].buffer = { 0, IN_SSBO_SIZE };

		write[1] = write[0];
		write[1].descriptorType = asset::EDT_STORAGE_BUFFER;
		write[1].binding = 1u;
		write[1].info = info + 1;
		info[1].desc = animated_spectrum;
		info[1].buffer = { 0, OUT_SSBO_SIZE };

		m_driver->updateDescriptorSets(2u, write, 0u, nullptr);
	}
	auto pc = [this, time]() {
		struct PC { dimension2du size; float time; vector2df length_unit; };
		return PC{ m_params.size, time, m_params.length_unit };
	} ();
	auto ds = m_spectrum_animating_descriptor_set.get();
	m_driver->bindDescriptorSets(video::EPBP_COMPUTE, m_spectrum_animating_pipeline->getLayout(), 0u, 1u, &ds, nullptr);
	m_driver->bindComputePipeline(m_spectrum_animating_pipeline.get());
	m_driver->pushConstants(m_spectrum_animating_pipeline->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(pc), &pc);
	{
		m_driver->dispatch((m_params.width + 15u) / 16u, (m_params.height + 15u) / 16u, 1u);
		COpenGLExtensionHandler::pGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	}
}

void WaveSimApp::GenerateHeightMap(const smart_refctd_ptr<nbl::video::IGPUBuffer>& h0, textureView& out, float time)
{
	using namespace ext::FFT;
	const uint32_t SSBO_SIZE = m_params.width * m_params.height * 2 * sizeof(float);
	auto animated_data_buffer = m_driver->createDeviceLocalGPUBufferOnDedMem(SSBO_SIZE);
	AnimateSpectrum(h0, animated_data_buffer, time);
	auto ifft_x_buffer = m_driver->createDeviceLocalGPUBufferOnDedMem(SSBO_SIZE);
	{
		video::IGPUDescriptorSet::SWriteDescriptorSet write[2];
		video::IGPUDescriptorSet::SDescriptorInfo info[2];
		write[0].dstSet = m_ifft_1_descriptor_set.get();
		write[0].binding = 0u;
		write[0].count = 1;
		write[0].arrayElement = 0u;
		write[0].descriptorType = asset::EDT_STORAGE_BUFFER;
		write[0].info = info;
		info[0].desc = animated_data_buffer;
		info[0].buffer = { 0, SSBO_SIZE };

		write[1] = write[0];
		write[1].descriptorType = asset::EDT_STORAGE_BUFFER;
		write[1].binding = 1u;
		write[1].info = info + 1;
		info[1].desc = ifft_x_buffer;
		info[1].buffer = { 0, SSBO_SIZE };

		m_driver->updateDescriptorSets(2u, write, 0u, nullptr);
	}
	{
		struct
		{
			FFT::Parameters_t params;
			dimension2du size;
			float time;
		} pc;
		pc.time = time;
		pc.size = m_params.size;
		uint8_t isInverse_u8 = false;
		uint8_t direction_u8 = static_cast<uint8_t>(FFT::Direction::Y);
		uint8_t paddingType_u8 = static_cast<uint8_t>(FFT::PaddingType::CLAMP_TO_EDGE);

		uint32_t packed = (direction_u8 << 16u) | (isInverse_u8 << 8u) | paddingType_u8;

		pc.params.dimension.x = m_params.width;
		pc.params.dimension.y = m_params.height;
		pc.params.dimension.z = 1;
		pc.params.dimension.w = packed;
		pc.params.padded_dimension.x = pc.params.dimension.x;
		pc.params.padded_dimension.y = pc.params.dimension.y;
		pc.params.padded_dimension.z = pc.params.dimension.z;
		pc.params.padded_dimension.w = 1;
		auto dispatch_info = FFT::buildParameters({ pc.params.dimension.x,
													pc.params.dimension.y,
													pc.params.dimension.z }, FFT::Direction::Y);

		auto ds = m_ifft_1_descriptor_set.get();
		m_driver->bindDescriptorSets(video::EPBP_COMPUTE, m_ifft_pipeline_1->getLayout(), 0u, 1u, &ds, nullptr);
		m_driver->bindComputePipeline(m_ifft_pipeline_1.get());
		m_driver->pushConstants(m_ifft_pipeline_1->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(pc), &pc);
		{
			m_driver->dispatch(dispatch_info.workGroupCount[0], dispatch_info.workGroupCount[1], dispatch_info.workGroupCount[2]);
			COpenGLExtensionHandler::pGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		}
	}

	{
		video::IGPUDescriptorSet::SWriteDescriptorSet write[2];
		video::IGPUDescriptorSet::SDescriptorInfo info[2];
		write[0].dstSet = m_ifft_2_descriptor_set.get();
		write[0].binding = 0u;
		write[0].count = 1;
		write[0].arrayElement = 0u;
		write[0].descriptorType = asset::EDT_STORAGE_BUFFER;
		write[0].info = info;
		info[0].desc = ifft_x_buffer;
		info[0].buffer = { 0, SSBO_SIZE };

		write[1] = write[0];
		write[1].descriptorType = asset::EDT_STORAGE_IMAGE;
		write[1].binding = 1u;
		write[1].info = info + 1;
		info[1].desc = out;
		info[1].image = { nullptr, EIL_UNDEFINED };

		m_driver->updateDescriptorSets(2u, write, 0u, nullptr);
	}

	{
		FFT::Parameters_t params;
		uint8_t isInverse_u8 = false;
		uint8_t direction_u8 = static_cast<uint8_t>(FFT::Direction::X);
		uint8_t paddingType_u8 = static_cast<uint8_t>(FFT::PaddingType::CLAMP_TO_EDGE);

		uint32_t packed = (direction_u8 << 16u) | (isInverse_u8 << 8u) | paddingType_u8;

		params.dimension.x = m_params.width;
		params.dimension.y = m_params.height;
		params.dimension.z = 1;
		params.dimension.w = packed;
		params.padded_dimension.x = params.dimension.x;
		params.padded_dimension.y = params.dimension.y;
		params.padded_dimension.z = params.dimension.z;
		params.padded_dimension.w = 1;

		auto dispatch_info = FFT::buildParameters({ params.dimension.x,
													params.dimension.y,
													params.dimension.z }, FFT::Direction::X);

		auto ds = m_ifft_2_descriptor_set.get();
		m_driver->bindDescriptorSets(video::EPBP_COMPUTE, m_ifft_pipeline_2->getLayout(), 0u, 1u, &ds, nullptr);
		m_driver->bindComputePipeline(m_ifft_pipeline_2.get());
		m_driver->pushConstants(m_ifft_pipeline_2->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(params), &params);
		{
			m_driver->dispatch(dispatch_info.workGroupCount[0], dispatch_info.workGroupCount[1], dispatch_info.workGroupCount[2]);
			COpenGLExtensionHandler::pGlMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
		}

	}
}

void WaveSimApp::GenerateNormalMap(const textureView& heightmap, textureView& normalmap)
{
	{
		video::IGPUDescriptorSet::SWriteDescriptorSet write[2];
		video::IGPUDescriptorSet::SDescriptorInfo info[2];
		write[0].dstSet = m_normalmap_descriptor_set.get();
		write[0].binding = 0u;
		write[0].count = 1;
		write[0].arrayElement = 0u;
		write[0].descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
		write[0].info = info;
		info[0].desc = heightmap;
		ISampler::SParams samplerParams = { ISampler::ETC_REPEAT, ISampler::ETC_REPEAT, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
		info[0].image.sampler = m_driver->createGPUSampler(samplerParams);
		info[0].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;

		write[1] = write[0];
		write[1].descriptorType = asset::EDT_STORAGE_IMAGE;
		write[1].binding = 1u;
		write[1].info = info + 1;
		info[1].desc = normalmap;
		info[1].image = { nullptr, EIL_UNDEFINED };

		m_driver->updateDescriptorSets(2u, write, 0u, nullptr);
	}
	{
		auto ds = m_normalmap_descriptor_set.get();
		m_driver->bindDescriptorSets(video::EPBP_COMPUTE, m_normalmap_generating_pipeline->getLayout(), 0u, 1u, &ds, nullptr);
		m_driver->bindComputePipeline(m_normalmap_generating_pipeline.get());
		m_driver->pushConstants(m_normalmap_generating_pipeline->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(m_params.size), &m_params.size);
		m_driver->dispatch((m_params.width + 15u) / 16u, (m_params.height + 15u) / 16u, 1u);
		COpenGLExtensionHandler::pGlMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	}

}



WaveSimApp::WaveSimApp(const WaveSimParams& params) : m_params{ params }
{
	assert(isPoT(params.width) && isPoT(params.height));
	// For some reason cannot call nbl::core::normalize
	auto normalize = [](vector2df vec)
	{
		float length = sqrt(vec.X * vec.X + vec.Y * vec.Y);
		return vec / length;
	};
	m_params.wind_dir = normalize(m_params.wind_dir);

	bool initialized = Init();
	initialized &= CreatePresentingPipeline();
	initialized &= CreateComputePipelines();
	assert(initialized);
}

void WaveSimApp::Run()
{
	float i = 1;

	auto initial_values = RandomizeWaveSpectrum();
	auto heightmap = CreateTexture(m_params.size, EF_R8_UNORM);
	auto normalmap = CreateTexture(m_params.size, EF_R8G8B8A8_UNORM);
	auto start_time = std::chrono::system_clock::now();
	while (m_device->run())
	{
		std::chrono::duration<double> time_passed = std::chrono::system_clock::now() - start_time;
		m_driver->beginScene(true);
		GenerateHeightMap(initial_values, heightmap, std::chrono::duration_cast<std::chrono::milliseconds>(time_passed).count() / 1000.f);
		GenerateNormalMap(heightmap, normalmap);
		PresentWaves2D(normalmap);
		m_driver->endScene();

	}
}
