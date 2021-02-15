#include "WaveSimApp.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"

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
	params.WindowSize = dimension2d<uint32_t>(1600, 900);
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
		SPECTRUM_RANDOMIZE,
		ANIMATE_STAGE_1,
		ANIMATE_STAGE_2
	};
	auto getShaderPath = [](EPipeline type)
	{
		switch (type)
		{
		case EPipeline::SPECTRUM_RANDOMIZE:
			return "../spectrum_randomizer.comp";
		case EPipeline::ANIMATE_STAGE_1:
			return "../animate_1.comp";
		case EPipeline::ANIMATE_STAGE_2:
			return "../animate_2.comp";
		default:
			os::Printer::log("Unsupporded pipeline");
			return "";
		}
	};

	// Spectrum generation layout
	{
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> ds_layout;
		IGPUDescriptorSetLayout::SBinding texture_bindings[1];
		texture_bindings[0].binding = 0;
		texture_bindings[0].type = EDT_STORAGE_BUFFER;
		texture_bindings[0].count = 1u;
		texture_bindings[0].stageFlags = static_cast<IGPUSpecializedShader::E_SHADER_STAGE>(IGPUSpecializedShader::ESS_COMPUTE);
		texture_bindings[0].samplers = nullptr;

		ds_layout = m_driver->createGPUDescriptorSetLayout(texture_bindings, texture_bindings + 1);
		m_randomizer_descriptor_set = m_driver->createGPUDescriptorSet(ds_layout);
	}
	{
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> ds_layout;
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

		ds_layout = m_driver->createGPUDescriptorSetLayout(texture_bindings, texture_bindings + 2);
		m_animating_1_descriptor_set = m_driver->createGPUDescriptorSet(ds_layout);
	}

	auto createComputePipeline = [&](EPipeline pipeline_type)
	{
		std::string filepath = getShaderPath(pipeline_type);

		core::smart_refctd_ptr<video::IGPUComputePipeline> comp_pipeline;
		{
			core::smart_refctd_ptr<video::IGPUPipelineLayout> layout;
			{

				smart_refctd_ptr<IGPUDescriptorSetLayout> ds_layout;
				switch (pipeline_type)
				{
				case EPipeline::SPECTRUM_RANDOMIZE:
				{
					IGPUDescriptorSetLayout::SBinding texture_bindings[1];
					texture_bindings[0].binding = 0;
					texture_bindings[0].type = EDT_STORAGE_BUFFER;
					texture_bindings[0].count = 1u;
					texture_bindings[0].stageFlags = static_cast<IGPUSpecializedShader::E_SHADER_STAGE>(IGPUSpecializedShader::ESS_COMPUTE);
					texture_bindings[0].samplers = nullptr;

					ds_layout = m_driver->createGPUDescriptorSetLayout(texture_bindings, texture_bindings + 1);
				}
				break;
				case EPipeline::ANIMATE_STAGE_1:
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

					ds_layout = m_driver->createGPUDescriptorSetLayout(texture_bindings, texture_bindings + 2);
				}
				break;
				}
				switch (pipeline_type)
				{
				case EPipeline::SPECTRUM_RANDOMIZE:
				{
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
				case EPipeline::ANIMATE_STAGE_1:
				{
					asset::SPushConstantRange range;
					range.size = sizeof(float) + sizeof(uint32_t) * 2;
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
			core::smart_refctd_ptr<video::IGPUSpecializedShader> shader;
			{
				auto f = core::smart_refctd_ptr<io::IReadFile>(m_filesystem->createAndOpenFile(filepath.c_str()));

				asset::IAssetLoader::SAssetLoadParams lp;
				auto cs_bundle = m_asset_manager->getAsset(filepath.c_str(), lp);
				auto cs = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*cs_bundle.getContents().begin());

				auto cs_rawptr = cs.get();
				shader = m_driver->getGPUObjectsFromAssets(&cs_rawptr, &cs_rawptr + 1)->front();
			}

			comp_pipeline = m_driver->createGPUComputePipeline(nullptr, std::move(layout), std::move(shader));
		}

		return comp_pipeline;
	};
	m_spectrum_randomizing_pipeline = createComputePipeline(EPipeline::SPECTRUM_RANDOMIZE);
	m_animating_pipeline_1 = createComputePipeline(EPipeline::ANIMATE_STAGE_1);
	return m_spectrum_randomizing_pipeline.get() != nullptr &&
		m_animating_pipeline_1.get() != nullptr;
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

void WaveSimApp::PresentWaves(const textureView& tex)
{
	auto sampler_descriptor_set = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr(m_gpu_descriptor_set_layout));

	IGPUDescriptorSet::SDescriptorInfo info;
	{
		info.desc = tex;
		ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
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

	//while (m_device->run())
	{
		m_driver->beginScene(true, true);


		m_driver->bindGraphicsPipeline(m_presenting_pipeline.get());
		m_driver->bindDescriptorSets(EPBP_GRAPHICS, m_presenting_pipeline->getLayout(), 3u, 1u, &sampler_descriptor_set.get(), nullptr);
		m_driver->drawMeshBuffer(m_current_gpu_mesh_buffer.get());

		m_driver->endScene();
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

void WaveSimApp::GetAnimatedHeightMap(const smart_refctd_ptr<nbl::video::IGPUBuffer>& h0, textureView& out, float time)
{
	const uint32_t SSBO_SIZE = m_params.width * m_params.height * 6 * sizeof(float);

	{
		video::IGPUDescriptorSet::SWriteDescriptorSet write[2];
		video::IGPUDescriptorSet::SDescriptorInfo info[2];
		write[0].dstSet = m_animating_1_descriptor_set.get();
		write[0].binding = 0u;
		write[0].count = 1;
		write[0].arrayElement = 0u;
		write[0].descriptorType = asset::EDT_STORAGE_BUFFER;
		info[0].desc = h0;
		info[0].buffer = { 0, SSBO_SIZE };
		write[0].info = info;

		write[1] = write[0];
		write[1].descriptorType = asset::EDT_STORAGE_IMAGE;
		write[1].binding = 1u;
		info[1].desc = out;
		info[1].image = { nullptr, EIL_UNDEFINED };
		write[1].info = info + 1;

		m_driver->updateDescriptorSets(2u, write, 0u, nullptr);
	}

	struct
	{
		float time;
		dimension2du size;
	} pc;
	pc.time = time;
	pc.size = m_params.size;
	{
		m_driver->beginScene(true);
		auto ds = m_animating_1_descriptor_set.get();
		m_driver->bindDescriptorSets(video::EPBP_COMPUTE, m_animating_pipeline_1->getLayout(), 0u, 1u, &ds, nullptr);
		m_driver->bindComputePipeline(m_animating_pipeline_1.get());
		m_driver->pushConstants(m_animating_pipeline_1->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(pc), &pc);
		m_driver->dispatch((m_params.width + 15u) / 16u, (m_params.height + 15u) / 16u, 1u);
		m_driver->endScene();
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
	m_params.m_wind_dir = normalize(m_params.m_wind_dir);

	bool initialized = Init();
	initialized &= CreatePresentingPipeline();
	initialized &= CreateComputePipelines();
	assert(initialized);
}

void WaveSimApp::Run()
{
	auto initial_values = RandomizeWaveSpectrum();
	auto animated_part = CreateTexture(m_params.size, EF_R8G8_UNORM);
	float i = 1;
	while (m_device->run())
	{
		GetAnimatedHeightMap(initial_values, animated_part, i+= 0.01f);
		PresentWaves(animated_part);
	}
}
