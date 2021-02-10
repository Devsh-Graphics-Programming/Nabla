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

void WaveSimApp::PresentWaves(textureView tex)
{
	os::Printer::log("Presenting waves simulation");

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

	while (m_device->run())
	{
		m_driver->beginScene(true, true);


		m_driver->bindGraphicsPipeline(m_presenting_pipeline.get());
		m_driver->bindDescriptorSets(EPBP_GRAPHICS, m_presenting_pipeline->getLayout(), 3u, 1u, &sampler_descriptor_set.get(), nullptr);
		m_driver->drawMeshBuffer(m_current_gpu_mesh_buffer.get());

		m_driver->endScene();
	}
}

WaveSimApp::WaveSimApp(const WaveSimParams& params) : m_params{ params }
{
	[[maybe_unused]] bool initialized = Init();
	initialized &= CreatePresentingPipeline();
	assert(initialized);
}

void WaveSimApp::Run()
{
	auto tex_view = CreateTexture(m_params.size, EF_R8G8_UNORM);
	PresentWaves(tex_view);
}
