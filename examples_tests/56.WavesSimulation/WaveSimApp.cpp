#include <chrono> 

#include "WaveSimApp.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/ext/FFT/FFT.h"
#include "../common/QToQuitEventReceiver.h"


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

	m_device->getCursorControl()->setVisible(false);
	m_device->setEventReceiver(&m_receiver);

	m_driver = m_device->getVideoDriver();
	m_filesystem = m_device->getFileSystem();
	m_asset_manager = m_device->getAssetManager();
	m_device->setWindowCaption(L"Tessendorf Waves Simulation");


	return true;
}

bool WaveSimApp::CreatePresenting2DPipeline()
{
	const char* fragment_shader_path = "../waves_display.frag";
	auto full_screen_triangle = ext::FullScreenTriangle::createFullScreenTriangle(m_device->getAssetManager(), m_device->getVideoDriver());
	IGPUDescriptorSetLayout::SBinding binding{ 0u, EDT_COMBINED_IMAGE_SAMPLER, 1u, IGPUSpecializedShader::ESS_FRAGMENT, nullptr };
	m_gpu_descriptor_set_layout_2d = m_driver->createGPUDescriptorSetLayout(&binding, &binding + 1u);

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

		auto gpu_pipeline_layout = m_driver->createGPUPipelineLayout(nullptr, nullptr, nullptr, nullptr, nullptr, core::smart_refctd_ptr(m_gpu_descriptor_set_layout_2d));

		return m_driver->createGPURenderpassIndependentPipeline(nullptr, std::move(gpu_pipeline_layout), shaders, shaders + sizeof(shaders) / sizeof(IGPUSpecializedShader*),
			std::get<SVertexInputParams>(full_screen_triangle), blend_params,
			std::get<SPrimitiveAssemblyParams>(full_screen_triangle), raster_params);
	};

	m_presenting_pipeline = createGPUPipeline(IImageView<ICPUImage>::E_TYPE::ET_2D);
	assert(m_presenting_pipeline.get());
	{
		SBufferBinding<IGPUBuffer> idxBinding{ 0ull, nullptr };
		m_2d_mesh_buffer = core::make_smart_refctd_ptr<IGPUMeshBuffer>(nullptr, nullptr, nullptr, std::move(idxBinding));
		m_2d_mesh_buffer->setIndexCount(3u);
		m_2d_mesh_buffer->setInstanceCount(1u);
	}
	return true;
}

bool WaveSimApp::CreatePresenting3DPipeline()
{
	struct VertexData
	{
		vector3df world_position;
		vector2df texture_position;
	};
	constexpr uint32_t INDICES_PER_QUAD = 6;
	const size_t INDEX_COUNT = (m_params.length - 1) * (m_params.width - 1) * 2 * 3;
	const size_t VERTEX_COUNT = m_params.length * m_params.width;

	uint32_t idx = 0;
	std::vector<uint32_t> indices(INDEX_COUNT);
	std::vector<VertexData> vertices(VERTEX_COUNT);
	for (int y = 0; y < m_params.length - 1; y++)
	{
		const size_t y_pos = y * m_params.width;
		const size_t y_plus_one_pos = (y + 1) * m_params.width;
		for (int x = 0; x < m_params.width - 1; x++)
		{
			indices[idx++] = y_pos + x;
			indices[idx++] = y_plus_one_pos + x;
			indices[idx++] = y_plus_one_pos + x + 1;
			indices[idx++] = y_pos + x;
			indices[idx++] = y_plus_one_pos + x + 1;
			indices[idx++] = y_pos + x + 1;
		}

	}
	for (int z = 0; z < m_params.length; z++)
	{
		float z_world = z;
		size_t z_pos = z * m_params.width;
		for (int x = 0; x < m_params.width; x++)
		{
			float x_world = x;
			vertices[z_pos + x] = { vector3df{x_world / m_params.width, 0.f, z_world / m_params.length}, vector2df{float(x) / m_params.width, float(z) / m_params.length } };
		}
	}
	auto up_stream_buff = m_driver->getDefaultUpStreamingBuffer();
	core::smart_refctd_ptr<video::IGPUBuffer> up_stream_ref(up_stream_buff->getBuffer());
	const void* data_to_place[2] = { vertices.data(), indices.data() };
	uint32_t offsets[2] = { video::StreamingTransientDataBufferMT<>::invalid_address,video::StreamingTransientDataBufferMT<>::invalid_address };
	uint32_t alignments[2] = { sizeof(decltype(vertices[0u])),sizeof(decltype(indices[0u])) };
	uint32_t sizes[2] = { VERTEX_COUNT * sizeof(VertexData), INDEX_COUNT * sizeof(uint32_t) };
	up_stream_buff->multi_place(2u, (const void* const*)data_to_place, (uint32_t*)offsets, (uint32_t*)sizes, (uint32_t*)alignments);
	if (up_stream_buff->needsManualFlushOrInvalidate())
	{
		auto up_stream_mem = up_stream_buff->getBuffer()->getBoundMemory();
		m_driver->flushMappedMemoryRanges({ video::IDriverMemoryAllocation::MappedMemoryRange(up_stream_mem,offsets[0],sizes[0]),video::IDriverMemoryAllocation::MappedMemoryRange(up_stream_mem,offsets[1],sizes[1]) });
	}

	nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> presenting_layout;
	{
		IGPUDescriptorSetLayout::SBinding texture_bindings[3];
		texture_bindings[0].binding = 0;
		texture_bindings[0].type = EDT_COMBINED_IMAGE_SAMPLER;
		texture_bindings[0].count = 1u;
		texture_bindings[0].stageFlags = static_cast<IGPUSpecializedShader::E_SHADER_STAGE>(IGPUSpecializedShader::ESS_VERTEX);
		texture_bindings[0].samplers = nullptr;

		texture_bindings[1].binding = 1;
		texture_bindings[1].type = EDT_COMBINED_IMAGE_SAMPLER;
		texture_bindings[1].count = 1u;
		texture_bindings[1].stageFlags = static_cast<IGPUSpecializedShader::E_SHADER_STAGE>(IGPUSpecializedShader::ESS_FRAGMENT);
		texture_bindings[1].samplers = nullptr;

		texture_bindings[2].binding = 2;
		texture_bindings[2].type = EDT_COMBINED_IMAGE_SAMPLER;
		texture_bindings[2].count = 1u;
		texture_bindings[2].stageFlags = static_cast<IGPUSpecializedShader::E_SHADER_STAGE>(IGPUSpecializedShader::ESS_FRAGMENT);
		texture_bindings[2].samplers = nullptr;

		presenting_layout = m_driver->createGPUDescriptorSetLayout(texture_bindings, texture_bindings + 3);
		m_3d_presenting_descriptor_set = m_driver->createGPUDescriptorSet(presenting_layout);
	}

	const char* vertex_shader_path = "../waves_display_3d.vert";
	const char* fragment_shader_path = "../waves_display_3d.frag";


	asset::SPushConstantRange ranges[2] = { { asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD) },
										    { asset::ISpecializedShader::ESS_FRAGMENT, sizeof(core::matrix4SIMD), sizeof(core::vector3df) } };

	core::smart_refctd_ptr<video::IGPUSpecializedShader> shaders[2] =
	{
		createGPUSpecializedShaderFromFile(vertex_shader_path, asset::ISpecializedShader::ESS_VERTEX),
		createGPUSpecializedShaderFromFileWithIncludes(fragment_shader_path, asset::ISpecializedShader::ESS_FRAGMENT, fragment_shader_path)
	};
	auto shadersPtr = reinterpret_cast<video::IGPUSpecializedShader**>(shaders);

	asset::SVertexInputParams inputParams;
	inputParams.enabledAttribFlags = 0b11u;
	inputParams.enabledBindingFlags = 0b1u;
	inputParams.attributes[0].binding = 0u;
	inputParams.attributes[0].format = asset::EF_R32G32B32_SFLOAT;
	inputParams.attributes[0].relativeOffset = offsetof(VertexData, world_position);
	inputParams.attributes[1].binding = 0u;
	inputParams.attributes[1].format = asset::EF_R32G32_SFLOAT;
	inputParams.attributes[1].relativeOffset = offsetof(VertexData, texture_position);
	inputParams.bindings[0].stride = sizeof(VertexData);
	inputParams.bindings[0].inputRate = asset::EVIR_PER_VERTEX;

	asset::SBlendParams blendParams;

	asset::SPrimitiveAssemblyParams assemblyParams = { asset::EPT_TRIANGLE_LIST,false,1u };

	asset::SStencilOpParams defaultStencil;
	asset::SRasterizationParams rasterParams;
	rasterParams.faceCullingMode = asset::EFCM_NONE;

	auto pipeline_layout = m_driver->createGPUPipelineLayout(ranges, ranges + 2u, std::move(presenting_layout), nullptr, nullptr, nullptr);
	auto pipeline = m_driver->createGPURenderpassIndependentPipeline(nullptr, std::move(pipeline_layout),
		shadersPtr, shadersPtr + sizeof(shaders) / sizeof(core::smart_refctd_ptr<video::IGPUSpecializedShader>),
		inputParams, blendParams, assemblyParams, rasterParams);

	asset::SBufferBinding<video::IGPUBuffer> bindings[video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
	bindings[0u] = { offsets[0],up_stream_ref };
	auto mb = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(std::move(pipeline), nullptr, bindings, asset::SBufferBinding<video::IGPUBuffer>{offsets[1], up_stream_ref});
	{
		mb->setIndexType(asset::EIT_32BIT);
		mb->setIndexCount(INDEX_COUNT);
	}

	m_3d_mesh_buffer = std::move(mb);
	up_stream_buff->multi_free(2u, (uint32_t*)&offsets, (uint32_t*)&sizes, m_driver->placeFence());
	return m_3d_mesh_buffer.get() != nullptr;

}

bool WaveSimApp::CreateSkyboxPresentingPipeline()
{
	nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> presenting_layout;
	{
		IGPUDescriptorSetLayout::SBinding texture_bindings[1];
		texture_bindings[0].binding = 0;
		texture_bindings[0].type = EDT_COMBINED_IMAGE_SAMPLER;
		texture_bindings[0].count = 1u;
		texture_bindings[0].stageFlags = static_cast<IGPUSpecializedShader::E_SHADER_STAGE>(IGPUSpecializedShader::ESS_FRAGMENT);
		texture_bindings[0].samplers = nullptr;

		presenting_layout = m_driver->createGPUDescriptorSetLayout(texture_bindings, texture_bindings + 1);
		m_skybox_presenting_descriptor_set = m_driver->createGPUDescriptorSet(presenting_layout);
	}

	const char* vertex_shader_path = "../skybox.vert";
	const char* fragment_shader_path = "../skybox.frag";
	auto sphereGeometry = m_device->getAssetManager()->getGeometryCreator()->createSphereMesh(300, 16, 16);

	core::smart_refctd_ptr<video::IGPUSpecializedShader> shaders[2] =
	{
		createGPUSpecializedShaderFromFile(vertex_shader_path, asset::ISpecializedShader::ESS_VERTEX),
		createGPUSpecializedShaderFromFileWithIncludes(fragment_shader_path, asset::ISpecializedShader::ESS_FRAGMENT, fragment_shader_path)
	};
	auto shadersPtr = reinterpret_cast<video::IGPUSpecializedShader**>(shaders);

	auto createGPUMeshBufferAndItsPipeline = [&](asset::IGeometryCreator::return_type& geometryObject)
	{
		asset::SBlendParams blendParams;
		asset::SRasterizationParams rasterParams;
		rasterParams.faceCullingMode = asset::EFCM_NONE;

		asset::SPushConstantRange range[1] = { asset::ISpecializedShader::ESS_VERTEX,0u,sizeof(core::matrix4SIMD) };
		auto pipeline = m_driver->createGPURenderpassIndependentPipeline(nullptr, m_driver->createGPUPipelineLayout(range, range + 1u, std::move(presenting_layout), nullptr, nullptr, nullptr),
			shadersPtr, shadersPtr + sizeof(shaders) / sizeof(core::smart_refctd_ptr<video::IGPUSpecializedShader>),
			geometryObject.inputParams, blendParams, geometryObject.assemblyParams, rasterParams);

		constexpr auto MAX_ATTR_BUF_BINDING_COUNT = video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT;
		constexpr auto MAX_DATA_BUFFERS = MAX_ATTR_BUF_BINDING_COUNT + 1;
		core::vector<asset::ICPUBuffer*> cpubuffers;
		cpubuffers.reserve(MAX_DATA_BUFFERS);
		for (auto i = 0; i < MAX_ATTR_BUF_BINDING_COUNT; i++)
		{
			auto buf = geometryObject.bindings[i].buffer.get();
			if (buf)
				cpubuffers.push_back(buf);
		}
		auto cpuindexbuffer = geometryObject.indexBuffer.buffer.get();
		if (cpuindexbuffer)
			cpubuffers.push_back(cpuindexbuffer);

		auto gpubuffers = m_driver->getGPUObjectsFromAssets(cpubuffers.data(), cpubuffers.data() + cpubuffers.size());

		asset::SBufferBinding<video::IGPUBuffer> bindings[MAX_DATA_BUFFERS];
		for (auto i = 0, j = 0; i < MAX_ATTR_BUF_BINDING_COUNT; i++)
		{
			if (!geometryObject.bindings[i].buffer)
				continue;
			auto buffPair = gpubuffers->operator[](j++);
			bindings[i].offset = buffPair->getOffset();
			bindings[i].buffer = core::smart_refctd_ptr<video::IGPUBuffer>(buffPair->getBuffer());
		}
		if (cpuindexbuffer)
		{
			auto buffPair = gpubuffers->back();
			bindings[MAX_ATTR_BUF_BINDING_COUNT].offset = buffPair->getOffset();
			bindings[MAX_ATTR_BUF_BINDING_COUNT].buffer = core::smart_refctd_ptr<video::IGPUBuffer>(buffPair->getBuffer());
		}

		auto mb = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(core::smart_refctd_ptr(pipeline), nullptr, bindings, std::move(bindings[MAX_ATTR_BUF_BINDING_COUNT]));
		{
			mb->setIndexType(geometryObject.indexType);
			mb->setIndexCount(geometryObject.indexCount);
			mb->setBoundingBox(geometryObject.bbox);
		}

		return mb;
	};

	auto gpuSphere = createGPUMeshBufferAndItsPipeline(sphereGeometry);
	m_gpu_sphere = gpuSphere;

	return gpuSphere.get() != nullptr;

}

bool WaveSimApp::CreateComputePipelines()
{
	enum class EPipeline
	{
		GENERATE_SPECTRUM,
		IFFT_STAGE_1,
		IFFT_STAGE_2,
		GENERATE_NORMALMAP
	};
	auto getFilePath = [](EPipeline type)
	{
		switch (type)
		{
		case EPipeline::GENERATE_SPECTRUM:
			return "../initial_spectrum.comp";
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
		case EPipeline::GENERATE_SPECTRUM:
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
			constexpr uint32_t DEFAULT_WORK_GROUP_SIZE = 256u;
			uint32_t maxPaddedDimensionSize = std::max(m_params.length, m_params.width);
			const uint32_t maxItemsPerThread = ((maxPaddedDimensionSize >> 1) - 1u) / (DEFAULT_WORK_GROUP_SIZE)+1u;
			const char* sourceFmt =
				R"===(#version 450

#define _NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_ %u
#define _NBL_GLSL_EXT_FFT_MAX_DIM_SIZE_ %u
#define _NBL_GLSL_EXT_FFT_MAX_ITEMS_PER_THREAD %u
#define _NBL_GLSL_EXT_FFT_HALF_STORAGE_ %u

#include "../ifft_x.comp"

)===";
			auto shader = IGLSLCompiler::createOverridenCopy(nullptr, sourceFmt, DEFAULT_WORK_GROUP_SIZE, maxPaddedDimensionSize, maxItemsPerThread, 1);

			auto cpuSpecializedShader = core::make_smart_refctd_ptr<ICPUSpecializedShader>(
				std::move(shader),
				ISpecializedShader::SInfo{ nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE }
			);

			auto gpuShader = m_driver->createGPUShader(nbl::core::smart_refctd_ptr<const ICPUShader>(cpuSpecializedShader->getUnspecialized()));

			auto gpuSpecializedShader = m_driver->createGPUSpecializedShader(gpuShader.get(), cpuSpecializedShader->getSpecializationInfo());

			return gpuSpecializedShader;
		}
		case EPipeline::IFFT_STAGE_2:
		{
			constexpr uint32_t DEFAULT_WORK_GROUP_SIZE = 256u;
			uint32_t maxPaddedDimensionSize = std::max(m_params.length, m_params.width);
			const uint32_t maxItemsPerThread = ((maxPaddedDimensionSize >> 1) - 1u) / (DEFAULT_WORK_GROUP_SIZE)+1u;
			const char* sourceFmt =
				R"===(#version 450

#define USE_SSBO_FOR_INPUT 1
#define _NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_ %u
#define _NBL_GLSL_EXT_FFT_MAX_DIM_SIZE_ %u
#define _NBL_GLSL_EXT_FFT_MAX_ITEMS_PER_THREAD %u
#define _NBL_GLSL_EXT_FFT_HALF_STORAGE_ %u
 
#include "../ifft_y.comp"

)===";
			auto shader = IGLSLCompiler::createOverridenCopy(nullptr, sourceFmt, DEFAULT_WORK_GROUP_SIZE, maxPaddedDimensionSize, maxItemsPerThread, 1);

			auto cpuSpecializedShader = core::make_smart_refctd_ptr<ICPUSpecializedShader>(
				std::move(shader),
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

	auto createComputePipeline = [&](EPipeline pipeline_type)
	{
		core::smart_refctd_ptr<video::IGPUComputePipeline> comp_pipeline;
		{
			core::smart_refctd_ptr<video::IGPUPipelineLayout> layout;
			{

				smart_refctd_ptr<IGPUDescriptorSetLayout> ds_layout;
				switch (pipeline_type)
				{
				case EPipeline::GENERATE_SPECTRUM:
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
					range.size = sizeof(float) * 3 + sizeof(uint32_t) * 2 + sizeof(ext::FFT::FFT::Parameters_t);
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
					range.size = sizeof(ext::FFT::FFT::Parameters_t) + sizeof(float);
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
				}

			}
			auto shader = createShader(pipeline_type);


			comp_pipeline = m_driver->createGPUComputePipeline(nullptr, std::move(layout), std::move(shader));
		}

		return comp_pipeline;
	};
	m_normalmap_generating_pipeline = createComputePipeline(EPipeline::GENERATE_NORMALMAP);
	m_spectrum_randomizing_pipeline = createComputePipeline(EPipeline::GENERATE_SPECTRUM);
	m_ifft_pipeline_1 = createComputePipeline(EPipeline::IFFT_STAGE_1);
	m_ifft_pipeline_2 = createComputePipeline(EPipeline::IFFT_STAGE_2);
	return m_spectrum_randomizing_pipeline.get() != nullptr &&
		m_ifft_pipeline_1.get() != nullptr &&
		m_ifft_pipeline_2.get() != nullptr &&
		m_normalmap_generating_pipeline.get() != nullptr;
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

WaveSimApp::textureView WaveSimApp::CreateTextureFromImageFile(const std::string_view image_file_path, E_FORMAT format) const
{
	smart_refctd_ptr<ICPUImageView> cpu_image_view, copy_image_view;

	IAssetLoader::SAssetLoadParams lp(0ull, nullptr, IAssetLoader::ECF_DONT_CACHE_REFERENCES);
	auto cpu_texture = m_asset_manager->getAsset(image_file_path.data(), lp);
	auto cpu_texture_contents = cpu_texture.getContents();

	io::path filename, extension, final_file_name_with_extension;
	core::splitFilename(image_file_path.data(), nullptr, &filename, &extension);
	final_file_name_with_extension = filename + ".";
	final_file_name_with_extension += extension;


	auto asset = *cpu_texture_contents.begin();

	switch (asset->getAssetType())
	{
	case IAsset::ET_IMAGE:
	{

		ICPUImageView::SCreationParams viewParams;
		viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
		viewParams.image = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(asset);

		viewParams.format = format;
		viewParams.viewType = static_cast<decltype(viewParams.viewType)>(viewParams.image->getCreationParameters().type);
		viewParams.subresourceRange.baseArrayLayer = 0u;
		viewParams.subresourceRange.layerCount = 1u;
		viewParams.subresourceRange.baseMipLevel = 0u;
		viewParams.subresourceRange.levelCount = 1u;

		copy_image_view = ICPUImageView::create(std::move(viewParams));
	} break;

	case IAsset::ET_IMAGE_VIEW:
	{
		copy_image_view = core::smart_refctd_ptr_static_cast<asset::ICPUImageView>(asset);
	} break;

	default:
	{
		os::Printer::log("EXPECTED IMAGE ASSET TYPE!", ELL_ERROR);
		break;
	}
	}

	core::smart_refctd_ptr<video::IGPUImageView> gpu_image_view;

	gpu_image_view = m_driver->getGPUObjectsFromAssets(&copy_image_view.get(), &copy_image_view.get() + 1u)->front();

	assert(gpu_image_view);

	return gpu_image_view;
}

void WaveSimApp::PresentWaves2D(const textureView& tex)
{
	auto sampler_descriptor_set = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr(m_gpu_descriptor_set_layout_2d));

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
		m_driver->drawMeshBuffer(m_2d_mesh_buffer.get());
	}
}

void WaveSimApp::PresentSkybox(const textureView& envmap, matrix4SIMD mvp)
{
	IGPUDescriptorSet::SDescriptorInfo info[1];
	{
		ISampler::SParams samplerParams_linear = { ISampler::ETC_REPEAT, ISampler::ETC_REPEAT, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
		info[0].desc = envmap;
		info[0].image.sampler = m_driver->createGPUSampler(samplerParams_linear);
		info[0].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
	}
	{
		IGPUDescriptorSet::SWriteDescriptorSet write[1];
		write[0].dstSet = m_skybox_presenting_descriptor_set.get();
		write[0].binding = 0u;
		write[0].arrayElement = 0u;
		write[0].count = 1u;
		write[0].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
		write[0].info = info;

		m_driver->updateDescriptorSets(1u, write, 0u, nullptr);
	}
	{
		m_driver->bindGraphicsPipeline(m_gpu_sphere->getPipeline());
		m_driver->bindDescriptorSets(EPBP_GRAPHICS, m_gpu_sphere->getPipeline()->getLayout(), 0, 1, &m_skybox_presenting_descriptor_set.get(), nullptr);
		m_driver->pushConstants(m_gpu_sphere->getPipeline()->getLayout(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), mvp.pointer());
		m_driver->drawMeshBuffer(m_gpu_sphere.get());
	}
}

void WaveSimApp::PresentWaves3D(const textureView& displacement_map, const textureView& normal_map, const textureView& env_map, const core::matrix4SIMD& mvp, const nbl::core::vector3df& camera)
{
	IGPUDescriptorSet::SDescriptorInfo info[3];
	{
		ISampler::SParams samplerParams_nearest = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
		ISampler::SParams samplerParams_linear = { ISampler::ETC_REPEAT, ISampler::ETC_REPEAT, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
		info[0].desc = displacement_map;
		info[0].image.sampler = m_driver->createGPUSampler(samplerParams_nearest);
		info[0].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
		info[1].desc = normal_map;
		info[1].image.sampler = m_driver->createGPUSampler(samplerParams_linear);
		info[1].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
		info[2].desc = env_map;
		info[2].image.sampler = m_driver->createGPUSampler(samplerParams_linear);
		info[2].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
	}
	{
		IGPUDescriptorSet::SWriteDescriptorSet write[3];
		write[0].dstSet = m_3d_presenting_descriptor_set.get();
		write[0].binding = 0u;
		write[0].arrayElement = 0u;
		write[0].count = 1u;
		write[0].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
		write[0].info = info;
		write[1] = write[0];
		write[1].binding = 1u;
		write[1].info = info + 1;
		write[2] = write[0];
		write[2].binding = 2u;
		write[2].info = info + 2;

		m_driver->updateDescriptorSets(3u, write, 0u, nullptr);
	}

	m_driver->bindDescriptorSets(EPBP_GRAPHICS, m_3d_mesh_buffer->getPipeline()->getLayout(), 0, 1, &m_3d_presenting_descriptor_set.get(), nullptr);
	m_driver->bindGraphicsPipeline(m_3d_mesh_buffer->getPipeline());
	m_driver->pushConstants(m_3d_mesh_buffer->getPipeline()->getLayout(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), mvp.pointer());
	m_driver->pushConstants(m_3d_mesh_buffer->getPipeline()->getLayout(), asset::ISpecializedShader::ESS_FRAGMENT, sizeof(core::matrix4SIMD), sizeof(core::vector3df), &camera);
	m_driver->drawMeshBuffer(m_3d_mesh_buffer.get());
}

smart_refctd_ptr<nbl::video::IGPUBuffer> WaveSimApp::GenerateWaveSpectrum()
{
	const uint32_t SSBO_SIZE = m_params.width * m_params.length * 6 * sizeof(float);

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
		m_driver->dispatch((m_params.width + 15u) / 16u, (m_params.length + 15u) / 16u, 1u);
	}
	return initial_buffer;
}

smart_refctd_ptr<IGPUSpecializedShader> WaveSimApp::createGPUSpecializedShaderFromFile(const std::string_view filepath, asset::ISpecializedShader::E_SHADER_STAGE stage)
{
	auto file = m_filesystem->createAndOpenFile(filepath.data());
	auto spirv = m_asset_manager->getGLSLCompiler()->createSPIRVFromGLSL(file, stage, "main", "runtimeID");
	auto unspec = m_driver->createGPUShader(std::move(spirv));
	return m_driver->createGPUSpecializedShader(unspec.get(), { nullptr, nullptr, "main", stage });
}

smart_refctd_ptr<IGPUSpecializedShader> WaveSimApp::createGPUSpecializedShaderFromFileWithIncludes(const std::string_view filepath, asset::ISpecializedShader::E_SHADER_STAGE stage, std::string_view orig_file_path)
{
	std::ifstream ifs(filepath.data());
	std::string source((std::istreambuf_iterator<char>(ifs)),
		std::istreambuf_iterator<char>());
	auto resolved_includes = m_device->getAssetManager()->getGLSLCompiler()->resolveIncludeDirectives(source.c_str(), stage, orig_file_path.data());
	auto spirv = m_asset_manager->getGLSLCompiler()->createSPIRVFromGLSL(reinterpret_cast<const char*>(resolved_includes->getSPVorGLSL()->getPointer()), stage, "main", "runtimeID");
	auto unspec = m_driver->createGPUShader(std::move(spirv));
	return m_driver->createGPUSpecializedShader(unspec.get(), { nullptr, nullptr, "main", stage });
}

void WaveSimApp::GenerateDisplacementMap(const smart_refctd_ptr<nbl::video::IGPUBuffer>& h0, textureView& out, float time)
{
	using namespace ext::FFT;
	const uint32_t H0_SSBO_SIZE = m_params.width * m_params.length * 4 * sizeof(float);
	const uint32_t SSBO_SIZE = m_params.width * m_params.length * 6 * sizeof(float);
	auto ifft_x_buffer = m_driver->createDeviceLocalGPUBufferOnDedMem(SSBO_SIZE);

	FFT::Parameters_t params[2];
	VkExtent3D dim = { m_params.width, m_params.length, 1 };
	FFT::DispatchInfo_t dispatch_info[2];
	ISampler::E_TEXTURE_CLAMP padding_type[2] = { ISampler::ETC_CLAMP_TO_BORDER, ISampler::ETC_CLAMP_TO_BORDER };
	uint32_t passes = FFT::buildParameters(true, 3u, dim, params, dispatch_info, padding_type);
	assert(passes == 2u);
	{
		video::IGPUDescriptorSet::SWriteDescriptorSet write[2];
		video::IGPUDescriptorSet::SDescriptorInfo info[2];
		write[0].dstSet = m_ifft_1_descriptor_set.get();
		write[0].binding = 0u;
		write[0].count = 1;
		write[0].arrayElement = 0u;
		write[0].descriptorType = asset::EDT_STORAGE_BUFFER;
		write[0].info = info;
		info[0].desc = h0;
		info[0].buffer = { 0, H0_SSBO_SIZE };

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
			vector2df length_unit;
			float time;
		} pc;
		pc.time = time;
		pc.size = m_params.size;
		pc.length_unit = m_params.length_unit;
		pc.params = params[0];

		auto ds = m_ifft_1_descriptor_set.get();
		m_driver->bindDescriptorSets(video::EPBP_COMPUTE, m_ifft_pipeline_1->getLayout(), 0u, 1u, &ds, nullptr);
		m_driver->bindComputePipeline(m_ifft_pipeline_1.get());
		m_driver->pushConstants(m_ifft_pipeline_1->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(pc), &pc);
		{
			m_driver->dispatch(dispatch_info[0].workGroupCount[0], dispatch_info[0].workGroupCount[1], dispatch_info[0].workGroupCount[2]);
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
		auto ds = m_ifft_2_descriptor_set.get();
		m_driver->bindDescriptorSets(video::EPBP_COMPUTE, m_ifft_pipeline_2->getLayout(), 0u, 1u, &ds, nullptr);
		m_driver->bindComputePipeline(m_ifft_pipeline_2.get());

		auto all_params = [&]() {
			struct PC { FFT::Parameters_t params; float choppiness; };
			return PC{ params[1], m_params.choppiness };
		} ();

		m_driver->pushConstants(m_ifft_pipeline_2->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(all_params), &all_params);
		m_driver->dispatch(dispatch_info[1].workGroupCount[0], dispatch_info[1].workGroupCount[1], dispatch_info[1].workGroupCount[2]);
		COpenGLExtensionHandler::pGlMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	}
}

void WaveSimApp::GenerateNormalMap(const textureView& displacement_map, textureView& normalmap)
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
		info[0].desc = displacement_map;
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
		m_driver->dispatch((m_params.width + 15u) / 16u, (m_params.length + 15u) / 16u, 1u);
		COpenGLExtensionHandler::pGlMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	}

}


WaveSimApp::WaveSimApp(const WaveSimParams& params) : m_params{ params }
{
	assert(isPoT(params.width) && isPoT(params.length));
	// For some reason cannot call nbl::core::normalize
	auto normalize = [](vector2df vec)
	{
		float length = sqrt(vec.X * vec.X + vec.Y * vec.Y);
		return vec / length;
	};
	m_params.wind_dir = normalize(m_params.wind_dir);

	bool initialized = Init();
	if constexpr (CURRENT_PRESENTING_MODE == PresentingMode::PM_2D)
	{
		initialized &= CreatePresenting2DPipeline();
	}
	else if constexpr (CURRENT_PRESENTING_MODE == PresentingMode::PM_3D)
	{
		initialized &= CreateSkyboxPresentingPipeline();
		initialized &= CreatePresenting3DPipeline();
	}
	initialized &= CreateComputePipelines();
	assert(initialized);
}

void WaveSimApp::Run()
{

	scene::ICameraSceneNode* camera = m_device->getSceneManager()->addCameraSceneNodeFPS(0, 100.0f, 0.001f);

	camera->setPosition(core::vector3df(50, 5, 50));
	camera->setTarget(core::vector3df(0, 0, 0));
	camera->setNearValue(0.01f);
	camera->setFarValue(10000.0f);

	//translating the skybox a bit lower because the envmap has too much earth visible
	core::matrix3x4SIMD modelMatrix;
	modelMatrix.setTranslation(nbl::core::vectorSIMDf(0, -50, 0, 0));

	m_device->getSceneManager()->setActiveCamera(camera);
	auto initial_values = GenerateWaveSpectrum();
	auto displacement_map = CreateTexture(m_params.size, EF_R16G16B16A16_SFLOAT);
	auto normal_map = CreateTexture(m_params.size, EF_R8G8B8A8_UNORM);
	auto environment_map = CreateTextureFromImageFile(m_envmap_file_path, EF_R32G32B32A32_SFLOAT);
	auto start_time = std::chrono::system_clock::now();
	while (m_device->run() && m_receiver.keepOpen())
	{
		std::chrono::duration<double> time_passed = std::chrono::system_clock::now() - start_time;
		m_driver->beginScene(true, true, SColor(255, 25, 170, 237));
		GenerateDisplacementMap(initial_values, displacement_map, std::chrono::duration_cast<std::chrono::milliseconds>(time_passed).count() / 1000.f);
		GenerateNormalMap(displacement_map, normal_map);

		camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(m_device->getTimer()->getTime()).count());
		camera->render();
		core::matrix4SIMD mvp = camera->getConcatenatedMatrix();

		if constexpr (CURRENT_PRESENTING_MODE == PresentingMode::PM_2D)
		{
			PresentWaves2D(displacement_map);
		}
		else if constexpr (CURRENT_PRESENTING_MODE == PresentingMode::PM_3D)
		{
			PresentWaves3D(displacement_map, normal_map, environment_map, mvp, camera->getAbsolutePosition());

			core::matrix4SIMD skybox_mvp = core::concatenateBFollowedByA(mvp, modelMatrix);
			PresentSkybox(environment_map, skybox_mvp);
		}
		m_driver->endScene();

	}
}
