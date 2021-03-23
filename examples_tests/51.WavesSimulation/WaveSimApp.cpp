#include <chrono> 

#include "WaveSimApp.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/ext/FFT/FFT.h"
#include "../common/QToQuitEventReceiver.h"

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
	const size_t INDEX_COUNT = m_params.length * m_params.width * 2 * 3;
	const size_t VERTEX_COUNT = m_params.length * m_params.width;

	std::vector<uint32_t> indices(INDEX_COUNT);
	std::vector<VertexData> vertices(VERTEX_COUNT);
	for (int y = 0; y < m_params.length - 1; y++)
	{
		const size_t y_pos = y * m_params.width;
		const size_t y_plus_one_pos = (y + 1) * m_params.width;
		for (int x = 0; x < m_params.width - 1; x++)
		{
			const size_t current_pos = (y_pos + x) * INDICES_PER_QUAD;
			indices[current_pos] = y_pos + x;
			indices[current_pos + 1] = y_plus_one_pos + x;
			indices[current_pos + 2] = y_plus_one_pos + x + 1;
			indices[current_pos + 3] = y_pos + x;
			indices[current_pos + 4] = y_plus_one_pos + x + 1;
			indices[current_pos + 5] = y_pos + x + 1;
		}

	}
	for (int z = 0; z < m_params.length; z++)
	{
		float z_world = z - m_params.length / 2.f;
		size_t z_pos = z * m_params.width;
		for (int x = 0; x < m_params.width; x++)
		{
			float x_world = x - m_params.width / 2.f;
			vertices[z_pos + x] = { vector3df{x_world / m_params.width * 2, 0.f, z_world / m_params.length * 2}, vector2df{float(x) / m_params.width, float(z) / m_params.length } };
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
		IGPUDescriptorSetLayout::SBinding texture_bindings[2];
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

		presenting_layout = m_driver->createGPUDescriptorSetLayout(texture_bindings, texture_bindings + 2);
		m_presenting_3d_descriptor_set = m_driver->createGPUDescriptorSet(presenting_layout);
	}

	const char* vertex_shader_path = "../waves_display_3d.vert";
	const char* fragment_shader_path = "../waves_display_3d.frag";


	asset::SPushConstantRange ranges[2] = { { asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD) },
										    { asset::ISpecializedShader::ESS_FRAGMENT, sizeof(core::matrix4SIMD), sizeof(core::vector3df) } };

	auto createGPUSpecializedShaderFromFile = [this](const char* filepath, asset::ISpecializedShader::E_SHADER_STAGE stage)
	{
		auto file = m_filesystem->createAndOpenFile(filepath);
		auto spirv = m_asset_manager->getGLSLCompiler()->createSPIRVFromGLSL(file, stage, "main", "runtimeID");
		auto unspec = m_driver->createGPUShader(std::move(spirv));
		return m_driver->createGPUSpecializedShader(unspec.get(), { nullptr, nullptr, "main", stage });
	};


	core::smart_refctd_ptr<video::IGPUSpecializedShader> shaders[2] =
	{
 		createGPUSpecializedShaderFromFile(vertex_shader_path, asset::ISpecializedShader::ESS_VERTEX),
		createGPUSpecializedShaderFromFile(fragment_shader_path, asset::ISpecializedShader::ESS_FRAGMENT)
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
			uint32_t maxPaddedDimensionSize = std::max(m_params.length, m_params.width);
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
			uint32_t maxPaddedDimensionSize = std::max(m_params.length, m_params.width);
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

void WaveSimApp::PresentWaves3D(const textureView& displacement_map, const textureView& normal_map, const core::matrix4SIMD& mvp, const nbl::core::vector3df& camera)
{
	IGPUDescriptorSet::SDescriptorInfo info[2];
	{
		ISampler::SParams samplerParams_nearest = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
		ISampler::SParams samplerParams_linear = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
		info[0].desc = displacement_map;
		info[0].image.sampler = m_driver->createGPUSampler(samplerParams_nearest);
		info[0].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
		info[1].desc = normal_map;
		info[1].image.sampler = m_driver->createGPUSampler(samplerParams_linear);
		info[1].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
	}
	{
		IGPUDescriptorSet::SWriteDescriptorSet write[2];
		write[0].dstSet = m_presenting_3d_descriptor_set.get();
		write[0].binding = 0u;
		write[0].arrayElement = 0u;
		write[0].count = 1u;
		write[0].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
		write[0].info = info;
		write[1] = write[0];
		write[1].binding = 1u;
		write[1].info = info + 1;

		m_driver->updateDescriptorSets(2u, write, 0u, nullptr);
	}

	m_driver->bindDescriptorSets(EPBP_GRAPHICS, m_3d_mesh_buffer->getPipeline()->getLayout(), 0, 2, &m_presenting_3d_descriptor_set.get(), nullptr);
	m_driver->bindGraphicsPipeline(m_3d_mesh_buffer->getPipeline());
	m_driver->pushConstants(m_3d_mesh_buffer->getPipeline()->getLayout(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), mvp.pointer());
	m_driver->pushConstants(m_3d_mesh_buffer->getPipeline()->getLayout(), asset::ISpecializedShader::ESS_FRAGMENT, sizeof(core::matrix4SIMD), sizeof(core::vector3df), &camera);
	m_driver->drawMeshBuffer(m_3d_mesh_buffer.get());
}

smart_refctd_ptr<nbl::video::IGPUBuffer> WaveSimApp::RandomizeWaveSpectrum()
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

void WaveSimApp::AnimateSpectrum(const nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer>& h0, nbl::core::smart_refctd_ptr<nbl::video::IGPUBuffer>& animated_spectrum, float time)
{
	const uint32_t IN_SSBO_SIZE = m_params.width * m_params.length * 4 * sizeof(float);
	const uint32_t OUT_SSBO_SIZE = m_params.width * m_params.length * 6 * sizeof(float);
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
		struct PC { dimension2du size; vector2df length_unit; float time; };
		return PC{ m_params.size, m_params.length_unit, time };
	} ();
	auto ds = m_spectrum_animating_descriptor_set.get();
	m_driver->bindDescriptorSets(video::EPBP_COMPUTE, m_spectrum_animating_pipeline->getLayout(), 0u, 1u, &ds, nullptr);
	m_driver->bindComputePipeline(m_spectrum_animating_pipeline.get());
	m_driver->pushConstants(m_spectrum_animating_pipeline->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(pc), &pc);
	{
		m_driver->dispatch((m_params.width + 15u) / 16u, (m_params.length + 15u) / 16u, 1u);
		COpenGLExtensionHandler::pGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	}
}

void WaveSimApp::GenerateDisplacementMap(const smart_refctd_ptr<nbl::video::IGPUBuffer>& h0, textureView& out, float time)
{
	using namespace ext::FFT;
	const uint32_t SSBO_SIZE = m_params.width * m_params.length * 6 * sizeof(float);
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
		pc.params.dimension.y = m_params.length;
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
		params.dimension.y = m_params.length;
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
		initialized &= CreatePresenting3DPipeline();
	}
	initialized &= CreateComputePipelines();
	assert(initialized);
}

void WaveSimApp::Run()
{

	scene::ICameraSceneNode* camera = m_device->getSceneManager()->addCameraSceneNodeFPS(0, 100.0f, 0.001f);

	camera->setPosition(core::vector3df(-4, 1, 0));
	camera->setTarget(core::vector3df(0, 0, 0));
	camera->setNearValue(0.01f);
	camera->setFarValue(10.0f);

	m_device->getSceneManager()->setActiveCamera(camera);

	auto initial_values = RandomizeWaveSpectrum();
	auto displacement_map = CreateTexture(m_params.size, EF_R8G8B8A8_UNORM);
	auto normal_map = CreateTexture(m_params.size, EF_R8G8B8A8_UNORM);
	auto start_time = std::chrono::system_clock::now();
	while (m_device->run())
	{
		std::chrono::duration<double> time_passed = std::chrono::system_clock::now() - start_time;
		m_driver->beginScene(true);
		GenerateDisplacementMap(initial_values, displacement_map, std::chrono::duration_cast<std::chrono::milliseconds>(time_passed).count() / 1000.f);
		GenerateNormalMap(displacement_map, normal_map);

		camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(m_device->getTimer()->getTime()).count());
		camera->render();
		core::matrix4SIMD mvp = camera->getConcatenatedMatrix();
		
		if constexpr (CURRENT_PRESENTING_MODE == PresentingMode::PM_2D)
		{
			PresentWaves2D(normal_map);
		}
		else if constexpr(CURRENT_PRESENTING_MODE == PresentingMode::PM_3D)
		{
			PresentWaves3D(displacement_map, normal_map, mvp, camera->getAbsolutePosition());
		}
		m_driver->endScene();

	}
}
