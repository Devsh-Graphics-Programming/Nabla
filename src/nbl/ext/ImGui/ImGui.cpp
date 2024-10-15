#include <iostream>
#include <map>
#include <ranges>
#include <vector>
#include <utility>
#include <locale>
#include <codecvt>

#include "nbl/system/CStdoutLogger.h"
#include "nbl/ext/ImGui/ImGui.h"
#include "nbl/ext/ImGui/builtin/hlsl/common.hlsl"
#include "nbl/ext/ImGui/builtin/builtinResources.h"
#include "nbl/ext/ImGui/builtin/CArchive.h"
#include "imgui/imgui.h"
#include "imgui/misc/cpp/imgui_stdlib.h"

using namespace nbl::video;
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::system;
using namespace nbl::ui;
using namespace nbl::hlsl;

namespace nbl::ext::imgui
{
static constexpr SPushConstantRange PushConstantRanges[] =
{
	{
		.stageFlags = IShader::E_SHADER_STAGE::ESS_VERTEX | IShader::E_SHADER_STAGE::ESS_FRAGMENT,
		.offset = 0,
		.size = sizeof(PushConstants)
	}
};

smart_refctd_ptr<IGPUPipelineLayout> UI::createDefaultPipelineLayout(ILogicalDevice* const device, const SResourceParameters::SBindingInfo texturesInfo, const SResourceParameters::SBindingInfo samplersInfo, uint32_t texturesCount)
{
	if (!device)
		return nullptr;

	if (texturesInfo.bindingIx == samplersInfo.bindingIx)
		return nullptr;

	if (!texturesCount)
		return nullptr;

	smart_refctd_ptr<IGPUSampler> fontAtlasUISampler, userTexturesSampler;

	using binding_flags_t = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS;
	{
		IGPUSampler::SParams params;
		params.AnisotropicFilter = 1u;
		params.TextureWrapU = ISampler::ETC_REPEAT;
		params.TextureWrapV = ISampler::ETC_REPEAT;
		params.TextureWrapW = ISampler::ETC_REPEAT;

		fontAtlasUISampler = device->createSampler(params);
		fontAtlasUISampler->setObjectDebugName("Nabla default ImGUI font UI sampler");
	}

	{
		IGPUSampler::SParams params;
		params.MinLod = 0.f;
		params.MaxLod = 0.f;
		params.TextureWrapU = ISampler::ETC_CLAMP_TO_EDGE;
		params.TextureWrapV = ISampler::ETC_CLAMP_TO_EDGE;
		params.TextureWrapW = ISampler::ETC_CLAMP_TO_EDGE;

		userTexturesSampler = device->createSampler(params);
		userTexturesSampler->setObjectDebugName("Nabla default ImGUI user texture sampler");
	}

	//! note we use immutable separate samplers and they are part of the descriptor set layout
	std::array<smart_refctd_ptr<IGPUSampler>, (uint32_t)DefaultSamplerIx::COUNT> immutableSamplers;
	immutableSamplers[(uint32_t)DefaultSamplerIx::FONT_ATLAS] = smart_refctd_ptr(fontAtlasUISampler);
	immutableSamplers[(uint32_t)DefaultSamplerIx::USER] = smart_refctd_ptr(userTexturesSampler);

	auto textureBinding = IGPUDescriptorSetLayout::SBinding
	{
		.binding = texturesInfo.bindingIx,
		.type = IDescriptor::E_TYPE::ET_SAMPLED_IMAGE,
		.createFlags = SResourceParameters::TexturesRequiredCreateFlags,
		.stageFlags = SResourceParameters::RequiredShaderStageFlags,
		.count = texturesCount
	};

	auto samplersBinding = IGPUDescriptorSetLayout::SBinding
	{
		.binding = samplersInfo.bindingIx,
		.type = IDescriptor::E_TYPE::ET_SAMPLER,
		.createFlags = SResourceParameters::SamplersRequiredCreateFlags,
		.stageFlags = SResourceParameters::RequiredShaderStageFlags,
		.count = immutableSamplers.size(),
		.immutableSamplers = immutableSamplers.data()
	};

	auto layouts = std::to_array<smart_refctd_ptr<IGPUDescriptorSetLayout>>({ nullptr, nullptr, nullptr, nullptr });

	if (texturesInfo.setIx == samplersInfo.setIx)
		layouts[texturesInfo.setIx] = device->createDescriptorSetLayout({ {textureBinding, samplersBinding} });
	else
	{
		layouts[texturesInfo.setIx] = device->createDescriptorSetLayout({ {textureBinding} });
		layouts[samplersInfo.setIx] = device->createDescriptorSetLayout({ {samplersBinding} });
	}

	if (!layouts[texturesInfo.setIx])
		return nullptr;

	if (!layouts[samplersInfo.setIx])
		return nullptr;

	return device->createPipelineLayout(PushConstantRanges, std::move(layouts[0u]), std::move(layouts[1u]), std::move(layouts[2u]), std::move(layouts[3u]));
}

// note we use archive entry explicitly for temporary compiler include search path & asset cwd to use keys directly
constexpr std::string_view NBL_ARCHIVE_ENTRY = nbl::ext::imgui::builtin::pathPrefix;

const smart_refctd_ptr<IFileArchive> UI::mount(smart_refctd_ptr<ILogger> logger, ISystem* system, const std::string_view archiveAlias)
{
	assert(system);
		
	if(!system)
		return nullptr;

	// extension should mount everything for you, regardless if content goes from virtual filesystem 
	// or disk directly - and you should never rely on application framework to expose extension data

	#ifdef NBL_EMBED_BUILTIN_RESOURCES
	auto archive = make_smart_refctd_ptr<builtin::CArchive>(smart_refctd_ptr(logger));
	system->mount(smart_refctd_ptr(archive), archiveAlias.data());
	#else
	auto NBL_EXTENSION_MOUNT_DIRECTORY_ENTRY = (path(_ARCHIVE_ABSOLUTE_ENTRY_PATH_) / NBL_ARCHIVE_ENTRY).make_preferred();
	auto archive = make_smart_refctd_ptr<nbl::system::CMountDirectoryArchive>(std::move(NBL_EXTENSION_MOUNT_DIRECTORY_ENTRY), smart_refctd_ptr(logger), system);
	system->mount(smart_refctd_ptr(archive), archiveAlias.data());
	#endif

	return smart_refctd_ptr(archive);
}

core::smart_refctd_ptr<video::IGPUGraphicsPipeline> UI::createPipeline(SCreationParameters& creationParams)
{
	auto pipelineLayout = smart_refctd_ptr<IGPUPipelineLayout>(creationParams.pipelineLayout);

	if (!pipelineLayout)
	{
		creationParams.utilities->getLogger()->log("Could not create pipeline layout!", ILogger::ELL_ERROR);
		return nullptr;
	}

	struct
	{
		smart_refctd_ptr<IGPUShader> vertex, fragment;
	} shaders;

	{		
		//! proxy the system, we will touch it gently
		auto system = smart_refctd_ptr<ISystem>(creationParams.assetManager->getSystem());

		auto* set = creationParams.assetManager->getCompilerSet();
		auto compiler = set->getShaderCompiler(IShader::E_CONTENT_TYPE::ECT_HLSL);
		auto includeFinder = make_smart_refctd_ptr<IShaderCompiler::CIncludeFinder>(smart_refctd_ptr(system));
		auto includeLoader = includeFinder->getDefaultFileSystemLoader();
		includeFinder->addSearchPath(NBL_ARCHIVE_ENTRY.data(), includeLoader);

		auto createShader = [&]<StringLiteral key, IShader::E_SHADER_STAGE stage>() -> smart_refctd_ptr<IGPUShader>
		{
			IAssetLoader::SAssetLoadParams params = {};
			params.logger = creationParams.utilities->getLogger();
			params.workingDirectory = NBL_ARCHIVE_ENTRY.data();

			auto bundle = creationParams.assetManager->getAsset(key.value, params);
			const auto assets = bundle.getContents();

			if (assets.empty())
			{
				creationParams.utilities->getLogger()->log("Could not load \"%s\" shader!", ILogger::ELL_ERROR, key.value);
				return nullptr;
			}

			const auto shader = IAsset::castDown<ICPUShader>(assets[0]);

			CHLSLCompiler::SOptions options = {};
			options.stage = stage;
			options.preprocessorOptions.sourceIdentifier = key.value;
			options.preprocessorOptions.logger = creationParams.utilities->getLogger();
			options.preprocessorOptions.includeFinder = includeFinder.get();

			auto compileToSPIRV = [&]() -> smart_refctd_ptr<ICPUShader>
			{
				auto toOptions = []<uint32_t N>(const std::array<std::string_view, N>& in) // options must be alive till compileToSPIRV ends
				{
					const auto required = CHLSLCompiler::getRequiredArguments();
					std::array<std::string, required.size() + N> options;

					std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
					for (uint32_t i = 0; i < required.size(); ++i)
						options[i] = converter.to_bytes(required[i]); // meh

					uint32_t offset = required.size();
					for (const auto& opt : in)
						options[offset++] = std::string(opt);

					return options;
				};

				const std::string_view code (reinterpret_cast<const char*>(shader->getContent()->getPointer()), shader->getContent()->getSize());

				if constexpr (stage == IShader::E_SHADER_STAGE::ESS_VERTEX)
				{
					const auto VERTEX_COMPILE_OPTIONS = toOptions(std::to_array<std::string_view>({ "-T", "vs_6_7", "-E", "VSMain", "-O3" }));
					options.dxcOptions = VERTEX_COMPILE_OPTIONS;

					return compiler->compileToSPIRV(code.data(), options); // we good here - no code patching
				}
				else if (stage == IShader::E_SHADER_STAGE::ESS_FRAGMENT)
				{
					const auto FRAGMENT_COMPILE_OPTIONS = toOptions(std::to_array<std::string_view>({ "-T", "ps_6_7", "-E", "PSMain", "-O3" }));
					options.dxcOptions = FRAGMENT_COMPILE_OPTIONS;

					std::stringstream stream;

					stream << "// -> this code has been autogenerated with Nabla ImGUI extension\n"
						<< "#define NBL_TEXTURES_BINDING_IX " << creationParams.resources.texturesInfo.bindingIx << "\n"
						<< "#define NBL_SAMPLER_STATES_BINDING_IX " << creationParams.resources.samplersInfo.bindingIx << "\n"
						<< "#define NBL_TEXTURES_SET_IX " << creationParams.resources.texturesInfo.setIx << "\n"
						<< "#define NBL_SAMPLER_STATES_SET_IX " << creationParams.resources.samplersInfo.setIx << "\n"
						<< "#define NBL_TEXTURES_COUNT " << creationParams.resources.texturesCount << "\n"
						<< "#define NBL_SAMPLERS_COUNT " << creationParams.resources.samplersCount << "\n"
						<< "// <-\n\n";

					const auto newCode = stream.str() + std::string(code);
					return compiler->compileToSPIRV(newCode.c_str(), options); // but here we do patch the code with additional define directives for which values are taken from the creation parameters
				}
				else
				{
					static_assert(stage != IShader::E_SHADER_STAGE::ESS_UNKNOWN, "Unknown shader stage!");
					return nullptr;
				}
			};

			auto spirv = compileToSPIRV();

			if (!spirv)
			{
				creationParams.utilities->getLogger()->log("Could not compile \"%s\" shader!", ILogger::ELL_ERROR, key.value);
				return nullptr;
			}

			auto gpu = creationParams.utilities->getLogicalDevice()->createShader(spirv.get());

			if (!gpu)
				creationParams.utilities->getLogger()->log("Could not create GPU shader for \"%s\"!", ILogger::ELL_ERROR, key.value);

			return gpu;
		};

		if (!system->areBuiltinsMounted())
		{
			creationParams.utilities->getLogger()->log("Nabla builtins are not mounted!", ILogger::ELL_ERROR);
			return nullptr;
		}

		//! but we should never assume user will mount our internal data since its the extension and not user's job to do it so we do to compile our extension sources
		if(!system->isDirectory(path(NBL_ARCHIVE_ENTRY.data())))
			mount(smart_refctd_ptr<ILogger>(creationParams.utilities->getLogger()), system.get(), NBL_ARCHIVE_ENTRY);

		shaders.vertex = createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("vertex.hlsl"), IShader::E_SHADER_STAGE::ESS_VERTEX > ();
		shaders.fragment = createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("fragment.hlsl"), IShader::E_SHADER_STAGE::ESS_FRAGMENT > ();

		if (!shaders.vertex)
		{
			creationParams.utilities->getLogger()->log("Failed to compile vertex shader!", ILogger::ELL_ERROR);
			return nullptr;
		}

		if (!shaders.fragment)
		{
			creationParams.utilities->getLogger()->log("Failed to compile fragment shader!", ILogger::ELL_ERROR);
			return nullptr;
		}
	}
	
	SVertexInputParams vertexInputParams{};
	{
		vertexInputParams.enabledBindingFlags = 0b1u;
		vertexInputParams.enabledAttribFlags = 0b111u;

		vertexInputParams.bindings[0].inputRate = SVertexInputBindingParams::EVIR_PER_VERTEX;
		vertexInputParams.bindings[0].stride = sizeof(ImDrawVert);

		auto& position = vertexInputParams.attributes[0];
		position.format = EF_R32G32_SFLOAT;
		position.relativeOffset = offsetof(ImDrawVert, pos);
		position.binding = 0u;

		auto& uv = vertexInputParams.attributes[1];
		uv.format = EF_R32G32_SFLOAT;
		uv.relativeOffset = offsetof(ImDrawVert, uv);
		uv.binding = 0u;

		auto& color = vertexInputParams.attributes[2];
		color.format = EF_R8G8B8A8_UNORM;
		color.relativeOffset = offsetof(ImDrawVert, col);
		color.binding = 0u;
	}

	SBlendParams blendParams{};
	{
		blendParams.logicOp = ELO_NO_OP;

		auto& param = blendParams.blendParams[0];

		// color blending factors (for RGB)
		param.srcColorFactor = EBF_SRC_ALPHA;
		param.dstColorFactor = EBF_ONE_MINUS_SRC_ALPHA;
		param.colorBlendOp = EBO_ADD;

		// alpha blending factors (for A)
		param.srcAlphaFactor = EBF_ONE;
		param.dstAlphaFactor = EBF_ONE_MINUS_SRC_ALPHA;
		param.alphaBlendOp = EBO_ADD;

		// Write all components (R, G, B, A)
		param.colorWriteMask = (1u << 0u) | (1u << 1u) | (1u << 2u) | (1u << 3u);
	}
	
	SRasterizationParams rasterizationParams{};
	{
		rasterizationParams.faceCullingMode = EFCM_NONE;
		rasterizationParams.depthWriteEnable = false;
		rasterizationParams.depthBoundsTestEnable = false;
		rasterizationParams.viewportCount = creationParams.viewportCount;
	}

	SPrimitiveAssemblyParams primitiveAssemblyParams{};
	{
		primitiveAssemblyParams.primitiveType = EPT_TRIANGLE_LIST;
	}

	core::smart_refctd_ptr<video::IGPUGraphicsPipeline> pipeline;
	{
		const IGPUShader::SSpecInfo specs[] =
		{
			{ .entryPoint = "VSMain", .shader = shaders.vertex.get() },
			{ .entryPoint = "PSMain", .shader = shaders.fragment.get() }
		};

		IGPUGraphicsPipeline::SCreationParams params[1];
		{
			auto& param = params[0u];
			param.layout = pipelineLayout.get();
			param.shaders = specs;
			param.renderpass = creationParams.renderpass.get();
			param.cached = { .vertexInput = vertexInputParams, .primitiveAssembly = primitiveAssemblyParams, .rasterization = rasterizationParams, .blend = blendParams, .subpassIx = creationParams.subpassIx };
		};
		
		if (!creationParams.utilities->getLogicalDevice()->createGraphicsPipelines(creationParams.pipelineCache.get(), params, &pipeline))
		{
			creationParams.utilities->getLogger()->log("Could not create pipeline!", ILogger::ELL_ERROR);
			return nullptr;
		}
	}

	return pipeline;
}

smart_refctd_ptr<IGPUImageView> UI::createFontAtlasTexture(const SCreationParameters& creationParams, void* const imFontAtlas)
{
	video::IQueue* queue = creationParams.transfer;
	system::logger_opt_ptr logger = creationParams.utilities->getLogger();
	auto* device = creationParams.utilities->getLogicalDevice();
	auto* const fontAtlas = reinterpret_cast<ImFontAtlas*>(imFontAtlas);

	// intialize command buffers
	constexpr auto TransfersInFlight = 2;
	std::array<smart_refctd_ptr<nbl::video::IGPUCommandBuffer>,TransfersInFlight> commandBuffers;
	{
		using pool_flags_t = IGPUCommandPool::CREATE_FLAGS;
		auto pool = device->createCommandPool(queue->getFamilyIndex(), pool_flags_t::RESET_COMMAND_BUFFER_BIT | pool_flags_t::TRANSIENT_BIT);
		if (!pool)
		{
			logger.log("Could not create command pool!", ILogger::ELL_ERROR);
			return nullptr;
		}
		if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { commandBuffers.data(),TransfersInFlight }, core::smart_refctd_ptr<system::ILogger>(logger.get())))
		{
			logger.log("Could not create transistent command buffers!", ILogger::ELL_ERROR);
			return nullptr;
		}
	}

	constexpr auto NBL_FORMAT_FONT = EF_R8G8B8A8_UNORM;
	
	// CCustomAllocatorCPUBuffer is kinda crap and doesn't support stateful allocators
	uint8_t* pixels = nullptr;
	auto freeMemory = core::makeRAIIExiter([&pixels]()->void
		{
			if (pixels)
				IM_FREE(pixels);
		}
	);

	// make buffer with image contents
	core::smart_refctd_ptr<ICPUImage> cpuImage;
	{
		int32_t width, height;
		fontAtlas->GetTexDataAsRGBA32(&pixels, &width, &height);
		if (!pixels)
			return nullptr;
		if (width<=0 || height<=0)
			return nullptr;
		const asset::VkExtent3D extent = {static_cast<uint32_t>(width),static_cast<uint32_t>(height),1u};

		// create image
		using usage_flags_t = ICPUImage::E_USAGE_FLAGS;
		cpuImage = asset::ICPUImage::create({
			.type = IImage::ET_2D,
			.samples = IImage::ESCF_1_BIT,
			.format = NBL_FORMAT_FONT,
			.extent = extent,
			.mipLevels = 1u,
			.arrayLayers = 1u,
			.flags = IImage::E_CREATE_FLAGS::ECF_NONE,
			.usage = usage_flags_t::EUF_SAMPLED_BIT // transfer should get patched in by having regions
		});
		if (!cpuImage)
		{
			logger.log("Could not create font ICPUImage!", ILogger::ELL_ERROR);
			return nullptr;
		}

		// set its contents
		{
			const size_t image_size = getTexelOrBlockBytesize(NBL_FORMAT_FONT)*width*height;
			auto buffer = make_smart_refctd_ptr<CCustomAllocatorCPUBuffer<null_allocator<uint8_t>>>(image_size,pixels,adopt_memory);
			auto regions = make_refctd_dynamic_array<smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1ull);
			{
				auto region = regions->begin();
				region->bufferOffset = 0ull;
				region->bufferRowLength = width;
				region->bufferImageHeight = 0u;
				region->imageSubresource = {
					.aspectMask = IImage::EAF_COLOR_BIT,
					.mipLevel = 0u,
					.baseArrayLayer = 0u,
					.layerCount = 1u
				};
				region->imageOffset = { 0u, 0u, 0u };
				region->imageExtent = extent;
			}
			if (!cpuImage->setBufferAndRegions(std::move(buffer),std::move(regions)))
			{
				logger.log("Could not set font ICPUImage contents!",ILogger::ELL_ERROR);
				return nullptr;
			}
			cpuImage->setContentHash(cpuImage->computeContentHash());
		}

		// note its by default but you can still change it at runtime, both the texture & sampler id
		SImResourceInfo defaultInfo;
		defaultInfo.textureID = FontAtlasTexId;
		defaultInfo.samplerIx = FontAtlasSamplerId;
		fontAtlas->SetTexID(defaultInfo);

	}

	// convert the CPU Image into a GPU image
	core::smart_refctd_ptr<IGPUImage> gpuImage;
	core::smart_refctd_ptr<ISemaphore> imgFillSemaphore = device->createSemaphore(0);
	{
		std::array<IQueue::SSubmitInfo::SCommandBufferInfo,TransfersInFlight> commandBufferInfos;
		for (uint32_t i=0u; i<TransfersInFlight; ++i)
		{
			commandBuffers[i]->setObjectDebugName(("ImGUI Font Upload Command Buffer #"+std::to_string(i)).c_str());
			commandBufferInfos[i].cmdbuf = commandBuffers[i].get();
		}
		// as per the `SIntendedSubmitInfo` one commandbuffer must be begun
		if (!commandBuffers[0]->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
		{
			logger.log("Could not begin command buffer \"%s\"",ILogger::ELL_ERROR,commandBuffers[0]->getObjectDebugName());
			return nullptr;
		}

		imgFillSemaphore->setObjectDebugName("ImGUI Font Image Fill Semaphore");
		auto converter = CAssetConverter::create({.device=device});
		{
			// We don't want to generate mip-maps for these images (YET), to ensure that we must override the default callbacks.
			struct SInputs final : CAssetConverter::SInputs
			{
				inline uint8_t getMipLevelCount(const size_t groupCopyID, const ICPUImage* image, const CAssetConverter::patch_t<asset::ICPUImage>& patch) const override
				{
					return image->getCreationParameters().mipLevels;
				}
				inline uint16_t needToRecomputeMips(const size_t groupCopyID, const ICPUImage* image, const CAssetConverter::patch_t<asset::ICPUImage>& patch) const override
				{
					return 0b0u;
				}
			} inputs = {};
			std::get<CAssetConverter::SInputs::asset_span_t<ICPUImage>>(inputs.assets) = { &cpuImage.get(),1 };
			inputs.logger = logger;

			auto reservation = converter->reserve(inputs);
			gpuImage = reservation.getGPUObjects<ICPUImage>().front().value;

			// now convert
			SIntendedSubmitInfo transfer = {};
			{
				transfer.queue = queue;
				transfer.waitSemaphores = {};
				transfer.scratchCommandBuffers = commandBufferInfos;
				transfer.scratchSemaphore =
				{
					.semaphore = imgFillSemaphore.get(),
					.value = 0u,
					.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
				};
			}
			// TODO: FIXME ImGUI needs to know what Queue will have ownership of the image AFTER its uploaded (need to know the family of the graphics queue)
			// right now, the transfer queue will stay the owner after upload
			CAssetConverter::SConvertParams params = {};
			params.transfer = &transfer;
			params.utilities = creationParams.utilities.get();
			queue->startCapture();
			auto result = reservation.convert(params);
			queue->endCapture();
			// block immediately
			if (result.copy()!=IQueue::RESULT::SUCCESS)
			{
				logger.log("Failed to record or submit conversion of ImGUI Font Image");
				return nullptr;
			}
		}
	}
	
	// create the view
	IGPUImageView::SCreationParams params;
	params.format = gpuImage->getCreationParameters().format;
	params.viewType = IImageView<IGPUImage>::ET_2D;
	params.image = std::move(gpuImage);
    return device->createImageView(std::move(params));
}

void UI::handleMouseEvents(const SUpdateParameters& params) const
{
	auto& io = ImGui::GetIO();

	io.AddMousePosEvent(params.mousePosition.x, params.mousePosition.y);

	for (const auto& e : params.mouseEvents)
	{
		switch (e.type)
		{
		case SMouseEvent::EET_CLICK:
		{
			ImGuiMouseButton_ button = ImGuiMouseButton_COUNT;
			if (e.clickEvent.mouseButton == EMB_LEFT_BUTTON)
				button = ImGuiMouseButton_Left;
			else if (e.clickEvent.mouseButton == EMB_RIGHT_BUTTON)
				button = ImGuiMouseButton_Right;
			else if (e.clickEvent.mouseButton == EMB_MIDDLE_BUTTON)
				button = ImGuiMouseButton_Middle;

			if (button == ImGuiMouseButton_COUNT)
				continue;

			if (e.clickEvent.action == SMouseEvent::SClickEvent::EA_PRESSED)
				io.AddMouseButtonEvent(button, true);
			else if (e.clickEvent.action == SMouseEvent::SClickEvent::EA_RELEASED)
				io.AddMouseButtonEvent(button, false);
		} break;

		case SMouseEvent::EET_SCROLL:
		{
			_NBL_STATIC_INLINE_CONSTEXPR auto scalar = 0.02f;
			const auto wheel = float32_t2(e.scrollEvent.horizontalScroll, e.scrollEvent.verticalScroll) * scalar;

			io.AddMouseWheelEvent(wheel.x, wheel.y);
		} break;

		case SMouseEvent::EET_MOVEMENT:

		default:
			break;
		}
	}
}

struct NBL_TO_IMGUI_KEY_BIND 
{
	ImGuiKey target;
	char physicalSmall;
	char physicalBig;
};

// maps Nabla keys to IMGUIs
constexpr std::array<NBL_TO_IMGUI_KEY_BIND, EKC_COUNT> createKeyMap()
{
	std::array<NBL_TO_IMGUI_KEY_BIND, EKC_COUNT> map = { { NBL_TO_IMGUI_KEY_BIND{ImGuiKey_None, '0', '0'} } };

	#define NBL_REGISTER_KEY(__NBL_KEY__, __IMGUI_KEY__) \
		map[__NBL_KEY__] = NBL_TO_IMGUI_KEY_BIND{__IMGUI_KEY__, keyCodeToChar(__NBL_KEY__, false), keyCodeToChar(__NBL_KEY__, true)};

	NBL_REGISTER_KEY(EKC_BACKSPACE, ImGuiKey_Backspace);
	NBL_REGISTER_KEY(EKC_TAB, ImGuiKey_Tab);
	NBL_REGISTER_KEY(EKC_ENTER, ImGuiKey_Enter);
	NBL_REGISTER_KEY(EKC_LEFT_SHIFT, ImGuiKey_LeftShift);
	NBL_REGISTER_KEY(EKC_RIGHT_SHIFT, ImGuiKey_RightShift);
	NBL_REGISTER_KEY(EKC_LEFT_CONTROL, ImGuiKey_LeftCtrl);
	NBL_REGISTER_KEY(EKC_RIGHT_CONTROL, ImGuiKey_RightCtrl);
	NBL_REGISTER_KEY(EKC_LEFT_ALT, ImGuiKey_LeftAlt);
	NBL_REGISTER_KEY(EKC_RIGHT_ALT, ImGuiKey_RightAlt);
	NBL_REGISTER_KEY(EKC_PAUSE, ImGuiKey_Pause);
	NBL_REGISTER_KEY(EKC_CAPS_LOCK, ImGuiKey_CapsLock);
	NBL_REGISTER_KEY(EKC_ESCAPE, ImGuiKey_Escape);
	NBL_REGISTER_KEY(EKC_SPACE, ImGuiKey_Space);
	NBL_REGISTER_KEY(EKC_PAGE_UP, ImGuiKey_PageUp);
	NBL_REGISTER_KEY(EKC_PAGE_DOWN, ImGuiKey_PageDown);
	NBL_REGISTER_KEY(EKC_END, ImGuiKey_End);
	NBL_REGISTER_KEY(EKC_HOME, ImGuiKey_Home);
	NBL_REGISTER_KEY(EKC_LEFT_ARROW, ImGuiKey_LeftArrow);
	NBL_REGISTER_KEY(EKC_RIGHT_ARROW, ImGuiKey_RightArrow);
	NBL_REGISTER_KEY(EKC_DOWN_ARROW, ImGuiKey_DownArrow);
	NBL_REGISTER_KEY(EKC_UP_ARROW, ImGuiKey_UpArrow);
	NBL_REGISTER_KEY(EKC_PRINT_SCREEN, ImGuiKey_PrintScreen);
	NBL_REGISTER_KEY(EKC_INSERT, ImGuiKey_Insert);
	NBL_REGISTER_KEY(EKC_DELETE, ImGuiKey_Delete);
	NBL_REGISTER_KEY(EKC_APPS, ImGuiKey_Menu);
	NBL_REGISTER_KEY(EKC_COMMA, ImGuiKey_Comma);
	NBL_REGISTER_KEY(EKC_PERIOD, ImGuiKey_Period);
	NBL_REGISTER_KEY(EKC_SEMICOLON, ImGuiKey_Semicolon);
	NBL_REGISTER_KEY(EKC_OPEN_BRACKET, ImGuiKey_LeftBracket);
	NBL_REGISTER_KEY(EKC_CLOSE_BRACKET, ImGuiKey_RightBracket);
	NBL_REGISTER_KEY(EKC_BACKSLASH, ImGuiKey_Backslash);
	NBL_REGISTER_KEY(EKC_APOSTROPHE, ImGuiKey_Apostrophe);
	NBL_REGISTER_KEY(EKC_ADD, ImGuiKey_KeypadAdd);
	NBL_REGISTER_KEY(EKC_SUBTRACT, ImGuiKey_KeypadSubtract);
	NBL_REGISTER_KEY(EKC_MULTIPLY, ImGuiKey_KeypadMultiply);
	NBL_REGISTER_KEY(EKC_DIVIDE, ImGuiKey_KeypadDivide);
	NBL_REGISTER_KEY(EKC_0, ImGuiKey_0);
	NBL_REGISTER_KEY(EKC_1, ImGuiKey_1);
	NBL_REGISTER_KEY(EKC_2, ImGuiKey_2);
	NBL_REGISTER_KEY(EKC_3, ImGuiKey_3);
	NBL_REGISTER_KEY(EKC_4, ImGuiKey_4);
	NBL_REGISTER_KEY(EKC_5, ImGuiKey_5);
	NBL_REGISTER_KEY(EKC_6, ImGuiKey_6);
	NBL_REGISTER_KEY(EKC_7, ImGuiKey_7);
	NBL_REGISTER_KEY(EKC_8, ImGuiKey_8);
	NBL_REGISTER_KEY(EKC_9, ImGuiKey_9);
	NBL_REGISTER_KEY(EKC_A, ImGuiKey_A);
	NBL_REGISTER_KEY(EKC_B, ImGuiKey_B);
	NBL_REGISTER_KEY(EKC_C, ImGuiKey_C);
	NBL_REGISTER_KEY(EKC_D, ImGuiKey_D);
	NBL_REGISTER_KEY(EKC_E, ImGuiKey_E);
	NBL_REGISTER_KEY(EKC_F, ImGuiKey_F);
	NBL_REGISTER_KEY(EKC_G, ImGuiKey_G);
	NBL_REGISTER_KEY(EKC_H, ImGuiKey_H);
	NBL_REGISTER_KEY(EKC_I, ImGuiKey_I);
	NBL_REGISTER_KEY(EKC_J, ImGuiKey_J);
	NBL_REGISTER_KEY(EKC_K, ImGuiKey_K);
	NBL_REGISTER_KEY(EKC_L, ImGuiKey_L);
	NBL_REGISTER_KEY(EKC_M, ImGuiKey_M);
	NBL_REGISTER_KEY(EKC_N, ImGuiKey_N);
	NBL_REGISTER_KEY(EKC_O, ImGuiKey_O);
	NBL_REGISTER_KEY(EKC_P, ImGuiKey_P);
	NBL_REGISTER_KEY(EKC_Q, ImGuiKey_Q);
	NBL_REGISTER_KEY(EKC_R, ImGuiKey_R);
	NBL_REGISTER_KEY(EKC_S, ImGuiKey_S);
	NBL_REGISTER_KEY(EKC_T, ImGuiKey_T);
	NBL_REGISTER_KEY(EKC_U, ImGuiKey_U);
	NBL_REGISTER_KEY(EKC_V, ImGuiKey_V);
	NBL_REGISTER_KEY(EKC_W, ImGuiKey_W);
	NBL_REGISTER_KEY(EKC_X, ImGuiKey_X);
	NBL_REGISTER_KEY(EKC_Y, ImGuiKey_Y);
	NBL_REGISTER_KEY(EKC_Z, ImGuiKey_Z);
	NBL_REGISTER_KEY(EKC_NUMPAD_0, ImGuiKey_Keypad0);
	NBL_REGISTER_KEY(EKC_NUMPAD_1, ImGuiKey_Keypad1);
	NBL_REGISTER_KEY(EKC_NUMPAD_2, ImGuiKey_Keypad2);
	NBL_REGISTER_KEY(EKC_NUMPAD_3, ImGuiKey_Keypad3);
	NBL_REGISTER_KEY(EKC_NUMPAD_4, ImGuiKey_Keypad4);
	NBL_REGISTER_KEY(EKC_NUMPAD_5, ImGuiKey_Keypad5);
	NBL_REGISTER_KEY(EKC_NUMPAD_6, ImGuiKey_Keypad6);
	NBL_REGISTER_KEY(EKC_NUMPAD_7, ImGuiKey_Keypad7);
	NBL_REGISTER_KEY(EKC_NUMPAD_8, ImGuiKey_Keypad8);
	NBL_REGISTER_KEY(EKC_NUMPAD_9, ImGuiKey_Keypad9);
	NBL_REGISTER_KEY(EKC_F1, ImGuiKey_F1);
	NBL_REGISTER_KEY(EKC_F2, ImGuiKey_F2);
	NBL_REGISTER_KEY(EKC_F3, ImGuiKey_F3);
	NBL_REGISTER_KEY(EKC_F4, ImGuiKey_F4);
	NBL_REGISTER_KEY(EKC_F5, ImGuiKey_F5);
	NBL_REGISTER_KEY(EKC_F6, ImGuiKey_F6);
	NBL_REGISTER_KEY(EKC_F7, ImGuiKey_F7);
	NBL_REGISTER_KEY(EKC_F8, ImGuiKey_F8);
	NBL_REGISTER_KEY(EKC_F9, ImGuiKey_F9);
	NBL_REGISTER_KEY(EKC_F10, ImGuiKey_F10);
	NBL_REGISTER_KEY(EKC_F11, ImGuiKey_F11);
	NBL_REGISTER_KEY(EKC_F12, ImGuiKey_F12);
	NBL_REGISTER_KEY(EKC_F13, ImGuiKey_F13);
	NBL_REGISTER_KEY(EKC_F14, ImGuiKey_F14);
	NBL_REGISTER_KEY(EKC_F15, ImGuiKey_F15);
	NBL_REGISTER_KEY(EKC_F16, ImGuiKey_F16);
	NBL_REGISTER_KEY(EKC_F17, ImGuiKey_F17);
	NBL_REGISTER_KEY(EKC_F18, ImGuiKey_F18);
	NBL_REGISTER_KEY(EKC_F19, ImGuiKey_F19);
	NBL_REGISTER_KEY(EKC_F20, ImGuiKey_F20);
	NBL_REGISTER_KEY(EKC_F21, ImGuiKey_F21);
	NBL_REGISTER_KEY(EKC_F22, ImGuiKey_F22);
	NBL_REGISTER_KEY(EKC_F23, ImGuiKey_F23);
	NBL_REGISTER_KEY(EKC_F24, ImGuiKey_F24);
	NBL_REGISTER_KEY(EKC_NUM_LOCK, ImGuiKey_NumLock);
	NBL_REGISTER_KEY(EKC_SCROLL_LOCK, ImGuiKey_ScrollLock);
	NBL_REGISTER_KEY(EKC_VOLUME_MUTE, ImGuiKey_None);
	NBL_REGISTER_KEY(EKC_VOLUME_UP, ImGuiKey_None);
	NBL_REGISTER_KEY(EKC_VOLUME_DOWN, ImGuiKey_None);

	return map;
}

void UI::handleKeyEvents(const SUpdateParameters& params) const
{
	auto& io = ImGui::GetIO();

	_NBL_STATIC_INLINE_CONSTEXPR auto keyMap = createKeyMap();

	const bool useBigLetters = [&]()  // TODO: we can later improve it to check for CAPS, etc
	{
		for (const auto& e : params.keyboardEvents)
			if (e.keyCode == EKC_LEFT_SHIFT && e.action == SKeyboardEvent::ECA_PRESSED)
				return true;

		return false;
	}();

	for (const auto& e : params.keyboardEvents)
	{
		const auto& bind = keyMap[e.keyCode];
		const auto& iCharacter = useBigLetters ? bind.physicalBig : bind.physicalSmall;

		if(bind.target == ImGuiKey_None)
			m_cachedCreationParams.utilities->getLogger()->log(std::string("Requested physical Nabla key \"") + iCharacter + std::string("\" has yet no mapping to IMGUI key!"), ILogger::ELL_ERROR);
		else
			if (e.action == SKeyboardEvent::ECA_PRESSED)
			{
				io.AddKeyEvent(bind.target, true);
				io.AddInputCharacter(iCharacter);
			}
			else if (e.action == SKeyboardEvent::ECA_RELEASED)
				io.AddKeyEvent(bind.target, false);
	}
}

bool UI::validateCreationParameters(SCreationParameters& creationParams)
{
	system::logger_opt_ptr logger = creationParams.utilities->getLogger();

	auto validateResourcesInfo = [&]() -> bool
	{
		auto* pipelineLayout = creationParams.pipelineLayout.get();

		if (pipelineLayout) // provided? we will validate your pipeline layout to check if you declared required UI resources
		{
			auto validateResource = [&]<IDescriptor::E_TYPE descriptorType>(const IGPUDescriptorSetLayout* const descriptorSetLayout)
			{
				static_assert(descriptorType != IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER, "Explicitly not supported.");
				constexpr std::string_view typeLiteral = descriptorType == IDescriptor::E_TYPE::ET_SAMPLED_IMAGE ? "ET_SAMPLED_IMAGE" : "ET_SAMPLER",
					ixLiteral = descriptorType == IDescriptor::E_TYPE::ET_SAMPLED_IMAGE ? "texturesBindingIx" : "samplersBindingIx";

				// we need to check if there is at least single "descriptorType" resource, if so we can validate the resource further
				auto anyBindingCount = [logger, &creationParams = creationParams, &log = std::as_const(typeLiteral)](const IDescriptorSetLayoutBase::CBindingRedirect* redirect, bool logError = true) -> bool
					{
						bool ok = redirect->getBindingCount();

						if (!ok && logError)
						{
							logger.log("Provided descriptor set layout has no bindings for IDescriptor::E_TYPE::%s, you are required to provide at least single default ImGUI Font Atlas texture resource & corresponsing sampler resource!", ILogger::ELL_ERROR, log.data());
							return false;
						}

						return ok;
					};

				if (!descriptorSetLayout)
				{
					logger.log("Provided descriptor set layout for IDescriptor::E_TYPE::%s is nullptr!", ILogger::ELL_ERROR, typeLiteral.data());
					return false;
				}

				using redirect_t = IDescriptorSetLayoutBase::CBindingRedirect;
				const redirect_t* redirect = &descriptorSetLayout->getDescriptorRedirect(descriptorType);

				if constexpr (descriptorType == IDescriptor::E_TYPE::ET_SAMPLED_IMAGE)
				{
					if (!anyBindingCount(redirect))
						return false;
				}
				else
				{
					if (!anyBindingCount(redirect, false))
					{
						redirect = &descriptorSetLayout->getImmutableSamplerRedirect(); // we must give it another try & request to look for immutable samplers

						if (!anyBindingCount(redirect))
							return false;
					}
				}

				const redirect_t::binding_number_t requestedBinding(descriptorType == IDescriptor::E_TYPE::ET_SAMPLED_IMAGE ? creationParams.resources.texturesInfo.bindingIx : creationParams.resources.samplersInfo.bindingIx);
				const auto storageIndex = redirect->findBindingStorageIndex(requestedBinding);
				if (!storageIndex)
				{
					logger.log("No IDescriptor::E_TYPE::%s binding exists for requested `creationParams.resources.%s=%d` in the Provided descriptor set layout!", ILogger::ELL_ERROR, typeLiteral.data(), ixLiteral.data(), requestedBinding.data);
					return false;
				}

				const auto creation = redirect->getCreateFlags(storageIndex);
				if (!creation.hasFlags(descriptorType == IDescriptor::E_TYPE::ET_SAMPLED_IMAGE ? creationParams.resources.TexturesRequiredCreateFlags : creationParams.resources.SamplersRequiredCreateFlags))
				{
					logger.log("Provided descriptor set layout has IDescriptor::E_TYPE::%s binding for requested `creationParams.resources.%s` index but doesn't meet create flags requirements!", ILogger::ELL_ERROR, typeLiteral.data(), ixLiteral.data());
					return false;
				}
				const auto stage = redirect->getStageFlags(storageIndex);
				if (!stage.hasFlags(creationParams.resources.RequiredShaderStageFlags))
				{
					logger.log("Provided descriptor set layout has IDescriptor::E_TYPE::%s binding for requested `creationParams.resources.%s` index but doesn't meet stage flags requirements!", ILogger::ELL_ERROR, typeLiteral.data(), ixLiteral.data());
					return false;
				}
				const auto count = redirect->getCount(storageIndex);
				if (!count)
				{
					logger.log("Provided descriptor set layout has IDescriptor::E_TYPE::%s binding for requested `creationParams.resources.%s` index but the binding resource count == 0u!", ILogger::ELL_ERROR, typeLiteral.data(), ixLiteral.data());
					return false;
				}

				if constexpr (descriptorType == IDescriptor::E_TYPE::ET_SAMPLED_IMAGE)
					creationParams.resources.texturesCount = count;
				else
					creationParams.resources.samplersCount = count;
				return true;
			};

			const auto& layouts = pipelineLayout->getDescriptorSetLayouts();
			const bool ok = validateResource.template operator() < IDescriptor::E_TYPE::ET_SAMPLED_IMAGE > (layouts[creationParams.resources.texturesInfo.setIx]) && validateResource.template operator() < IDescriptor::E_TYPE::ET_SAMPLER > (layouts[creationParams.resources.samplersInfo.setIx]);

			if (!ok)
				return false;
		}

		return true;
	};

	const auto validation = std::to_array
	({
		std::make_pair(bool(creationParams.assetManager), "Invalid `creationParams.assetManager` is nullptr!"),
		std::make_pair(bool(creationParams.assetManager->getSystem()), "Invalid `creationParams.assetManager->getSystem()` is nullptr!"),
		std::make_pair(bool(creationParams.utilities), "Invalid `creationParams.utilities` is nullptr!"),
		std::make_pair(bool(creationParams.transfer), "Invalid `creationParams.transfer` is nullptr!"),
		std::make_pair(bool(creationParams.renderpass), "Invalid `creationParams.renderpass` is nullptr!"),
		(creationParams.assetManager && creationParams.utilities && creationParams.transfer && creationParams.renderpass) ? std::make_pair(bool(creationParams.utilities->getLogicalDevice()->getPhysicalDevice()->getQueueFamilyProperties()[creationParams.transfer->getFamilyIndex()].queueFlags.hasFlags(IQueue::FAMILY_FLAGS::TRANSFER_BIT)), "Invalid `creationParams.transfer` is not capable of transfer operations!") : std::make_pair(false, "Pass valid required UI::S_CREATION_PARAMETERS!"),
		std::make_pair(bool(creationParams.resources.texturesInfo.setIx <= 3u), "Invalid `creationParams.resources.textures` is outside { 0u, 1u, 2u, 3u } set!"),
		std::make_pair(bool(creationParams.resources.samplersInfo.setIx <= 3u), "Invalid `creationParams.resources.samplers` is outside { 0u, 1u, 2u, 3u } set!"),
		std::make_pair(bool(creationParams.resources.texturesInfo.bindingIx != creationParams.resources.samplersInfo.bindingIx), "Invalid `creationParams.resources.textures.bindingIx` is equal to `creationParams.resources.samplers.bindingIx`!"),
		std::make_pair(bool(validateResourcesInfo()), "Invalid `creationParams.resources` content!")
	});

	for (const auto& [ok, error] : validation)
		if (!ok)
		{
			logger.log(error, ILogger::ELL_ERROR);
			return false;
		}

	return true;
}

core::smart_refctd_ptr<UI> UI::create(SCreationParameters&& creationParams, void* const imSharedFontAtlas)
{
	auto* const logger = creationParams.utilities->getLogger();

	if (!validateCreationParameters(creationParams))
	{
		logger->log("Failed creation parameters validation!", ILogger::ELL_ERROR);
		return nullptr;
	}

	auto pipeline = createPipeline(creationParams);

	if (!pipeline)
	{
		logger->log("Failed pipeline creation!", ILogger::ELL_ERROR);
		return nullptr;
	}

	if (!createMDIBuffer(creationParams))
	{
		logger->log("Failed at mdi buffer creation!", ILogger::ELL_ERROR);
		return nullptr;
	}

	// note that imgui context allows you to share atlas at creation time so we do too, 
	// hence you can have *custom* default font texture we will create a image view from 
	// + obvsly ownership of the atlas is yours then
	auto* const inAtlas = reinterpret_cast<ImFontAtlas*>(imSharedFontAtlas);
	const bool isFontAtlasShared = inAtlas;
	auto* imFontAtlas = isFontAtlasShared ? inAtlas : IM_NEW(ImFontAtlas)();

	auto fontView = createFontAtlasTexture(creationParams,imFontAtlas);
	if (!fontView)
	{
		if (!isFontAtlasShared) // carefully here
			IM_DELETE(imFontAtlas); // if that was supposed to be ours then on fail we kill it
		logger->log("Failed default font image view creation!", ILogger::ELL_ERROR);
		return nullptr;
	}

	// Dear ImGui context
	IMGUI_CHECKVERSION();

	// now create imgui context only if we created default image view to default font atlas image,
	// we benefit from sharing font atlas with the context & allow to decidewho owns the atlas
	// - the UI extension instance or user (because it comes from outside)
	ImGuiContext* context = ImGui::CreateContext(imFontAtlas);
	if (!context)
	{
		logger->log("Failed to create ImGUI context!", ILogger::ELL_ERROR);
		return nullptr;
	}

	// note that you can still change im font atlas at runtime but if the atlas belongs to us
	// then we must track the pointer since we own it and must free it at destruction
	void* const trackedAtlasPointer = !isFontAtlasShared ? imFontAtlas : nullptr;

	return core::smart_refctd_ptr<UI>(new UI(std::move(creationParams), pipeline, fontView, trackedAtlasPointer, context), core::dont_grab);
}

UI::UI(SCreationParameters&& creationParams, core::smart_refctd_ptr<video::IGPUGraphicsPipeline> pipeline, core::smart_refctd_ptr<video::IGPUImageView> defaultFont, void* const imFontAtlas, void* const imContext)
	: m_cachedCreationParams(std::move(creationParams)), m_pipeline(std::move(pipeline)), m_fontAtlasTexture(std::move(defaultFont)), m_imFontAtlasBackPointer(imFontAtlas), m_imContextBackPointer(imContext)
{
	auto& io = ImGui::GetIO();

	// using AddKeyEvent() - it's new way of handling ImGUI events our backends supports
	io.BackendUsingLegacyKeyArrays = 0;
}

UI::~UI()
{
	// I assume somebody has not killed ImGUI context & atlas but if so then we do nothing

	// we must call it to unlock atlas from potential "render" state before we kill it (obvsly if its ours!)
	if(m_imFontAtlasBackPointer)
		ImGui::EndFrame();

	// context belongs to the instance, we must free it
	if(m_imContextBackPointer)
		ImGui::DestroyContext(reinterpret_cast<ImGuiContext*>(m_imContextBackPointer));

	// and if we own the atlas we must free it as well, if user passed its own at creation time then its "shared" - at this point m_imFontAtlasBackPointer is nullptr and we don't free anything
	if (m_imFontAtlasBackPointer)
		IM_DELETE(reinterpret_cast<ImFontAtlas*>(m_imFontAtlasBackPointer));
}

bool UI::createMDIBuffer(SCreationParameters& creationParams)
{
	constexpr static uint32_t minStreamingBufferAllocationSize = 128u, maxStreamingBufferAllocationAlignment = 4096u, mdiBufferDefaultSize = /* 2MB */ 1024u * 1024u * 2u;

	auto getRequiredAccessFlags = [&](const bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS>& properties)
	{
		bitflag<IDeviceMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAGS> flags (IDeviceMemoryAllocation::EMCAF_NO_MAPPING_ACCESS);

		if (properties.hasFlags(IDeviceMemoryAllocation::EMPF_HOST_READABLE_BIT))
			flags |= IDeviceMemoryAllocation::EMCAF_READ;
		if (properties.hasFlags(IDeviceMemoryAllocation::EMPF_HOST_WRITABLE_BIT))
			flags |= IDeviceMemoryAllocation::EMCAF_WRITE;

		return flags;
	};

	if (!creationParams.streamingBuffer)
	{
		IGPUBuffer::SCreationParams mdiCreationParams = {};
		mdiCreationParams.usage = SCachedCreationParams::RequiredUsageFlags;
		mdiCreationParams.size = mdiBufferDefaultSize;

		auto buffer = creationParams.utilities->getLogicalDevice()->createBuffer(std::move(mdiCreationParams));
		buffer->setObjectDebugName("MDI Upstream Buffer");

		auto memoryReqs = buffer->getMemoryReqs();
		memoryReqs.memoryTypeBits &= creationParams.utilities->getLogicalDevice()->getPhysicalDevice()->getUpStreamingMemoryTypeBits();

		auto allocation = creationParams.utilities->getLogicalDevice()->allocate(memoryReqs,buffer.get(),SCachedCreationParams::RequiredAllocateFlags);
		{
			const bool allocated = allocation.isValid();
			assert(allocated);
		}
		auto memory = allocation.memory;

		if (!memory->map({ 0ull, memoryReqs.size }, getRequiredAccessFlags(memory->getMemoryPropertyFlags())))
			creationParams.utilities->getLogger()->log("Could not map device memory!", ILogger::ELL_ERROR);

		creationParams.streamingBuffer = make_smart_refctd_ptr<SCachedCreationParams::streaming_buffer_t>(SBufferRange<IGPUBuffer>{0ull,mdiCreationParams.size,std::move(buffer)},maxStreamingBufferAllocationAlignment,minStreamingBufferAllocationSize);
	}

	auto buffer = creationParams.streamingBuffer->getBuffer();
	auto binding = buffer->getBoundMemory();

	const auto validation = std::to_array
	({
		std::make_pair(buffer->getCreationParams().usage.hasFlags(SCachedCreationParams::RequiredUsageFlags), "MDI buffer must be created with IBuffer::EUF_INDIRECT_BUFFER_BIT | IBuffer::EUF_INDEX_BUFFER_BIT | IBuffer::EUF_VERTEX_BUFFER_BIT | IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT enabled!"),
		std::make_pair(bool(buffer->getMemoryReqs().memoryTypeBits & creationParams.utilities->getLogicalDevice()->getPhysicalDevice()->getUpStreamingMemoryTypeBits()), "MDI buffer must have up-streaming memory type bits enabled!"),
		std::make_pair(binding.memory->getAllocateFlags().hasFlags(SCachedCreationParams::RequiredAllocateFlags), "MDI buffer's memory must be allocated with IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT enabled!"),
		std::make_pair(binding.memory->isCurrentlyMapped(), "MDI buffer's memory must be mapped!"), // streaming buffer contructor already validates it, but cannot assume user won't unmap its own buffer for some reason (sorry if you have just hit it)
		std::make_pair(binding.memory->getCurrentMappingAccess().hasFlags(getRequiredAccessFlags(binding.memory->getMemoryPropertyFlags())), "MDI buffer's memory current mapping access flags don't meet requirements!")
	});

	for (const auto& [ok, error] : validation)
		if (!ok)
		{
			creationParams.utilities->getLogger()->log(error, ILogger::ELL_ERROR);
			return false;
		}

	return true;
}

bool UI::render(IGPUCommandBuffer* const commandBuffer, ISemaphore::SWaitInfo waitInfo, const std::chrono::steady_clock::time_point waitPoint, const std::span<const VkRect2D> scissors)
{
	system::logger_opt_ptr logger = m_cachedCreationParams.utilities->getLogger();
	if (!commandBuffer)
	{
		logger.log("Invalid command buffer!", ILogger::ELL_ERROR);
		return false;
	}

	if (commandBuffer->getState() != IGPUCommandBuffer::STATE::RECORDING)
	{
		logger.log("Command buffer is not in recording state!", ILogger::ELL_ERROR);
		return false;
	}

	{
		const auto info = commandBuffer->getCachedInheritanceInfo();
		const bool recordingSubpass = info.subpass != IGPURenderpass::SCreationParams::SSubpassDependency::External;

		if(!recordingSubpass)
		{
			logger.log("Command buffer is not recording a subpass!", ILogger::ELL_ERROR);
			return false;
		}
	}

	// TODO: clean the typedef up
	using streaming_buffer_t = video::StreamingTransientDataBufferST<core::allocator<uint8_t>>;
	using offset_t = streaming_buffer_t::size_type;
	// A little bit of math to workout our max alignment
	constexpr auto MdiSizes = std::to_array<offset_t>({ sizeof(VkDrawIndexedIndirectCommand), sizeof(PerObjectData), sizeof(ImDrawIdx), sizeof(ImDrawVert) });
	// shared nPoT alignment needs to be divisible by all smaller ones to satisfy an allocation from all
	constexpr offset_t MaxAlignment = std::reduce(MdiSizes.begin(),MdiSizes.end(),1,[](const offset_t a, const offset_t b)->offset_t{return std::lcm(a,b);});
	// allocator initialization needs us to round up to PoT
	const auto MaxPOTAlignment = roundUpToPoT(MaxAlignment);
	// quick sanity check
	auto* streaming = m_cachedCreationParams.streamingBuffer.get();
	if (streaming->getAddressAllocator().max_alignment()<MaxPOTAlignment)
	{
		logger.log("IMGUI Streaming Buffer cannot guarantee the alignments we require!");
		return false;
	}

	ImGui::Render(); // note it doesn't touch GPU or graphics API at all, its an internal ImGUI call to update & prepare the data for rendering so we can call GetDrawData()

	ImGuiIO& io = ImGui::GetIO();

	if (!io.Fonts->IsBuilt())
	{
		logger.log("Font atlas not built! It is generally built by the renderer backend. Missing call to renderer _NewFrame() function? e.g. ImGui_ImplOpenGL3_NewFrame().", ILogger::ELL_ERROR);
		return false;
	}
		
	auto const* drawData = ImGui::GetDrawData();

	if (!drawData)
		return false;
		
	// Avoid rendering when minimized, scale coordinates for retina displays (screen coordinates != framebuffer coordinates)
	float const frameBufferWidth = drawData->DisplaySize.x * drawData->FramebufferScale.x;
	float const frameBufferHeight = drawData->DisplaySize.y * drawData->FramebufferScale.y;
	if (frameBufferWidth > 0 && frameBufferHeight > 0 && drawData->TotalVtxCount > 0)
	{
		// set our viewport and scissor, there's a deferred todo https://github.com/Devsh-Graphics-Programming/Nabla/issues/751
		SViewport const viewport
		{
			.x = 0,
			.y = 0,
			.width = frameBufferWidth,
			.height = frameBufferHeight,
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};
		commandBuffer->setViewport(0u, 1u, &viewport);
		{
			if (scissors.empty())
			{
				VkRect2D defaultScisors[] = { {.offset = {(int32_t)viewport.x, (int32_t)viewport.y}, .extent = {(uint32_t)viewport.width, (uint32_t)viewport.height}} };
				commandBuffer->setScissor(defaultScisors); // cover whole viewport (dynamic scissors must be set only to not throw validation errors)
			}
			else
				commandBuffer->setScissor(scissors);
		}

		// get structs for computing the NDC coordinates ready
		struct TRS
		{
			float32_t2 scale;
			float32_t2 translate;

			float32_t2 toNDC(float32_t2 in) const
			{
				return in * scale + translate;
			}
		};
		const TRS trs = {
			.scale = float32_t2{ 2.0f / drawData->DisplaySize.x , 2.0f / drawData->DisplaySize.y },
			.translate = float32_t2 { -1.0f, -1.0f } - float32_t2{ drawData->DisplayPos.x, drawData->DisplayPos.y }*trs.scale,
		};
		
		// everything binds the buffer once at base and works via local offsets
		const SBufferBinding<const IGPUBuffer> binding =
		{
			.offset = 0u,
			.buffer = smart_refctd_ptr<const IGPUBuffer>(streaming->getBuffer())
		};
		{
			constexpr auto IndexType = sizeof(ImDrawIdx)==2u ? EIT_16BIT:EIT_32BIT;
			if (!commandBuffer->bindIndexBuffer(binding,IndexType))
				return false; // commandBuffer should log failures if any
		}
		if (!commandBuffer->bindVertexBuffers(0u,1u,&binding))
			return false;

		// now proceed with filling the streaming buffer with our data
		{
			// Use LinearAddressAllocators to allocate vertex and index spans
			using suballocator_t = core::LinearAddressAllocatorST<offset_t>;
			constexpr offset_t ImaginarySizeUpperBound = 0x1<<30;
			constexpr offset_t GeoAlignment = std::lcm(sizeof(ImDrawIdx),sizeof(ImDrawVert));
			// We allocate in an imaginary infinte linear buffers, then actually commit to the allocations when we no longer can fit in the allocated chunk
			struct MetaAllocator
			{
				public:
					MetaAllocator() : geoAllocator(nullptr,0,0,core::roundUpToPoT(GeoAlignment),ImaginarySizeUpperBound) {}

					offset_t getElementCount() const {return drawIndirectCount;}

					// doesn't reset the `memBlock` that gets handled outside
					void reset()
					{
						geoAllocator.reset();
						drawIndirectCount = 0;
					}

					struct PushResult
					{
						// these offsets are relative to the offset for all geometry data
						offset_t indexByteOffset;
						offset_t vertexByteOffset;
					};
					PushResult push(const ImDrawList* list)
					{
						drawIndirectCount += list->CmdBuffer.Size;
						PushResult retval = {};
						retval.indexByteOffset = geoAllocator.alloc_addr(list->IdxBuffer.size_in_bytes(),sizeof(ImDrawIdx));
						retval.vertexByteOffset = geoAllocator.alloc_addr(list->VtxBuffer.size_in_bytes(),sizeof(ImDrawVert));
						// should never happen, the linear address allocator space is enormous
						const auto InvalidAddress = suballocator_t::invalid_address;
						assert(retval.indexByteOffset!=InvalidAddress && retval.vertexByteOffset!=InvalidAddress);
						return retval;
					}

					struct FinalizeResult
					{
						// these are total, from streaming buffer start
						offset_t drawIndirectByteOffset;
						offset_t perDrawByteOffset;
						offset_t geometryByteOffset;
						offset_t totalSize;
					};
					// so now if we were (we are not yet) to make a linear address allocator over a free chunk, how much space would our allocations take and what would be their offsets
					FinalizeResult finalize() const
					{
						suballocator_t imaginaryChunk(nullptr,memBlockOffset,0,roundUpToPoT(MaxAlignment),ImaginarySizeUpperBound);
						FinalizeResult retval = {};
						retval.drawIndirectByteOffset = imaginaryChunk.alloc_addr(sizeof(VkDrawIndexedIndirectCommand)*drawIndirectCount,sizeof(VkDrawIndexedIndirectCommand));
						retval.perDrawByteOffset = imaginaryChunk.alloc_addr(sizeof(PerObjectData)*drawIndirectCount,sizeof(PerObjectData));
						retval.geometryByteOffset = imaginaryChunk.alloc_addr(geoAllocator.get_allocated_size(),GeoAlignment);
						retval.totalSize = imaginaryChunk.get_allocated_size();
						// should never happen, the linear address allocator space is enormous
						const auto InvalidAddress = suballocator_t::invalid_address;
						assert(retval.drawIndirectByteOffset!=InvalidAddress && retval.perDrawByteOffset!=InvalidAddress && retval.geometryByteOffset!=InvalidAddress);
						// should always allocate at the start, the alignment was correct
						assert(retval.drawIndirectByteOffset==memBlockOffset && (retval.drawIndirectByteOffset%sizeof(VkDrawIndexedIndirectCommand))==0);
						// element alignment check only, geometry will be checked after every push in `endDrawBatch`
						assert((retval.perDrawByteOffset%sizeof(PerObjectData))==0);
						return retval;
					}

					bool tryPush(const ImDrawList* list)
					{
						// backup old state
						const auto oldAllocatorCursor = geoAllocator.get_allocated_size();
						// perform and check
						push(list);
						if (finalize().totalSize>memBlockSize)
						{
							// unwind the allocations
							drawIndirectCount -= list->CmdBuffer.Size;
							geoAllocator.reset(oldAllocatorCursor);
							return false;
						}
						return true;
					}
					
					// allocate max sized chunk and set up our linear allocators
					offset_t memBlockOffset;
					offset_t memBlockSize = 0;

				private:
					suballocator_t geoAllocator;
					// however DrawIndirect and ElementData structs need to be close together
					offset_t drawIndirectCount = 0;
			} metaAlloc;

			// Run Length Encode pattern, you advance iterator until some change occurs, which forces a "block end" behaviour
			auto listIt = drawData->CmdLists.begin();
			auto lastBatchEnd = listIt;

			// can't be a static unfortunately
			const auto noWait = std::chrono::steady_clock::now();
			// get this set up once
			const struct
			{
				ImVec2 off;			// (0,0) unless using multi-viewports
				ImVec2 scale;		// (1,1) unless using retina display which are often (2,2)
				ImVec2 framebuffer; // width, height

				// Project scissor/clipping rectangles into frame-buffer space
				ImVec4 getClipRectangle(const ImDrawCmd& cmd) const
				{
					ImVec4 rectangle;
					rectangle.x = (cmd.ClipRect.x - off.x) * scale.x;
					rectangle.y = (cmd.ClipRect.y - off.y) * scale.y;
					rectangle.z = (cmd.ClipRect.z - off.x) * scale.x;
					rectangle.w = (cmd.ClipRect.w - off.y) * scale.y;

					return rectangle;
				}

				VkRect2D getScissor(ImVec4 clipRectangle) const
				{
					// Negative offsets are illegal for vkCmdSetScissor
					if (clipRectangle.x < 0.0f)
						clipRectangle.x = 0.0f;
					if (clipRectangle.y < 0.0f)
						clipRectangle.y = 0.0f;

					{// Apply scissor/clipping rectangle
						VkRect2D scissor{};
						scissor.offset.x = static_cast<int32_t>(clipRectangle.x);
						scissor.offset.y = static_cast<int32_t>(clipRectangle.y);
						scissor.extent.width = static_cast<uint32_t>(clipRectangle.z - clipRectangle.x);
						scissor.extent.height = static_cast<uint32_t>(clipRectangle.w - clipRectangle.y);

						return scissor;
					}
				}
			} clip{ .off = drawData->DisplayPos, .scale = drawData->FramebufferScale, .framebuffer = { frameBufferWidth, frameBufferHeight } };

			// actual lambda
			auto beginDrawBatch = [&]()->bool
			{
				// push first item, because we need to fit at least one draw
				metaAlloc.push(*(listIt++));
				metaAlloc.memBlockOffset = 0;
				const uint32_t SmallestAlloc = metaAlloc.finalize().totalSize;
				metaAlloc.memBlockOffset = streaming_buffer_t::invalid_value;
				// 2 tries
				for (auto t=0; t<2; t++)
				{
					// Allocate a chunk as large as possible, a bit of trivia, `max_size` pessimizes the size assuming you're going to ask for allocation with Allocator's Max Alignment
					// There's a bit of a delay/inaccuracy with `max_size` so try many sizes before giving up
					// Also `max_size` doesn't defragment, meaning it cannot be trusted to be accurate, so force at least one try with the size we need
					metaAlloc.memBlockSize = core::max(streaming->max_size(),SmallestAlloc);
					while (metaAlloc.memBlockSize>=SmallestAlloc)
					{
						// first time don't wait and require block ASAP, second time wait till timeout point
						// NOTE: With dynamic rendering we'd do a shorter wait and overflow submit if we still fail
						if (streaming->multi_allocate(t ? waitPoint:noWait,1,&metaAlloc.memBlockOffset,&metaAlloc.memBlockSize,&MaxAlignment)==0)
							return true;
						// after first-fail always poll events to completion, this will execute all deferred deallocations it can
						streaming->cull_frees();
						// go fast, O(logN) tries
						metaAlloc.memBlockSize >>= 1;
					}
				}
				logger.log("Failed to allocate even the smallest chunk from streaming buffer for the next drawcall batch.",ILogger::ELL_ERROR);
				return false;
			};

			// the batch end is a lambda cause unfortunately whenever you perform the RLE pattern, you need to perform an implicit run-break outside the loop
			const auto streamingBaseAddress = streaming->getBuffer()->getDeviceAddress();
			auto* const streamingPtr = reinterpret_cast<uint8_t*>(streaming->getBufferPointer());
			assert(streamingPtr);
			// implementation detail of the allocator stays the same
			const auto minBlockSize = streaming->getAddressAllocator().min_size();
			auto endDrawBatch = [&]()->void
			{
				const auto offsets = metaAlloc.finalize();
				const auto endOffset = offsets.drawIndirectByteOffset+offsets.totalSize;
				uint32_t drawID = 0;
				auto* drawIndirectIt = reinterpret_cast<VkDrawIndexedIndirectCommand*>(streamingPtr+offsets.drawIndirectByteOffset);
				auto* elementIt = reinterpret_cast<PerObjectData*>(streamingPtr+offsets.perDrawByteOffset);
				// replay allocations and this time actually memcpy
				{
					const auto endByte = streamingPtr+endOffset;
					metaAlloc.reset();
					for (; lastBatchEnd!=listIt; lastBatchEnd++)
					{
						const auto* list = *lastBatchEnd;
						auto geo = metaAlloc.push(list);
						// now add the global offsets
						geo.indexByteOffset += offsets.geometryByteOffset;
						geo.vertexByteOffset += offsets.geometryByteOffset;
						assert(geo.indexByteOffset<endOffset);
						assert(geo.vertexByteOffset<endOffset);
						// alignments should match
						assert((geo.indexByteOffset%sizeof(ImDrawIdx))==0);
						assert((geo.vertexByteOffset%sizeof(ImDrawVert))==0);
						// offsets since start of buffer, but counted in objects of size T
						const auto idxGlobalObjectOffset = geo.indexByteOffset/sizeof(ImDrawIdx);
						const auto vtxGlobalObjectOffset = geo.vertexByteOffset/sizeof(ImDrawVert);
						// IMGUI's API prevents us from more finely splitting draws (vertex buffer is shared for whole list)
						const auto& imCmdBuf = list->CmdBuffer;
						for (auto j=0; j!=imCmdBuf.size(); j++)
						{
							const auto& cmd = imCmdBuf[j];
							// we use base instance as `gl_DrawID` in case GPU is missing it
							drawIndirectIt->firstInstance = drawID++;
							drawIndirectIt->indexCount = cmd.ElemCount;
							drawIndirectIt->instanceCount = 1u;
							drawIndirectIt->vertexOffset = vtxGlobalObjectOffset+cmd.VtxOffset;
							drawIndirectIt->firstIndex = idxGlobalObjectOffset+cmd.IdxOffset;
							drawIndirectIt++;
							// per obj filling involves clipping
							{
								const auto clipRectangle = clip.getClipRectangle(cmd);
								const auto scissor = clip.getScissor(clipRectangle);
								auto packSnorm16 = [](float ndc) -> int16_t
								{
									return std::round<int16_t>(std::clamp(ndc, -1.0f, 1.0f) * 32767.0f); // TODO: ok encodePixels<EF_R16_SNORM, double>(void* _pix, const double* _input) but iirc we have issues with our encode/decode utils
								};
								const auto vMin = trs.toNDC(float32_t2(scissor.offset.x, scissor.offset.y));
								const auto vMax = trs.toNDC(float32_t2(scissor.offset.x + scissor.extent.width, scissor.offset.y + scissor.extent.height));
								struct snorm16_t2_packed
								{
									int16_t x, y;
								};
								reinterpret_cast<snorm16_t2_packed&>(elementIt->aabbMin) = { .x = packSnorm16(vMin.x), .y = packSnorm16(vMin.y) };
								reinterpret_cast<snorm16_t2_packed&>(elementIt->aabbMax) = { .x = packSnorm16(vMax.x), .y = packSnorm16(vMax.y) };
								elementIt->texId = cmd.TextureId.textureID;
								elementIt->samplerIx = cmd.TextureId.samplerIx;
							}
							elementIt++;
						}
						memcpy(streamingPtr+geo.indexByteOffset,list->IdxBuffer.Data,list->IdxBuffer.size_in_bytes());
						memcpy(streamingPtr+geo.vertexByteOffset,list->VtxBuffer.Data,list->VtxBuffer.size_in_bytes());
						// not writing past the end
						assert(streamingPtr+geo.indexByteOffset+list->IdxBuffer.size_in_bytes()<=endByte);
						assert(streamingPtr+geo.vertexByteOffset+list->VtxBuffer.size_in_bytes()<=endByte);
					}
				}
				// the offets were enough and allocation should not overlap
				assert(reinterpret_cast<VkDrawIndexedIndirectCommand*>(streamingPtr+offsets.perDrawByteOffset)>=drawIndirectIt);
				assert(reinterpret_cast<PerObjectData*>(streamingPtr+offsets.geometryByteOffset)>=elementIt);

				// flush the used range
				assert(!streaming->needsManualFlushOrInvalidate());

				// record draw call
				{
					/*
						Setup scale and translation, our visible imgui space lies from draw_data->DisplayPps (top left) to
						draw_data->DisplayPos+data_data->DisplaySize (bottom right). DisplayPos is (0,0) for single viewport apps.
					*/
					{
						PushConstants constants
						{
							.elementBDA = streamingBaseAddress+offsets.perDrawByteOffset,
							.elementCount = drawID,
							.scale = { trs.scale[0u], trs.scale[1u] },
							.translate = { trs.translate[0u], trs.translate[1u] },
							.viewport = { viewport.x, viewport.y, viewport.width, viewport.height }
						};

						commandBuffer->pushConstants(m_pipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_VERTEX|IShader::E_SHADER_STAGE::ESS_FRAGMENT, 0u, sizeof(constants), &constants);
					}

					{
						auto mdiBinding = binding;
						mdiBinding.offset = offsets.drawIndirectByteOffset;
						commandBuffer->drawIndexedIndirect(mdiBinding,drawID,sizeof(VkDrawIndexedIndirectCommand));
					}
				}

				// See if we can trim our chunk (dealloc unused range at the end immediately). This relies on an implementation detail of GeneralPurposeAllocator,
				// you can slice up an allocation and manage slices independently as along as they are larger than a minimum block/allocation size
				if (offsets.totalSize>=minBlockSize)
				if (const offset_t unusedStart=metaAlloc.memBlockOffset+offsets.totalSize, unusedSize=metaAlloc.memBlockSize-offsets.totalSize; unusedSize>=minBlockSize)
				{
					assert(unusedStart==endOffset);
					assert(unusedStart+unusedSize==metaAlloc.memBlockOffset+metaAlloc.memBlockSize);
					streaming->multi_deallocate(1,&unusedStart,&unusedSize);
					// trim the leftover actually used block
					metaAlloc.memBlockSize = offsets.totalSize;
				}

				// latch our used chunk free
				streaming->multi_deallocate(1,&metaAlloc.memBlockOffset,&metaAlloc.memBlockSize,waitInfo);
				// reset to initial state
				metaAlloc.reset();
			};

			if (!beginDrawBatch())
				return false;
			// We will iterate over the lists and split them into multiple MDI calls if necessary
			for (const auto listEnd=drawData->CmdLists.end(); listIt!=listEnd;)
			{
				if (!metaAlloc.tryPush(*listIt))
				{
					endDrawBatch();
					if (!beginDrawBatch())
						return false;
				}
				// only advance if we fit (otherwise we need to revisit current list for sizing)
				else
					listIt++;
			}
			endDrawBatch();
		}
	}
	
	return true;
}

bool UI::update(const SUpdateParameters& params)
{
	auto & io = ImGui::GetIO();
	
	io.DisplaySize = ImVec2(params.displaySize.x, params.displaySize.y);

	handleMouseEvents(params);
	handleKeyEvents(params);

	ImGui::NewFrame();

	for (auto const& subscriber : m_subscribers)
		subscriber();

	return true;
}

size_t UI::registerListener(const std::function<void()>& listener)
{
	assert(listener != nullptr);
	m_subscribers.emplace_back(listener);
	return m_subscribers.size() - 1u;
}

std::optional<size_t> UI::unregisterListener(size_t id)
{
	if (id < m_subscribers.size())
	{
		m_subscribers.erase(m_subscribers.begin() + id);
		return id;
	}

	return std::nullopt;
}

void* UI::getContext()
{
	return reinterpret_cast<void*>(ImGui::GetCurrentContext());
}

void UI::setContext(void* imguiContext)
{
	ImGui::SetCurrentContext(reinterpret_cast<ImGuiContext*>(imguiContext));
}
}