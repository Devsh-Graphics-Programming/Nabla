#include <iostream>
#include <map>
#include <ranges>
#include <vector>
#include <utility>
#include <locale>
#include <codecvt>

#include "nbl/system/CStdoutLogger.h"
#include "nbl/ext/ImGui/ImGui.h"
#include "shaders/common.hlsl"
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
	using mdi_buffer_t = UI::SMdiBuffer;
	using compose_t = typename mdi_buffer_t::compose_t;
	using mdi_size_t = compose_t::size_type;

	static constexpr auto InvalidAddress = compose_t::invalid_value;
	static constexpr auto MdiAlignments = std::to_array<mdi_size_t>({ alignof(VkDrawIndexedIndirectCommand), alignof(PerObjectData), alignof(ImDrawIdx), alignof(ImDrawVert) });
	static constexpr auto MdiMaxAlignment = *std::max_element(MdiAlignments.begin(), MdiAlignments.end());

	// those must be allocated within single block for content - even though its possible to tell vkCmdDrawIndexedIndirect about stride we cannot guarantee each one will be the same size with our allocation strategy
	enum class TightContent : uint16_t
	{
		INDIRECT_STRUCTURES,
		ELEMENT_STRUCTURES,

		COUNT,
	};

	static constexpr SPushConstantRange PushConstantRanges[] =
	{
		{
			.stageFlags = IShader::E_SHADER_STAGE::ESS_VERTEX | IShader::E_SHADER_STAGE::ESS_FRAGMENT,
			.offset = 0,
			.size = sizeof(PushConstants)
		}
	};

	smart_refctd_ptr<IGPUPipelineLayout> UI::createDefaultPipelineLayout(IUtilities* const utilities, const SResourceParameters::SBindingInfo texturesInfo, const SResourceParameters::SBindingInfo samplersInfo, uint32_t texturesCount)
	{
		if (!utilities)
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

			fontAtlasUISampler = utilities->getLogicalDevice()->createSampler(params);
			fontAtlasUISampler->setObjectDebugName("Nabla default ImGUI font UI sampler");
		}

		{
			IGPUSampler::SParams params;
			params.MinLod = 0.f;
			params.MaxLod = 0.f;
			params.TextureWrapU = ISampler::ETC_CLAMP_TO_EDGE;
			params.TextureWrapV = ISampler::ETC_CLAMP_TO_EDGE;
			params.TextureWrapW = ISampler::ETC_CLAMP_TO_EDGE;

			userTexturesSampler = utilities->getLogicalDevice()->createSampler(params);
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
			layouts[texturesInfo.setIx] = utilities->getLogicalDevice()->createDescriptorSetLayout({ {textureBinding, samplersBinding} });
		else
		{
			layouts[texturesInfo.setIx] = utilities->getLogicalDevice()->createDescriptorSetLayout({ {textureBinding} });
			layouts[samplersInfo.setIx] = utilities->getLogicalDevice()->createDescriptorSetLayout({ {samplersBinding} });
		}

		assert(layouts[texturesInfo.setIx]);
		assert(layouts[samplersInfo.setIx]);

		return utilities->getLogicalDevice()->createPipelineLayout(PushConstantRanges, std::move(layouts[0u]), std::move(layouts[1u]), std::move(layouts[2u]), std::move(layouts[3u]));
	}

	const smart_refctd_ptr<IFileArchive> UI::mount(smart_refctd_ptr<ILogger> logger, ISystem* system, const std::string_view archiveAlias)
	{
		assert(system);
		
		if(!system)
			return nullptr;

		auto archive = make_smart_refctd_ptr<builtin::CArchive>(smart_refctd_ptr(logger));
		system->mount(smart_refctd_ptr(archive), archiveAlias.data());

		return smart_refctd_ptr(archive);
	}

	void UI::createPipeline(SCreationParameters& creationParams)
	{
		auto pipelineLayout = smart_refctd_ptr<IGPUPipelineLayout>(creationParams.pipelineLayout);

		if (!pipelineLayout)
		{
			creationParams.utilities->getLogger()->log("Could not create pipeline layout!", ILogger::ELL_ERROR);
			assert(false);
		}

		struct
		{
			smart_refctd_ptr<IGPUShader> vertex, fragment;
		} shaders;

		{
			constexpr std::string_view NBL_ARCHIVE_ALIAS = "nbl/ext/imgui/shaders";
				
			//! proxy the system, we will touch it gently
			auto system = smart_refctd_ptr<ISystem>(creationParams.assetManager->getSystem());

			//! note we are out of default logical device's compiler set scope so also a few special steps are required to compile our extension shaders to SPIRV
			auto compiler = make_smart_refctd_ptr<CHLSLCompiler>(smart_refctd_ptr(system));	
			auto includeFinder = make_smart_refctd_ptr<IShaderCompiler::CIncludeFinder>(smart_refctd_ptr(system));
			auto includeLoader = includeFinder->getDefaultFileSystemLoader();
			includeFinder->addSearchPath(NBL_ARCHIVE_ALIAS.data(), includeLoader);

			auto createShader = [&]<StringLiteral key, IShader::E_SHADER_STAGE stage>() -> smart_refctd_ptr<IGPUShader>
			{
				IAssetLoader::SAssetLoadParams params = {};
				params.logger = creationParams.utilities->getLogger();
				params.workingDirectory = NBL_ARCHIVE_ALIAS.data();

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

			//! we assume user has all Nabla builtins mounted - we don't check it at release
			assert(system->areBuiltinsMounted());

			//! but we should never assume user will mount our internal archive since its the extension and not user's job to do it, hence we mount ourselves temporary archive to compile our extension sources then unmount it
			auto archive = mount(smart_refctd_ptr<ILogger>(creationParams.utilities->getLogger()), system.get(), NBL_ARCHIVE_ALIAS.data());
			shaders.vertex = createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("vertex.hlsl"), IShader::E_SHADER_STAGE::ESS_VERTEX > ();
			shaders.fragment = createShader.template operator() < NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("fragment.hlsl"), IShader::E_SHADER_STAGE::ESS_FRAGMENT > ();
			system->unmount(archive.get(), NBL_ARCHIVE_ALIAS.data());

			assert(shaders.vertex);
			assert(shaders.fragment);
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
			
			if (!creationParams.utilities->getLogicalDevice()->createGraphicsPipelines(creationParams.pipelineCache.get(), params, &m_pipeline))
			{
				creationParams.utilities->getLogger()->log("Could not create pipeline!", ILogger::ELL_ERROR);
				assert(false);
			}
		}
	}

	ISemaphore::future_t<IQueue::RESULT> UI::createFontAtlasTexture(IGPUCommandBuffer* cmdBuffer, SCreationParameters& creationParams)
	{
		// Load Fonts
		// - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
		// - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
		// - If the file cannot be loaded, the function will return NULL. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
		// - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
		// - Read 'docs/FONTS.md' for more instructions and details.
		// - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
		//io.Fonts->AddFontDefault();
		//io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
		//io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
		//io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
		//io.Fonts->AddFontFromFileTTF("../../misc/fonts/ProggyTiny.ttf", 10.0f);
		//ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());
		ImGuiIO& io = ImGui::GetIO();

		// TODO: don't `pixels` need to be freed somehow!? (Use a uniqueptr with custom deleter lambda)
		uint8_t* pixels = nullptr;
		int32_t width, height;
		io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
		SImResourceInfo info;
		info.textureID = FontAtlasTexId;
		info.samplerIx = FontAtlasSamplerId;

		io.Fonts->SetTexID(info);

		if (!pixels || width<=0 || height<=0)
			return IQueue::RESULT::OTHER_ERROR;

		const size_t componentsCount = 4, image_size = width * height * componentsCount * sizeof(uint8_t);
		
		_NBL_STATIC_INLINE_CONSTEXPR auto NBL_FORMAT_FONT = EF_R8G8B8A8_UNORM;
		const auto buffer = make_smart_refctd_ptr< CCustomAllocatorCPUBuffer<null_allocator<uint8_t>, true> >(image_size, pixels, adopt_memory);
		
		IGPUImage::SCreationParams params;
		params.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);
		params.type = IImage::ET_2D;
		params.format = NBL_FORMAT_FONT;
		params.extent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1u };
		params.mipLevels = 1;
		params.arrayLayers = 1u;
		params.samples = IImage::ESCF_1_BIT;
		params.usage |= IGPUImage::EUF_TRANSFER_DST_BIT | IGPUImage::EUF_SAMPLED_BIT | IGPUImage::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT; // do you really need the SRC bit?

		struct
		{
			smart_refctd_dynamic_array<ICPUImage::SBufferCopy> data = make_refctd_dynamic_array<smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1ull);		
			SRange <ICPUImage::SBufferCopy> range = { data->begin(), data->end() };
			IImage::SSubresourceRange subresource = 
			{
				.aspectMask = IImage::EAF_COLOR_BIT,
				.baseMipLevel = 0u,
				.levelCount = 1u,
				.baseArrayLayer = 0u,
				.layerCount = 1u
			};
		} regions;
		{
			auto* region = regions.data->begin();
			region->bufferOffset = 0ull;
			region->bufferRowLength = params.extent.width;
			region->bufferImageHeight = 0u;
			region->imageSubresource = {};
			region->imageSubresource.aspectMask = IImage::EAF_COLOR_BIT;
			region->imageSubresource.layerCount = 1u;
			region->imageOffset = { 0u, 0u, 0u };
			region->imageExtent = { params.extent.width, params.extent.height, 1u };
		}

		auto image = creationParams.utilities->getLogicalDevice()->createImage(std::move(params));

		if (!image)
		{
			creationParams.utilities->getLogger()->log("Could not create font image!", ILogger::ELL_ERROR);
			return IQueue::RESULT::OTHER_ERROR;
		}
		image->setObjectDebugName("Nabla ImGUI default font");

		if (!creationParams.utilities->getLogicalDevice()->allocate(image->getMemoryReqs(), image.get()).isValid())
		{
			creationParams.utilities->getLogger()->log("Could not allocate memory for font image!", ILogger::ELL_ERROR);
			return IQueue::RESULT::OTHER_ERROR;
		}

		SIntendedSubmitInfo sInfo;
		{
			IQueue::SSubmitInfo::SCommandBufferInfo cmdInfo = { cmdBuffer };

			auto scratchSemaphore = creationParams.utilities->getLogicalDevice()->createSemaphore(0);
			if (!scratchSemaphore)
			{
				creationParams.utilities->getLogger()->log("Could not create scratch semaphore", ILogger::ELL_ERROR);
				return IQueue::RESULT::OTHER_ERROR;
			}
			scratchSemaphore->setObjectDebugName("Nabla IMGUI extension Scratch Semaphore");

			sInfo.queue = creationParams.transfer;
			sInfo.waitSemaphores = {};
			sInfo.commandBuffers = { &cmdInfo, 1 };
			sInfo.scratchSemaphore =
			{
				.semaphore = scratchSemaphore.get(),
				.value = 0u,
				.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
			};
			
			// we have no explicit source stage and access to sync against, brand new clean image.
			const SMemoryBarrier toTransferDep = {
				.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
				.dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT,
			};
			const auto transferLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL;
			// transition to TRANSFER_DST
			IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> barriers[] = 
			{ 
				{
					.barrier = {.dep = toTransferDep},
					.image = image.get(),
					.subresourceRange = regions.subresource,
					.oldLayout = IGPUImage::LAYOUT::UNDEFINED, // wiping transition
					.newLayout = transferLayout
				} 
			};

			cmdBuffer->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			cmdBuffer->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE,{.imgBarriers=barriers});
			// We cannot use the `AutoSubmit` variant of the util because we need to add a pipeline barrier with a transition onto the command buffer after the upload.
			// old layout is UNDEFINED because we don't want a content preserving transition, we can just put ourselves in transfer right away
			if (!creationParams.utilities->updateImageViaStagingBuffer(sInfo,pixels,image->getCreationParameters().format,image.get(),transferLayout,regions.range))
			{
				creationParams.utilities->getLogger()->log("Could not upload font image contents", ILogger::ELL_ERROR);
				return IQueue::RESULT::OTHER_ERROR;
			}

			// we only need to sync with semaphore signal
			barriers[0].barrier.dep = toTransferDep.nextBarrier(sInfo.scratchSemaphore.stageMask,ACCESS_FLAGS::NONE);
			// transition to READ_ONLY_OPTIMAL ready for rendering with sampling
			barriers[0].oldLayout = barriers[0].newLayout;
			barriers[0].newLayout = IGPUImage::LAYOUT::READ_ONLY_OPTIMAL;
			cmdBuffer->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE,{.imgBarriers=barriers});
			cmdBuffer->end();

			const auto submit = sInfo.popSubmit({});
			if (creationParams.transfer->submit(submit)!=IQueue::RESULT::SUCCESS)
			{
				creationParams.utilities->getLogger()->log("Could not submit workload for font texture upload.", ILogger::ELL_ERROR);
				return IQueue::RESULT::OTHER_ERROR;
			}
		}
		 
		{
			IGPUImageView::SCreationParams params;
			params.format = image->getCreationParameters().format;
			params.viewType = IImageView<IGPUImage>::ET_2D;
			params.subresourceRange = regions.subresource;
			params.image = smart_refctd_ptr(image);

			m_fontAtlasTexture = creationParams.utilities->getLogicalDevice()->createImageView(std::move(params));
		}
		
        ISemaphore::future_t<IQueue::RESULT> retval(IQueue::RESULT::SUCCESS);
        retval.set({sInfo.scratchSemaphore.semaphore,sInfo.scratchSemaphore.value});
        return retval;
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
	_NBL_STATIC_INLINE_CONSTEXPR std::array<NBL_TO_IMGUI_KEY_BIND, EKC_COUNT> createKeyMap()
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

	UI::UI(SCreationParameters&& creationParams)
	{
		auto validateResourcesInfo = [&]() -> bool
		{
			auto* pipelineLayout = creationParams.pipelineLayout.get();

			if (pipelineLayout) // provided? we will validate your pipeline layout to check if you declared required UI resources
			{
				auto validateResource = [&]<IDescriptor::E_TYPE descriptorType>(const IGPUDescriptorSetLayout* const descriptorSetLayout)
				{
					constexpr std::string_view typeLiteral = descriptorType == IDescriptor::E_TYPE::ET_SAMPLED_IMAGE ? "ET_SAMPLED_IMAGE" : "ET_SAMPLER",
					ixLiteral = descriptorType == IDescriptor::E_TYPE::ET_SAMPLED_IMAGE ? "texturesBindingIx" : "samplersBindingIx";

					// we need to check if there is at least single "descriptorType" resource, if so we can validate the resource further
					auto anyBindingCount = [&creationParams = creationParams, &log = std::as_const(typeLiteral)](const IDescriptorSetLayoutBase::CBindingRedirect* redirect, bool logError = true) -> bool
					{
						bool ok = redirect->getBindingCount();

						if (!ok && logError)
						{
							creationParams.utilities->getLogger()->log("Provided descriptor set layout has no bindings for IDescriptor::E_TYPE::%s, you are required to provide at least single default ImGUI Font Atlas texture resource & corresponsing sampler resource!", ILogger::ELL_ERROR, log.data());
							return false;
						}

						return ok;
					};

					if(!descriptorSetLayout)
					{
						creationParams.utilities->getLogger()->log("Provided descriptor set layout for IDescriptor::E_TYPE::%s is nullptr!", ILogger::ELL_ERROR, typeLiteral.data());
						return false;
					}

					const auto* redirect = &descriptorSetLayout->getDescriptorRedirect(descriptorType);

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

					const auto bindingCount = redirect->getBindingCount();

					bool ok = false;
					for (uint32_t i = 0u; i < bindingCount; ++i)
					{
						const auto rangeStorageIndex = IDescriptorSetLayoutBase::CBindingRedirect::storage_range_index_t(i);
						const auto binding = redirect->getBinding(rangeStorageIndex);
						const auto requestedBindingIx = descriptorType == IDescriptor::E_TYPE::ET_SAMPLED_IMAGE ? creationParams.resources.texturesInfo.bindingIx : creationParams.resources.samplersInfo.bindingIx;

						if (binding.data == requestedBindingIx)
						{
							const auto count = redirect->getCount(binding);

							if(!count)
							{
								creationParams.utilities->getLogger()->log("Provided descriptor set layout has IDescriptor::E_TYPE::%s binding for requested `creationParams.resources.%s` index but the binding resource count == 0u!", ILogger::ELL_ERROR, typeLiteral.data(), ixLiteral.data());
								return false;
							}

							if constexpr (descriptorType == IDescriptor::E_TYPE::ET_SAMPLED_IMAGE)
								creationParams.resources.texturesCount = count;
							else
								creationParams.resources.samplersCount = count;

							const auto stage = redirect->getStageFlags(binding);

							if(!stage.hasFlags(creationParams.resources.RequiredShaderStageFlags))
							{
								creationParams.utilities->getLogger()->log("Provided descriptor set layout has IDescriptor::E_TYPE::%s binding for requested `creationParams.resources.%s` index but doesn't meet stage flags requirements!", ILogger::ELL_ERROR, typeLiteral.data(), ixLiteral.data());
								return false;
							}

							const auto creation = redirect->getCreateFlags(rangeStorageIndex);

							if (!creation.hasFlags(descriptorType == IDescriptor::E_TYPE::ET_SAMPLED_IMAGE ? creationParams.resources.TexturesRequiredCreateFlags : creationParams.resources.SamplersRequiredCreateFlags))
							{
								creationParams.utilities->getLogger()->log("Provided descriptor set layout has IDescriptor::E_TYPE::%s binding for requested `creationParams.resources.%s` index but doesn't meet create flags requirements!", ILogger::ELL_ERROR, typeLiteral.data(), ixLiteral.data());
								return false;
							}

							ok = true;
							break;
						}
					}

					if (!ok)
					{
						creationParams.utilities->getLogger()->log("Provided descriptor set layout has no IDescriptor::E_TYPE::%s binding for requested `creationParams.resources.%s` index or it is invalid!", ILogger::ELL_ERROR, typeLiteral.data(), ixLiteral.data());
						return false;
					}

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
				creationParams.utilities->getLogger()->log(error, ILogger::ELL_ERROR);
				assert(false);
			}

		smart_refctd_ptr<IGPUCommandBuffer> transistentCMD;
		{
			using pool_flags_t = IGPUCommandPool::CREATE_FLAGS;

			smart_refctd_ptr<IGPUCommandPool> pool = creationParams.utilities->getLogicalDevice()->createCommandPool(creationParams.transfer->getFamilyIndex(), pool_flags_t::RESET_COMMAND_BUFFER_BIT|pool_flags_t::TRANSIENT_BIT);
			if (!pool)
			{
				creationParams.utilities->getLogger()->log("Could not create command pool!", ILogger::ELL_ERROR);
				assert(false);
			}
			
			if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &transistentCMD))
			{
				creationParams.utilities->getLogger()->log("Could not create transistent command buffer!", ILogger::ELL_ERROR);
				assert(false);
			}
		}

		// Dear ImGui context
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();

		createPipeline(creationParams);
		createMDIBuffer(creationParams);
		createFontAtlasTexture(transistentCMD.get(), creationParams);

		auto & io = ImGui::GetIO();
		io.BackendUsingLegacyKeyArrays = 0; // using AddKeyEvent() - it's new way of handling ImGUI events our backends supports

		m_cachedCreationParams = std::move(creationParams);
	}

	UI::~UI() = default;

	void UI::createMDIBuffer(SCreationParameters& m_cachedCreationParams)
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

		if (m_cachedCreationParams.streamingBuffer)
			m_mdi.compose = smart_refctd_ptr<typename compose_t>(m_cachedCreationParams.streamingBuffer);
		else
		{
			IGPUBuffer::SCreationParams mdiCreationParams = {};
			mdiCreationParams.usage = SMdiBuffer::RequiredUsageFlags;
			mdiCreationParams.size = mdiBufferDefaultSize;

			auto buffer = m_cachedCreationParams.utilities->getLogicalDevice()->createBuffer(std::move(mdiCreationParams));
			buffer->setObjectDebugName("MDI Upstream Buffer");

			auto memoryReqs = buffer->getMemoryReqs();
			memoryReqs.memoryTypeBits &= m_cachedCreationParams.utilities->getLogicalDevice()->getPhysicalDevice()->getUpStreamingMemoryTypeBits();

			auto allocation = m_cachedCreationParams.utilities->getLogicalDevice()->allocate(memoryReqs, buffer.get(), SMdiBuffer::RequiredAllocateFlags);
			{
				const bool allocated = allocation.isValid();
				assert(allocated);
			}
			auto memory = allocation.memory;

			if (!memory->map({ 0ull, memoryReqs.size }, getRequiredAccessFlags(memory->getMemoryPropertyFlags())))
				m_cachedCreationParams.utilities->getLogger()->log("Could not map device memory!", ILogger::ELL_ERROR);

			m_mdi.compose = make_smart_refctd_ptr<compose_t>(SBufferRange<IGPUBuffer>{0ull, mdiCreationParams.size, std::move(buffer)}, maxStreamingBufferAllocationAlignment, minStreamingBufferAllocationSize);
		}

		auto buffer = m_mdi.compose->getBuffer();
		auto binding = buffer->getBoundMemory();

		const auto validation = std::to_array
		({
			std::make_pair(buffer->getCreationParams().usage.hasFlags(SMdiBuffer::RequiredUsageFlags), "MDI buffer must be created with IBuffer::EUF_INDIRECT_BUFFER_BIT | IBuffer::EUF_INDEX_BUFFER_BIT | IBuffer::EUF_VERTEX_BUFFER_BIT | IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT enabled!"),
			std::make_pair(bool(buffer->getMemoryReqs().memoryTypeBits & m_cachedCreationParams.utilities->getLogicalDevice()->getPhysicalDevice()->getUpStreamingMemoryTypeBits()), "MDI buffer must have up-streaming memory type bits enabled!"),
			std::make_pair(binding.memory->getAllocateFlags().hasFlags(SMdiBuffer::RequiredAllocateFlags), "MDI buffer's memory must be allocated with IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT enabled!"),
			std::make_pair(binding.memory->isCurrentlyMapped(), "MDI buffer's memory must be mapped!"), // streaming buffer contructor already validates it, but cannot assume user won't unmap its own buffer for some reason (sorry if you have just hit it)
			std::make_pair(binding.memory->getCurrentMappingAccess().hasFlags(getRequiredAccessFlags(binding.memory->getMemoryPropertyFlags())), "MDI buffer's memory current mapping access flags don't meet requirements!")
		});

		for (const auto& [ok, error] : validation)
			if (!ok)
			{
				m_cachedCreationParams.utilities->getLogger()->log(error, ILogger::ELL_ERROR);
				assert(false);
			}
	}

	bool UI::render(IGPUCommandBuffer* const commandBuffer, ISemaphore::SWaitInfo waitInfo, const std::chrono::steady_clock::time_point waitPoint, const std::span<const VkRect2D> scissors)
	{
		if (!commandBuffer)
		{
			m_cachedCreationParams.utilities->getLogger()->log("Invalid command buffer!", ILogger::ELL_ERROR);
			return false;
		}

		if (commandBuffer->getState() != IGPUCommandBuffer::STATE::RECORDING)
		{
			m_cachedCreationParams.utilities->getLogger()->log("Command buffer is not in recording state!", ILogger::ELL_ERROR);
			return false;
		}

		{
			const auto info = commandBuffer->getCachedInheritanceInfo();
			const bool recordingSubpass = info.subpass != IGPURenderpass::SCreationParams::SSubpassDependency::External;

			if(!recordingSubpass)
			{
				m_cachedCreationParams.utilities->getLogger()->log("Command buffer is not recording a subpass!", ILogger::ELL_ERROR);
				return false;
			}
		}

		ImGui::Render(); // note it doesn't touch GPU or graphics API at all, its an internal ImGUI call to update & prepare the data for rendering so we can call GetDrawData()

		ImGuiIO& io = ImGui::GetIO();

		if (!io.Fonts->IsBuilt())
		{
			m_cachedCreationParams.utilities->getLogger()->log("Font atlas not built! It is generally built by the renderer backend. Missing call to renderer _NewFrame() function? e.g. ImGui_ImplOpenGL3_NewFrame().", ILogger::ELL_ERROR);
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
			const struct
			{
				ImVec2 off;			// (0,0) unless using multi-viewports
				ImVec2 scale;		// (1,1) unless using retina display which are often (2,2)
				ImVec2 framebuffer; // width, height

				// Project scissor/clipping rectangles into frame-buffer space
				ImVec4 getClipRectangle(const ImDrawCmd* cmd) const
				{
					assert(cmd);

					ImVec4 rectangle;
					rectangle.x = (cmd->ClipRect.x - off.x) * scale.x;
					rectangle.y = (cmd->ClipRect.y - off.y) * scale.y;
					rectangle.z = (cmd->ClipRect.z - off.x) * scale.x;
					rectangle.w = (cmd->ClipRect.w - off.y) * scale.y;

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
						VkRect2D scissor {};
						scissor.offset.x = static_cast<int32_t>(clipRectangle.x);
						scissor.offset.y = static_cast<int32_t>(clipRectangle.y);
						scissor.extent.width = static_cast<uint32_t>(clipRectangle.z - clipRectangle.x);
						scissor.extent.height = static_cast<uint32_t>(clipRectangle.w - clipRectangle.y);
						
						return scissor;
					}
				}
			} clip { .off = drawData->DisplayPos, .scale = drawData->FramebufferScale, .framebuffer = { frameBufferWidth, frameBufferHeight } };
			
			struct TRS
			{
				vector2df_SIMD scale;
				vector2df_SIMD translate;

				vector2df_SIMD toNDC(vector2df_SIMD in) const
				{
					return in * scale + translate;
				}
			};

			const TRS trs = [&]() 
			{
				TRS retV;

				retV.scale = vector2df_SIMD{ 2.0f / drawData->DisplaySize.x , 2.0f / drawData->DisplaySize.y };
				retV.translate = vector2df_SIMD { -1.0f, -1.0f } - vector2df_SIMD{ drawData->DisplayPos.x, drawData->DisplayPos.y } * trs.scale;

				return std::move(retV);
			}();

			const auto totalIndirectDrawCount = [&]()
			{
				uint32_t count = {};

				for (uint32_t i = 0; i < drawData->CmdListsCount; i++)
				{
					const ImDrawList* commandList = drawData->CmdLists[i];
					count += commandList->CmdBuffer.Size;
				}

				return count;
			}();

			//! we compose a struct of offsets for our mdi streaming iteration
			struct SAllocationParams
			{
				//! for a mdi streaming buffer iteration we compose offsets for which
				//! first slot is reserved for TightContent::INDIRECT_STRUCTURES,
				//! second for TightContent::ELEMENT_STRUCTURES
				//! and left slots are for indirect object buffers ([vertex, index] .. [vertex, index])
				std::vector<mdi_size_t> offsets;
				std::vector<bool> filled;
			} allocation;

			struct SMdiLimits
			{
				//! sum of allocation.sizes - all bytes which needs to be uploaded to cover all of totalIndirectDrawCount objects
				mdi_size_t totalByteSizeRequest = {},

				//! amount of total objects to draw
				totalIndirectDrawCount = {};

				//! corresponding allocation sizes & alignments for allocation.offsets - index shader between offset & size (we don't std::pair (and similar) those on purpose - need them separate in continous memory each to align nicely with our allocator api)
				std::vector<mdi_size_t> sizes, alignments;
			};

			const SMdiLimits mdiLimits = [&]()
			{
				SMdiLimits limits;

				limits.sizes.emplace_back() = {}; allocation.offsets.emplace_back() = InvalidAddress; limits.alignments.emplace_back() = alignof(VkDrawIndexedIndirectCommand); allocation.filled.emplace_back() = false;
				limits.sizes.emplace_back() = {}; allocation.offsets.emplace_back() = InvalidAddress; limits.alignments.emplace_back() = alignof(PerObjectData); allocation.filled.emplace_back() = false;

				for (uint32_t i = 0; i < drawData->CmdListsCount; i++)
				{
					const ImDrawList* commandList = drawData->CmdLists[i];
					limits.totalIndirectDrawCount += commandList->CmdBuffer.Size;

					allocation.offsets.emplace_back() = InvalidAddress; limits.sizes.emplace_back() = commandList->VtxBuffer.Size * sizeof(ImDrawVert); limits.alignments.emplace_back() = alignof(ImDrawVert); allocation.filled.emplace_back() = false;
					allocation.offsets.emplace_back() = InvalidAddress; limits.sizes.emplace_back() = commandList->IdxBuffer.Size * sizeof(ImDrawIdx); limits.alignments.emplace_back() = alignof(ImDrawIdx); allocation.filled.emplace_back() = false;
				}

				limits.totalByteSizeRequest += limits.sizes[(uint32_t)TightContent::INDIRECT_STRUCTURES] = (limits.totalIndirectDrawCount * sizeof(VkDrawIndexedIndirectCommand));
				limits.totalByteSizeRequest += limits.sizes[(uint32_t)TightContent::ELEMENT_STRUCTURES] = (limits.totalIndirectDrawCount * sizeof(PerObjectData));
				limits.totalByteSizeRequest += drawData->TotalVtxCount * sizeof(ImDrawVert);
				limits.totalByteSizeRequest += drawData->TotalIdxCount * sizeof(ImDrawIdx);
				
				assert([&]() -> bool // we should never hit it
				{
					return (allocation.offsets.size() == limits.sizes.size())
						&& (allocation.offsets.size() == allocation.filled.size())
						&& (std::reduce(std::begin(limits.sizes), std::end(limits.sizes)) == limits.totalByteSizeRequest) 
						&& (std::all_of(std::cbegin(allocation.offsets), std::cend(allocation.offsets), [](const auto& offset) { return offset == InvalidAddress; }));
				}()); // debug check only

				return limits;
			}();

			auto streamingBuffer = m_mdi.compose;
			{
				auto binding = streamingBuffer->getBuffer()->getBoundMemory();
				assert(binding.memory->isCurrentlyMapped());

				auto* const mdiData = reinterpret_cast<uint8_t*>(streamingBuffer->getBufferPointer());

				struct
				{
					mdi_size_t offset, size;
				} bigChunkRequestState;

				//! we will try to upload entrie MDI buffer with all available indirect data to our streaming buffer, but we cannot guarantee the allocation can be done in single request nor we allocate all totalIndirectDrawCount at all - we can hit timeout and it may appear not all of totalIndirectDrawCount will be uploaded then
				for (mdi_size_t uploadedSize = 0ull; uploadedSize < mdiLimits.totalByteSizeRequest;)
				{
					bigChunkRequestState.offset = InvalidAddress;
					bigChunkRequestState.size = streamingBuffer->max_size(); // request available block memory size which we can try to allocate

					constexpr auto StreamingAllocationCount = 1u;
					const size_t unallocatedSize = m_mdi.compose->multi_allocate(std::chrono::steady_clock::now() + std::chrono::microseconds(100u), StreamingAllocationCount, &bigChunkRequestState.offset, &bigChunkRequestState.size, &MdiMaxAlignment); //! (*) note we request single tight chunk of memory with fixed max alignment - big address space from which we fill try to suballocate to fill data 

					if (bigChunkRequestState.offset == InvalidAddress)
						continue; // failed? lets try again, TODO: should I here have my "blockMemoryFactor =* 0.5" and apply to bigChunkRequestState.size?
					else
					{
						constexpr auto AlignOffsetNeeded = 0u;
						SMdiBuffer::suballocator_traits_t::allocator_type fillSubAllocator(mdiData, bigChunkRequestState.offset, AlignOffsetNeeded, MdiMaxAlignment, bigChunkRequestState.size); //! (*) we create linear suballocator to fill the allocated chunk of memory (some of at least) 	
						SMdiBuffer::suballocator_traits_t::multi_alloc_addr(fillSubAllocator, allocation.offsets.size(), allocation.offsets.data(), mdiLimits.sizes.data(), mdiLimits.alignments.data()); //! (*) we suballocate memory regions from the allocated chunk with required alignments - multi request all with single traits call

						auto upload = [&]() -> size_t
						{
							size_t uploaded = {};

							// they are very small & negligible in size compared to buffers, but this small pool which we will conditionally fill on successfull object buffer suballocations is required to not complicate things (if we cannot allocate all mdiLimits.totalIndirectDrawCount object buffers then simply those coresponding structures wont be filled, we treat both components as arrays)
							const bool structuresSuballocated = allocation.offsets[(uint32_t)TightContent::INDIRECT_STRUCTURES] != InvalidAddress && allocation.offsets[(uint32_t)TightContent::ELEMENT_STRUCTURES] != InvalidAddress;

							if (structuresSuballocated) // note that suballocated only means we have valid address(es) we can work on, it doesn't mean we filled anything
							{
								auto* const indirectStructures = reinterpret_cast<VkDrawIndexedIndirectCommand*>(mdiData + allocation.offsets[(uint32_t)TightContent::INDIRECT_STRUCTURES]);
								auto* const elementStructures = reinterpret_cast<PerObjectData*>(mdiData + allocation.offsets[(uint32_t)TightContent::ELEMENT_STRUCTURES]);
								{
									if (!allocation.filled[(uint32_t)TightContent::INDIRECT_STRUCTURES])
									{
										uploaded += mdiLimits.sizes[(uint32_t)TightContent::INDIRECT_STRUCTURES];
										allocation.filled[(uint32_t)TightContent::INDIRECT_STRUCTURES] = true; // I make a assumption here since I can access them later - but I don't guarantee all of them will be present since we can fail other suballocations which are required for the struct, note that in reality we fill them below & conditionally
									}

									if (!allocation.filled[(uint32_t)TightContent::ELEMENT_STRUCTURES])
									{
										uploaded += mdiLimits.sizes[(uint32_t)TightContent::ELEMENT_STRUCTURES];
										allocation.filled[(uint32_t)TightContent::ELEMENT_STRUCTURES] = true; // I make a assumption here since I can access them later - but I don't guarantee all of them will be present since we can fail other suballocations which are required for the struct, note that in reality we fill them below & conditionally
									}
								}

								uint32_t drawID = {};
								for (uint32_t i = 0; i < drawData->CmdListsCount; i++)
								{
									const auto* commandList = drawData->CmdLists[i];
									const auto& [vertexBuffer, indexBuffer] = std::make_tuple(commandList->VtxBuffer, commandList->IdxBuffer);
									const auto [vtxAllocationIx, idxAllocationIx] = std::make_tuple((uint32_t)TightContent::COUNT + 2u * i + 0u, (uint32_t)TightContent::COUNT + 2u * i + 1u); // look at SAllocationParams::offsets if you dont remember why like this

									auto fillBuffer = [&](const auto* in, const uint32_t allocationIx)
									{
										auto& offset = allocation.offsets[allocationIx]; const auto& bytesToFill = mdiLimits.sizes[allocationIx];

										if (offset == InvalidAddress)
											return false;
										else
											if (!allocation.filled[allocationIx])
											{
												::memcpy(mdiData + offset, in, bytesToFill);
												uploaded += bytesToFill;
												allocation.filled[allocationIx] = true;
											}

										return true;
									};

									// we consider buffers valid if we suballocated them (under the hood filled)
									const auto buffersSuballocated = fillBuffer(vertexBuffer.Data, vtxAllocationIx) && fillBuffer(indexBuffer.Data, idxAllocationIx);
									const auto [vtxGlobalObjectOffset, idxGlobalObjectOffset] = buffersSuballocated ? std::make_tuple(allocation.offsets[vtxAllocationIx] / sizeof(ImDrawVert), allocation.offsets[idxAllocationIx] / sizeof(ImDrawIdx)) : std::make_tuple((size_t)0u, (size_t)0u);
							
									for (uint32_t j = 0; j < commandList->CmdBuffer.Size; j++)
									{
										const auto* cmd = &commandList->CmdBuffer[j];
										auto* indirect = indirectStructures + drawID;
										auto* element = elementStructures + drawID;

										// we will make a trick to keep indirect & element structs in the mdi iteration but explicitly execute dummy null invocation if we don't have vertex or index buffer for the struct (suballocation failed for any of those 2 buffers). 
										// TODO: we could make the current structs pool "dynamic" in size and treat as simple stack instead (trying it first to make things easier)
										indirect->indexCount = buffersSuballocated ? cmd->ElemCount /* valid invocation */ : 0u /* null invocation */;

										indirect->firstInstance = drawID; // we use base instance as draw ID
										indirect->instanceCount = 1u;

										// starting to wonder, for some reason imgui decided to keep single vertex & index shared between cmds within cmd list
										// but maybe we should cut current [vertexBuffer, indexBuffer] with respect to cmd->IdxOffset & cmd->VtxOffset (therefore we could have even smaller alloc requests, now a few structs can point to the same buffer but with different offsets [indirect])
										// though not sure if I don't double some data then
										indirect->vertexOffset = vtxGlobalObjectOffset + cmd->VtxOffset; // safe to assume due to indirect->indexCount depending on buffersSuballocated
										indirect->firstIndex = idxGlobalObjectOffset + cmd->IdxOffset; // safe to assume due to indirect->indexCount depending on buffersSuballocated

										const auto clipRectangle = clip.getClipRectangle(cmd);
										const auto scissor = clip.getScissor(clipRectangle);

										auto packSnorm16 = [](float ndc) -> int16_t
										{
											return std::round<int16_t>(std::clamp(ndc, -1.0f, 1.0f) * 32767.0f); // TODO: ok encodePixels<EF_R16_SNORM, double>(void* _pix, const double* _input) but iirc we have issues with our encode/decode utils
										};

										const auto vMin = trs.toNDC(vector2df_SIMD(scissor.offset.x, scissor.offset.y));
										const auto vMax = trs.toNDC(vector2df_SIMD(scissor.offset.x + scissor.extent.width, scissor.offset.y + scissor.extent.height));

										struct snorm16_t2_packed
										{
											int16_t x, y;
										};

										reinterpret_cast<snorm16_t2_packed&>(element->aabbMin) = { .x = packSnorm16(vMin.x), .y = packSnorm16(vMin.y) };
										reinterpret_cast<snorm16_t2_packed&>(element->aabbMax) = { .x = packSnorm16(vMax.x), .y = packSnorm16(vMax.y) };

										element->texId = cmd->TextureId.textureID;
										element->samplerIx = cmd->TextureId.samplerIx;

										++drawID;
									}
								}
							}

							return uploaded;
						};

						uploadedSize += upload();
					}
					streamingBuffer->multi_deallocate(StreamingAllocationCount, &bigChunkRequestState.offset, &bigChunkRequestState.size, waitInfo); //! (*) block allocated, we just latch offsets deallocation to keep it alive as long as required
				
					// we let to run it at least once
					const bool timeout = std::chrono::steady_clock::now() >= waitPoint;

					if (timeout)
					{
						if (uploadedSize >= mdiLimits.totalByteSizeRequest)
							break; // must be lucky to hit it or on debug

						streamingBuffer->cull_frees();
						return false;
					}
				}
			}

			auto mdiBuffer = smart_refctd_ptr<IGPUBuffer>(m_mdi.compose->getBuffer());
			const auto offset = mdiBuffer->getBoundMemory().offset;
			{
				const SBufferBinding<const IGPUBuffer> binding =
				{
					.offset = 0u,
					.buffer = smart_refctd_ptr(mdiBuffer)
				};

				if (!commandBuffer->bindIndexBuffer(binding, sizeof(ImDrawIdx) == 2 ? EIT_16BIT : EIT_32BIT))
				{
					m_cachedCreationParams.utilities->getLogger()->log("Could not bind index buffer!", ILogger::ELL_ERROR);
					assert(false);
				}
			}

			{
				const SBufferBinding<const IGPUBuffer> bindings[] =
				{{
					.offset = 0u,
					.buffer = smart_refctd_ptr(mdiBuffer)
				}};

				if(!commandBuffer->bindVertexBuffers(0, 1, bindings))
				{
					m_cachedCreationParams.utilities->getLogger()->log("Could not bind vertex buffer!", ILogger::ELL_ERROR);
					assert(false);
				}
			}

			SViewport const viewport
			{
				.x = 0,
				.y = 0,
				.width = frameBufferWidth,
				.height = frameBufferHeight,
				.minDepth = 0.0f,
				.maxDepth = 1.0f,
			};

			commandBuffer->setViewport(0, 1, &viewport);
			{
				if (scissors.empty())
				{
					VkRect2D defaultScisors[] = { {.offset = {(int32_t)viewport.x, (int32_t)viewport.y}, .extent = {(uint32_t)viewport.width, (uint32_t)viewport.height}} };
					commandBuffer->setScissor(defaultScisors); // cover whole viewport (dynamic scissors must be set only to not throw validation errors)
				}
				else
					commandBuffer->setScissor(scissors);
			}
			
			/*
				Setup scale and translation, our visible imgui space lies from draw_data->DisplayPps (top left) to 
				draw_data->DisplayPos+data_data->DisplaySize (bottom right). DisplayPos is (0,0) for single viewport apps.
			*/

			{
				PushConstants constants
				{
					.elementBDA = { mdiBuffer->getDeviceAddress() + allocation.offsets[(uint32_t)TightContent::ELEMENT_STRUCTURES]},
					.elementCount = { mdiLimits.totalIndirectDrawCount },
					.scale = { trs.scale[0u], trs.scale[1u] },
					.translate = { trs.translate[0u], trs.translate[1u] },
					.viewport = { viewport.x, viewport.y, viewport.width, viewport.height }
				};

				commandBuffer->pushConstants(m_pipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_VERTEX | IShader::E_SHADER_STAGE::ESS_FRAGMENT, 0u, sizeof(constants), &constants);
			}

			const SBufferBinding<const IGPUBuffer> binding =
			{
				.offset = allocation.offsets[(uint32_t)TightContent::INDIRECT_STRUCTURES],
				.buffer = smart_refctd_ptr(mdiBuffer)
			};

			commandBuffer->drawIndexedIndirect(binding, mdiLimits.totalIndirectDrawCount, sizeof(VkDrawIndexedIndirectCommand));
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