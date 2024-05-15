#include <iostream>
#include <map>
#include <ranges>
#include <vector>
#include <utility>

// internal & nabla
#include "nbl/system/IApplicationFramework.h"
#include "nbl/system/CStdoutLogger.h"
#include "nbl/ext/ImGui/ImGui.h"
#include "shaders/common.hlsl"
#include "ext/imgui/spirv/builtin/builtinResources.h"
#include "ext/imgui/spirv/builtin/CArchive.h"

// 3rdparty
#include "imgui/imgui.h"
#include "imgui/misc/cpp/imgui_stdlib.h"

using namespace nbl::video;
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::ui;

namespace nbl::ext::imgui
{
	smart_refctd_ptr<IGPUDescriptorSetLayout> UI::CreateDescriptorSetLayout()
	{
		nbl::video::IGPUDescriptorSetLayout::SBinding bindings[] = {
			{
				.binding = 0,
				.type = asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
				.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags = IShader::ESS_FRAGMENT,
				.count = 1,
				.samplers = nullptr // TODO: m_fontSampler?
			}
		};

		return m_device->createDescriptorSetLayout(bindings);
	}

	void UI::CreatePipeline(video::IGPURenderpass* renderpass, IGPUPipelineCache* pipelineCache)
	{
		// Constants: we are using 'vec2 offset' and 'vec2 scale' instead of a full 3d projection matrix
		SPushConstantRange pushConstantRanges[] = {
			{
				.stageFlags = IShader::ESS_VERTEX,
				.offset = 0,
				.size = sizeof(PushConstants)
			}
		};

		auto descriptorSetLayout = CreateDescriptorSetLayout();
		m_gpuDescriptorSet = m_descriptorPool->createDescriptorSet(descriptorSetLayout); 
		assert(m_gpuDescriptorSet);

		auto pipelineLayout = m_device->createPipelineLayout(pushConstantRanges, std::move(descriptorSetLayout));

		struct
		{
			core::smart_refctd_ptr<video::IGPUShader> vertex, fragment;
		} shaders;

		{
			struct
			{
				const system::SBuiltinFile vertex = ::ext::imgui::spirv::builtin::get_resource<"ext/imgui/spirv/vertex.spv">();
				const system::SBuiltinFile fragment = ::ext::imgui::spirv::builtin::get_resource<"ext/imgui/spirv/fragment.spv">();
			} spirv;

			auto createShader = [&](const system::SBuiltinFile& in, asset::IShader::E_SHADER_STAGE stage) -> core::smart_refctd_ptr<video::IGPUShader>
			{
				const auto buffer = core::make_smart_refctd_ptr<asset::CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t>, true> >(in.size, /*this cast is awful but my custom buffer won't free it's memory*/ (void*)in.contents, core::adopt_memory); // no copy
				const auto shader = make_smart_refctd_ptr<ICPUShader>(core::smart_refctd_ptr(buffer), stage, IShader::E_CONTENT_TYPE::ECT_SPIRV, "");
				
				return m_device->createShader(shader.get());
			};

			shaders.vertex = createShader(spirv.vertex, IShader::ESS_VERTEX);
			shaders.fragment = createShader(spirv.fragment, IShader::ESS_FRAGMENT);
		}
	
		SVertexInputParams vertexInputParams{};
		{
			vertexInputParams.enabledBindingFlags = 0b1u;
			vertexInputParams.enabledAttribFlags = 0b111u;

			vertexInputParams.bindings[0].inputRate = asset::SVertexInputBindingParams::EVIR_PER_VERTEX;
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
			param.srcColorFactor = EBF_SRC_ALPHA;//VK_BLEND_FACTOR_SRC_ALPHA;
			param.dstColorFactor = EBF_ONE_MINUS_SRC_ALPHA;//VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
			param.colorBlendOp = EBO_ADD;//VK_BLEND_OP_ADD;
			param.srcAlphaFactor = EBF_ONE_MINUS_SRC_ALPHA;//VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
			param.dstAlphaFactor = EBF_ZERO;//VK_BLEND_FACTOR_ZERO;
			param.alphaBlendOp = EBO_ADD;//VK_BLEND_OP_ADD;
			param.colorWriteMask = (1u << 0u) | (1u << 1u) | (1u << 2u) | (1u << 3u);//VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		}
	
		SRasterizationParams rasterizationParams{};
		{
			rasterizationParams.faceCullingMode = EFCM_NONE;
			rasterizationParams.depthWriteEnable = false; // TODO: check if it disabled depth test?
			rasterizationParams.depthBoundsTestEnable = false;
			// rasterizationParams.stencilTestEnable = false; // TODO: check if stencil test is disabled
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
				auto& param = params[0];
				param.layout = pipelineLayout.get();
				param.shaders = specs;
				param.renderpass = renderpass;
				param.cached = { .vertexInput = vertexInputParams, .primitiveAssembly = primitiveAssemblyParams, .rasterization = rasterizationParams, .blend = blendParams, .subpassIx = 0u }; // TODO: check "subpassIx"
			};
			
			if (!m_device->createGraphicsPipelines(pipelineCache, params, &pipeline))
			{
				logger->log("Could not create pipeline!", system::ILogger::ELL_ERROR);
				assert(false);
			}
		}
	}

	void UI::CreateFontTexture(video::IGPUCommandBuffer* cmdBuffer, video::IQueue* transfer)
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

		uint8_t* pixels = nullptr;
		int32_t width, height;
		io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
		assert(pixels != nullptr);
		assert(width > 0);
		assert(height > 0);
		const size_t componentsCount = 4, image_size = width * height * componentsCount * sizeof(uint8_t);
		
		_NBL_STATIC_INLINE_CONSTEXPR auto NBL_FORMAT_FONT = EF_R8G8B8A8_UNORM;
		const auto buffer = core::make_smart_refctd_ptr< asset::CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t>, true> >(image_size, pixels, core::adopt_memory);
		
		IGPUImage::SCreationParams params;
		params.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);
		params.type = IImage::ET_2D;
		params.format = NBL_FORMAT_FONT;
		params.extent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1u };
		params.mipLevels = 1;
		params.arrayLayers = 1u;
		params.samples = IImage::ESCF_1_BIT;
		params.usage |= IGPUImage::EUF_TRANSFER_DST_BIT | IGPUImage::EUF_SAMPLED_BIT | IGPUImage::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT;

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
			region->imageOffset = { 0, 0, 0 };
			region->imageExtent = { params.extent.width, params.extent.height, 1u };
		}

		auto image = m_device->createImage(std::move(params));

		if (!image)
		{
			logger->log("Could not create font image!", system::ILogger::ELL_ERROR);
			assert(false);
		}

		if (!m_device->allocate(image->getMemoryReqs(), image.get()).isValid())
		{
			logger->log("Could not allocate memory for font image!", system::ILogger::ELL_ERROR);
			assert(false);
		}
		
		image->setObjectDebugName("Nabla IMGUI extension Font Image");
		{
			IQueue::SSubmitInfo::SCommandBufferInfo cmdInfo = { cmdBuffer };
			SIntendedSubmitInfo sInfo;

			auto scratchSemaphore = m_device->createSemaphore(0);
			if (!scratchSemaphore)
			{
				logger->log("Could not create scratch semaphore", system::ILogger::ELL_ERROR);
				assert(false);
			}
			scratchSemaphore->setObjectDebugName("Nabla IMGUI extension Scratch Semaphore");

			sInfo.queue = transfer;
			sInfo.waitSemaphores = {};
			sInfo.commandBuffers = { &cmdInfo, 1 };
			sInfo.scratchSemaphore = // TODO: do I really need it?
			{
				.semaphore = scratchSemaphore.get(),
				.value = 0,
				.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
			};

			const SMemoryBarrier toTransferBarrier = {
				.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
				.dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT
			};

			cmdBuffer->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> barriers[] = 
			{ 
				{
					.barrier = { .dep = toTransferBarrier },
					.image = image.get(),
					.subresourceRange = regions.subresource,
					.newLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL
				} 
			};

			cmdBuffer->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = barriers });

			utilities->updateImageViaStagingBufferAutoSubmit(sInfo, buffer->getPointer(), NBL_FORMAT_FONT, image.get(), IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL, regions.range);
		}
		 
		{
			IGPUImageView::SCreationParams params;
			params.format = image->getCreationParameters().format;
			params.viewType = IImageView<IGPUImage>::ET_2D;
			params.subresourceRange = regions.subresource;
			params.image = core::smart_refctd_ptr(image);

			m_fontTexture = m_device->createImageView(std::move(params));
		}
	}

	void prepareKeyMapForDesktop()
	{
		ImGuiIO& io = ImGui::GetIO();
		// TODO:
		// Keyboard mapping. ImGui will use those indices to peek into the io.KeysDown[] array.
		//io.KeyMap[ImGuiKey_Tab] = MSDL::SDL_SCANCODE_TAB;
		//io.KeyMap[ImGuiKey_LeftArrow] = MSDL::SDL_SCANCODE_LEFT;
		//io.KeyMap[ImGuiKey_RightArrow] = MSDL::SDL_SCANCODE_RIGHT;
		//io.KeyMap[ImGuiKey_UpArrow] = MSDL::SDL_SCANCODE_UP;
		//io.KeyMap[ImGuiKey_DownArrow] = MSDL::SDL_SCANCODE_DOWN;
		//io.KeyMap[ImGuiKey_PageUp] = MSDL::SDL_SCANCODE_PAGEUP;
		//io.KeyMap[ImGuiKey_PageDown] = MSDL::SDL_SCANCODE_PAGEDOWN;
		//io.KeyMap[ImGuiKey_Home] = MSDL::SDL_SCANCODE_HOME;
		//io.KeyMap[ImGuiKey_End] = MSDL::SDL_SCANCODE_END;
		//io.KeyMap[ImGuiKey_Insert] = MSDL::SDL_SCANCODE_INSERT;
		//io.KeyMap[ImGuiKey_Delete] = MSDL::SDL_SCANCODE_DELETE;
		//io.KeyMap[ImGuiKey_Backspace] = MSDL::SDL_SCANCODE_BACKSPACE;
		//io.KeyMap[ImGuiKey_Space] = MSDL::SDL_SCANCODE_SPACE;
		//io.KeyMap[ImGuiKey_Enter] = MSDL::SDL_SCANCODE_RETURN;
		//io.KeyMap[ImGuiKey_Escape] = MSDL::SDL_SCANCODE_ESCAPE;
		//io.KeyMap[ImGuiKey_KeyPadEnter] = MSDL::SDL_SCANCODE_KP_ENTER;
		//io.KeyMap[ImGuiKey_A] = MSDL::SDL_SCANCODE_A;
		//io.KeyMap[ImGuiKey_C] = MSDL::SDL_SCANCODE_C;
		//io.KeyMap[ImGuiKey_V] = MSDL::SDL_SCANCODE_V;
		//io.KeyMap[ImGuiKey_X] = MSDL::SDL_SCANCODE_X;
		//io.KeyMap[ImGuiKey_Y] = MSDL::SDL_SCANCODE_Y;
		//io.KeyMap[ImGuiKey_Z] = MSDL::SDL_SCANCODE_Z;
	}

	static void adjustGlobalFontScale()
	{
		ImGuiIO& io = ImGui::GetIO();
		io.FontGlobalScale = 1.0f;
	}

	void UI::UpdateDescriptorSets()
	{
		IGPUDescriptorSet::SDescriptorInfo info;
		{
			info.desc = m_fontTexture;
			info.info.image.sampler = m_fontSampler;
			info.info.image.imageLayout = nbl::asset::IImage::LAYOUT::READ_ONLY_OPTIMAL;
		}

		IGPUDescriptorSet::SWriteDescriptorSet writeDescriptorSet{};
		writeDescriptorSet.dstSet = m_gpuDescriptorSet.get();
		writeDescriptorSet.binding = 0;
		writeDescriptorSet.arrayElement = 0;
		writeDescriptorSet.count = 1;
		writeDescriptorSet.info = &info;

		m_device->updateDescriptorSets(1, &writeDescriptorSet, 0, nullptr);
	}

	void UI::CreateFontSampler()
	{
		// TODO: Recheck this settings
		IGPUSampler::SParams params{};
		params.MinLod = -1000;
		params.MaxLod = 1000;
		params.AnisotropicFilter = 1.0f;
		params.TextureWrapU = ISampler::ETC_REPEAT;
		params.TextureWrapV = ISampler::ETC_REPEAT;
		params.TextureWrapW = ISampler::ETC_REPEAT;

		m_fontSampler = m_device->createSampler(params);
	}

	void UI::CreateDescriptorPool()
	{
		static constexpr int TotalSetCount = 1;
		IDescriptorPool::SCreateInfo createInfo = {};
		createInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER)] = TotalSetCount;
		createInfo.maxSets = 1;
		createInfo.flags = IDescriptorPool::E_CREATE_FLAGS::ECF_NONE;

		m_descriptorPool = m_device->createDescriptorPool(std::move(createInfo));
		assert(m_descriptorPool);
	}
	
	void UI::HandleMouseEvents(
		float const mousePosX, 
		float const mousePosY,
		size_t const mouseEventsCount,
		SMouseEvent const * mouseEvents
	) const
	{
		auto& io = ImGui::GetIO();

		io.MousePos.x = mousePosX - m_window->getX();
		io.MousePos.y = mousePosY - m_window->getY();

		for (size_t i = 0; i < mouseEventsCount; ++i)
		{
			auto const & event = mouseEvents[i];
			if(event.type == SMouseEvent::EET_CLICK)
			{
				int buttonIndex = -1;
				if (event.clickEvent.mouseButton == EMB_LEFT_BUTTON) 
				{
					buttonIndex = 0;
				} else if (event.clickEvent.mouseButton == EMB_RIGHT_BUTTON)
				{
					buttonIndex = 1;
				} else if (event.clickEvent.mouseButton == EMB_MIDDLE_BUTTON)
				{
					buttonIndex = 2;
				}

				if (buttonIndex == -1)
				{
					assert(false);
					continue;
				}

				if(event.clickEvent.action == SMouseEvent::SClickEvent::EA_PRESSED) {
					io.MouseDown[buttonIndex] = true;
				} else if (event.clickEvent.action == SMouseEvent::SClickEvent::EA_RELEASED) {
					io.MouseDown[buttonIndex] = false;
				}
			}
		}
	}

	UI::UI(smart_refctd_ptr<ILogicalDevice> device, int maxFramesInFlight, video::IGPURenderpass* renderpass, IGPUPipelineCache* pipelineCache, smart_refctd_ptr<IWindow> window)
		: m_device(core::smart_refctd_ptr(device)), m_window(core::smart_refctd_ptr(window))
	{
		createSystem();
		struct
		{
			struct
			{
				uint8_t transfer, graphics;
			} id;
		} families;

		const nbl::video::IPhysicalDevice* pDevice = device->getPhysicalDevice();
		ILogicalDevice::SCreationParams params = {};

		auto properties = pDevice->getQueueFamilyProperties();

		auto requestFamilyQueueId = [&](IQueue::FAMILY_FLAGS requried, std::string_view onError)
		{
			uint8_t index = 0;
			for (const auto& fProperty : properties)
			{
				if (fProperty.queueFlags.hasFlags(requried))
				{
					++params.queueParams[index].count;

					return index;
				}
				++index;
			}

			logger->log(onError.data(), system::ILogger::ELL_ERROR);
			assert(false);
		};

		// get & validate families' capabilities
		families.id.transfer = requestFamilyQueueId(IQueue::FAMILY_FLAGS::TRANSFER_BIT, "Could not find any queue with TRANSFER_BIT enabled!");
		families.id.graphics = requestFamilyQueueId(IQueue::FAMILY_FLAGS::GRAPHICS_BIT, "Could not find any queue with GRAPHICS_BIT enabled!");

		// allocate temporary command buffer
		auto* tQueue = device->getThreadSafeQueue(families.id.transfer, 0);

		if (!tQueue)
		{
			logger->log("Could not get queue!", system::ILogger::ELL_ERROR);
			assert(false);
		}

		smart_refctd_ptr<nbl::video::IGPUCommandBuffer> transistentCMD;
		{
			smart_refctd_ptr<nbl::video::IGPUCommandPool> pool = device->createCommandPool(families.id.transfer, IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
			if (!pool)
			{
				logger->log("Could not create command pool!", system::ILogger::ELL_ERROR);
				assert(false);
			}
			
			
			if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &transistentCMD))
			{
				logger->log("Could not create transistent command buffer!", system::ILogger::ELL_ERROR);
				assert(false);
			}
		}

		tQueue->startCapture();
		{
			// Setup Dear ImGui context
			IMGUI_CHECKVERSION();
			ImGui::CreateContext();

			CreateFontSampler();
			CreateDescriptorPool();
			CreateDescriptorSetLayout();
			CreatePipeline(renderpass, pipelineCache);
			CreateFontTexture(transistentCMD.get(), tQueue);
			prepareKeyMapForDesktop();
			adjustGlobalFontScale();
			UpdateDescriptorSets();
		}
		tQueue->endCapture();

		auto & io = ImGui::GetIO();
		io.DisplaySize = ImVec2(m_window->getWidth(), m_window->getHeight());
		io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);

		m_vertexBuffers.resize(maxFramesInFlight);
		m_indexBuffers.resize(maxFramesInFlight);
	}

	UI::~UI() = default;

	void UI::createSystem()
	{
		system = system::IApplicationFramework::createSystem();
		const auto logLevel = nbl::core::bitflag(system::ILogger::ELL_ALL);
		logger = core::make_smart_refctd_ptr<system::CColoredStdoutLoggerWin32>(logLevel);
		auto archive = core::make_smart_refctd_ptr<::ext::imgui::spirv::builtin::CArchive>(smart_refctd_ptr(logger));
		
		system->mount(core::smart_refctd_ptr(archive));

		utilities = make_smart_refctd_ptr<video::IUtilities>(core::smart_refctd_ptr(m_device), core::smart_refctd_ptr(logger));

		if (!utilities)
		{
			logger->log("Failed to create nbl::video::IUtilities!", system::ILogger::ELL_ERROR);
			assert(false);
		}
	}

	//-------------------------------------------------------------------------------------------------
	// TODO: Handle mouse cursor for nabla
	//static void UpdateMouseCursor()
	//{
		//auto & io = ImGui::GetIO();

		//if (io.ConfigFlags & ImGuiConfigFlags_NoMouseCursorChange)
		//{
		//    return;
		//}
		//ImGuiMouseCursor imgui_cursor = ImGui::GetMouseCursor();
		//if (io.MouseDrawCursor || imgui_cursor == ImGuiMouseCursor_None)
		//{
		//    // Hide OS mouse cursor if imgui is drawing it or if it wants no cursor
		//    MSDL::SDL_ShowCursor(MSDL::SDL_FALSE);
		//}
		//else
		//{
		//    // Show OS mouse cursor
		//    MSDL::SDL_SetCursor(mouseCursors[imgui_cursor] ? mouseCursors[imgui_cursor] : mouseCursors[ImGuiMouseCursor_Arrow]);
		//    MSDL::SDL_ShowCursor(MSDL::SDL_TRUE);
		//}
	//}

	bool UI::Render(IGPUCommandBuffer* commandBuffer, int const frameIndex)
	{
		ImGuiIO& io = ImGui::GetIO();

		if (!io.Fonts->IsBuilt())
		{
			logger->log("Font atlas not built! It is generally built by the renderer backend. Missing call to renderer _NewFrame() function? e.g. ImGui_ImplOpenGL3_NewFrame().", system::ILogger::ELL_ERROR);
			assert(false);
		}

		auto* rawPipeline = pipeline.get();
		commandBuffer->bindGraphicsPipeline(rawPipeline);
		commandBuffer->bindDescriptorSets(EPBP_GRAPHICS, rawPipeline->getLayout(), 0, 1, &m_gpuDescriptorSet.get());
		
		auto const* drawData = ImGui::GetDrawData();

		if (!drawData)
			return false;
		
		// Avoid rendering when minimized, scale coordinates for retina displays (screen coordinates != framebuffer coordinates)
		float const frameBufferWidth = drawData->DisplaySize.x * drawData->FramebufferScale.x;
		float const frameBufferHeight = drawData->DisplaySize.y * drawData->FramebufferScale.y;
		if (frameBufferWidth > 0 && frameBufferHeight > 0 && drawData->TotalVtxCount > 0)
		{
			// Create or resize the vertex/index buffers
			size_t const vertexSize = drawData->TotalVtxCount * sizeof(ImDrawVert);
			size_t const indexSize = drawData->TotalIdxCount * sizeof(ImDrawIdx);

			IGPUBuffer::SCreationParams vertexCreationParams = {};
			vertexCreationParams.usage = nbl::core::bitflag(nbl::asset::IBuffer::EUF_VERTEX_BUFFER_BIT) | nbl::asset::IBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF;
			vertexCreationParams.size = vertexSize;

			auto & vertexBuffer = m_vertexBuffers[frameIndex];

			if (static_cast<bool>(vertexBuffer) == false || vertexBuffer->getSize() < vertexSize)
			{
				vertexBuffer = m_device->createBuffer(std::move(vertexCreationParams));

				video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = vertexBuffer->getMemoryReqs();
				memReq.memoryTypeBits &= m_device->getPhysicalDevice()->getUpStreamingMemoryTypeBits();
				auto memOffset = m_device->allocate(memReq, vertexBuffer.get());
				assert(memOffset.isValid());
			}

			IGPUBuffer::SCreationParams indexCreationParams = {};
			indexCreationParams.usage = nbl::core::bitflag(nbl::asset::IBuffer::EUF_VERTEX_BUFFER_BIT) | nbl::asset::IBuffer::EUF_INDEX_BUFFER_BIT
				| nbl::asset::IBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF;
			indexCreationParams.size = indexSize;

			auto & indexBuffer = m_indexBuffers[frameIndex];

			if (static_cast<bool>(indexBuffer) == false || indexBuffer->getSize() < indexSize)
			{
				indexBuffer = m_device->createBuffer(std::move(indexCreationParams));

				video::IDeviceMemoryBacked::SDeviceMemoryRequirements memReq = indexBuffer->getMemoryReqs();
				memReq.memoryTypeBits &= m_device->getPhysicalDevice()->getUpStreamingMemoryTypeBits();
				auto memOffset = m_device->allocate(memReq, indexBuffer.get());
				assert(memOffset.isValid());
			}

			{
				auto vBinding = vertexBuffer->getBoundMemory();
				auto iBinding = indexBuffer->getBoundMemory();

				{
					if (!vBinding.memory->map({ 0ull, vBinding.memory->getAllocationSize() }, IDeviceMemoryAllocation::EMCAF_READ))
						logger->log("Could not map device memory for vertex buffer data!", system::ILogger::ELL_WARNING);

					assert(vBinding.memory->isCurrentlyMapped());
				}

				{
					if (!iBinding.memory->map({ 0ull, iBinding.memory->getAllocationSize() }, IDeviceMemoryAllocation::EMCAF_READ))
						logger->log("Could not map device memory for index buffer data!", system::ILogger::ELL_WARNING);
				
					assert(iBinding.memory->isCurrentlyMapped());
				}

				auto* vertex_ptr = static_cast<ImDrawVert*>(vBinding.memory->getMappedPointer());
				auto* index_ptr = static_cast<ImDrawIdx*>(iBinding.memory->getMappedPointer());

				for (int n = 0; n < drawData->CmdListsCount; n++)
				{
					const ImDrawList* cmd = drawData->CmdLists[n];
					::memcpy(vertex_ptr, cmd->VtxBuffer.Data, cmd->VtxBuffer.Size * sizeof(ImDrawVert));
					::memcpy(index_ptr, cmd->IdxBuffer.Data, cmd->IdxBuffer.Size * sizeof(ImDrawIdx));
					vertex_ptr += cmd->VtxBuffer.Size;
					index_ptr += cmd->IdxBuffer.Size;
				}

				vBinding.memory->unmap();
				iBinding.memory->unmap();
			}

			{
				const asset::SBufferBinding<const video::IGPUBuffer> binding =
				{
					.offset = 0,
					.buffer = core::smart_refctd_ptr(indexBuffer)
				};

				if (!commandBuffer->bindIndexBuffer(binding, sizeof(ImDrawIdx) == 2 ? EIT_16BIT : EIT_32BIT))
				{
					logger->log("Could not bind index buffer!", system::ILogger::ELL_ERROR);
					assert(false);
				}
			}

			{
				const asset::SBufferBinding<const video::IGPUBuffer> bindings[] =
				{
					{
						.offset = 0,
						.buffer = core::smart_refctd_ptr(vertexBuffer)
					}
				};

				if(!commandBuffer->bindVertexBuffers(0, 1, bindings))
				{
					logger->log("Could not bind vertex buffer!", system::ILogger::ELL_ERROR);
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

			// Setup scale and translation:
			// Our visible imgui space lies from draw_data->DisplayPps (top left) to draw_data->DisplayPos+data_data->DisplaySize (bottom right). DisplayPos is (0,0) for single viewport apps.
			{
				PushConstants constants{};
				constants.scale[0] = 2.0f / drawData->DisplaySize.x;
				constants.scale[1] = 2.0f / drawData->DisplaySize.y;
				constants.translate[0] = -1.0f - drawData->DisplayPos.x * constants.scale[0];
				constants.translate[1] = -1.0f - drawData->DisplayPos.y * constants.scale[1];

				commandBuffer->pushConstants(pipeline->getLayout(), IShader::ESS_VERTEX, 0, sizeof(constants), &constants);
			}

			// Will project scissor/clipping rectangles into frame-buffer space
			ImVec2 const clip_off = drawData->DisplayPos;         // (0,0) unless using multi-viewports
			ImVec2 const clip_scale = drawData->FramebufferScale; // (1,1) unless using retina display which are often (2,2)

			// Render command lists
			// (Because we merged all buffers into a single one, we maintain our own offset into them)
			int global_vtx_offset = 0;
			int global_idx_offset = 0;
			for (int n = 0; n < drawData->CmdListsCount; n++)
			{
				const ImDrawList* cmd_list = drawData->CmdLists[n];
				for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; cmd_i++)
				{
					const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[cmd_i];

					// Project scissor/clipping rectangles into frame-buffer space
					ImVec4 clip_rect;
					clip_rect.x = (pcmd->ClipRect.x - clip_off.x) * clip_scale.x;
					clip_rect.y = (pcmd->ClipRect.y - clip_off.y) * clip_scale.y;
					clip_rect.z = (pcmd->ClipRect.z - clip_off.x) * clip_scale.x;
					clip_rect.w = (pcmd->ClipRect.w - clip_off.y) * clip_scale.y;

					if (clip_rect.x < frameBufferWidth && clip_rect.y < frameBufferHeight && clip_rect.z >= 0.0f && clip_rect.w >= 0.0f)
					{
						// Negative offsets are illegal for vkCmdSetScissor
						if (clip_rect.x < 0.0f)
							clip_rect.x = 0.0f;
						if (clip_rect.y < 0.0f)
							clip_rect.y = 0.0f;

						{// Apply scissor/clipping rectangle
							VkRect2D scissor{};
							scissor.offset.x = static_cast<int32_t>(clip_rect.x);
							scissor.offset.y = static_cast<int32_t>(clip_rect.y);
							scissor.extent.width = static_cast<uint32_t>(clip_rect.z - clip_rect.x);
							scissor.extent.height = static_cast<uint32_t>(clip_rect.w - clip_rect.y);
							commandBuffer->setScissor(0, 1, &scissor);
						}

						// Draw
						commandBuffer->drawIndexed(
							pcmd->ElemCount,
							1,
							pcmd->IdxOffset + global_idx_offset,
							pcmd->VtxOffset + global_vtx_offset,
							0
						);
					}
				}
				global_idx_offset += cmd_list->IdxBuffer.Size;
				global_vtx_offset += cmd_list->VtxBuffer.Size;
			}
		}

		return true;

	}

	void UI::Update(float const deltaTimeInSec, float const mousePosX, float const mousePosY, size_t const mouseEventsCount, ui::SMouseEvent const * mouseEvents) // TODO: Keyboard events
	{
		auto & io = ImGui::GetIO();
		io.DeltaTime = deltaTimeInSec;
		io.DisplaySize = ImVec2(m_window->getWidth(), m_window->getHeight());

		HandleMouseEvents(mousePosX, mousePosY, mouseEventsCount, mouseEvents);

		ImGui::NewFrame();
		hasFocus = false;
		for (auto const& subscriber : m_subscribers)
			subscriber.listener();

		ImGui::Render();
	}

	void UI::BeginWindow(char const* windowName)
	{
		ImGui::Begin(windowName);
	}

	void UI::EndWindow()
	{
		if (ImGui::IsWindowFocused())
			hasFocus = true;

		ImGui::End();
	}

	int UI::Register(std::function<void()> const& listener)
	{
		assert(listener != nullptr);
		static int NextId = 0;
		m_subscribers.emplace_back(NextId++, listener);
		return m_subscribers.back().id;
	}

	bool UI::UnRegister(int const listenerId)
	{
		for (int i = m_subscribers.size() - 1; i >= 0; --i)
		{
			if (m_subscribers[i].id == listenerId)
			{
				m_subscribers.erase(m_subscribers.begin() + i);
				return true;
			}
		}
		return false;
	}

	void UI::SetNextItemWidth(float const nextItemWidth)
	{
		ImGui::SetNextItemWidth(nextItemWidth);
	}

	void UI::SetWindowSize(float const width, float const height)
	{
		ImGui::SetWindowSize(ImVec2(width, height));
	}

	void UI::Text(char const* label, ...)
	{
		va_list args;
		va_start(args, label);
		ImGui::TextV(label, args);
		va_end(args);
	}

	void UI::InputFloat(char const* label, float* value)
	{
		ImGui::InputFloat(label, value);
	}

	void UI::InputFloat2(char const* label, float* value)
	{
		ImGui::InputFloat2(label, value);
	}

	void UI::InputFloat3(char const* label, float* value)
	{
		ImGui::InputFloat3(label, value);
	}

	void UI::InputFloat4(char const* label, float* value)
	{
		ImGui::InputFloat3(label, value);
	}

	void UI::InputFloat3(char const* label, nbl::core::vector3df&value)
	{
		float tempValue[3]{ value.X, value.Y, value.Z };
		InputFloat3(label, tempValue);

		if (memcmp(tempValue, &value.X, sizeof(float) * 3) != 0)
		{
			memcpy(&value.X, tempValue, sizeof(float) * 3);
		}
	}

	bool UI::Combo(char const* label, int32_t* selectedItemIndex, char const** items, int32_t const itemsCount)
	{
		return ImGui::Combo(label, selectedItemIndex, items, itemsCount);
	}

	//-------------------------------------------------------------------------------------------------
	// Based on https://eliasdaler.github.io/using-imgui-with-sfml-pt2/
	static auto vector_getter = [](void* vec, int idx, const char** out_text)
	{
		auto const& vector = *static_cast<std::vector<std::string>*>(vec);
		if (idx < 0 || idx >= static_cast<int>(vector.size())) { return false; }
		*out_text = vector.at(idx).c_str();
		return true;
	};

	bool UI::Combo(const char* label, int* selectedItemIndex, std::vector<std::string>& values)
	{
		if (values.empty())
			return false;

		return ImGui::Combo(label, selectedItemIndex, vector_getter, &values, static_cast<int>(values.size()));
	}

	void UI::SliderInt(char const* label, int* value, int const minValue, int const maxValue)
	{
		ImGui::SliderInt(label, value, minValue, maxValue);
	}

	void UI::SliderFloat(char const* label, float* value, float const minValue, float const maxValue)
	{
		ImGui::SliderFloat(label, value, minValue, maxValue);
	}

	void UI::Checkbox(char const* label, bool* value)
	{
		ImGui::Checkbox(label, value);
	}

	void UI::Spacing()
	{
		ImGui::Spacing();
	}

	void UI::Button(char const* label, std::function<void()> const& onPress)
	{
		if (ImGui::Button(label))
		{
			assert(onPress != nullptr);
			//SceneManager::AssignMainThreadTask([onPress]()->void{
			onPress();
			//});
		}
	}

	void UI::InputText(char const* label, std::string& outValue)
	{
		ImGui::InputText(label, &outValue);
	}

	bool UI::HasFocus()
	{
		return hasFocus;
	}

	bool UI::IsItemActive()
	{
		return ImGui::IsItemActive();
	}

	bool UI::TreeNode(char const* name)
	{
		return ImGui::TreeNode(name);
	}

	void UI::TreePop()
	{
		ImGui::TreePop();
	}
}