#include <iostream>
#include <map>
#include <ranges>
#include <vector>
#include <utility>

#include "nbl/system/IApplicationFramework.h"
#include "nbl/system/CStdoutLogger.h"
#include "nbl/ext/ImGui/ImGui.h"
#include "shaders/common.hlsl"
#include "ext/imgui/spirv/builtin/builtinResources.h"
#include "ext/imgui/spirv/builtin/CArchive.h"

#include "imgui/imgui.h"
#include "imgui/misc/cpp/imgui_stdlib.h"

using namespace nbl::video;
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::ui;

namespace nbl::ext::imgui
{
	void UI::createPipeline(core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> descriptorSetLayout, video::IGPURenderpass* renderpass, IGPUPipelineCache* pipelineCache)
	{
		// Constants: we are using 'vec2 offset' and 'vec2 scale' instead of a full 3d projection matrix
		SPushConstantRange pushConstantRanges[] = 
		{
			{
				.stageFlags = IShader::E_SHADER_STAGE::ESS_VERTEX | IShader::E_SHADER_STAGE::ESS_FRAGMENT,
				.offset = 0,
				.size = sizeof(PushConstants)
			}
		};

		auto pipelineLayout = m_device->createPipelineLayout(pushConstantRanges, core::smart_refctd_ptr(descriptorSetLayout));

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

			shaders.vertex = createShader(spirv.vertex, IShader::E_SHADER_STAGE::ESS_VERTEX);
			shaders.fragment = createShader(spirv.fragment, IShader::E_SHADER_STAGE::ESS_FRAGMENT);
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
				auto& param = params[0u];
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

	ISemaphore::future_t<IQueue::RESULT> UI::createFontAtlasTexture(video::IGPUCommandBuffer* cmdBuffer, video::IQueue* transfer)
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
		io.Fonts->SetTexID(NBL_FONT_ATLAS_TEX_ID);

		if (!pixels || width<=0 || height<=0)
			return IQueue::RESULT::OTHER_ERROR;

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

		auto image = m_device->createImage(std::move(params));

		if (!image)
		{
			logger->log("Could not create font image!", system::ILogger::ELL_ERROR);
			return IQueue::RESULT::OTHER_ERROR;
		}
		image->setObjectDebugName("Nabla IMGUI extension Font Image");

		if (!m_device->allocate(image->getMemoryReqs(), image.get()).isValid())
		{
			logger->log("Could not allocate memory for font image!", system::ILogger::ELL_ERROR);
			return IQueue::RESULT::OTHER_ERROR;
		}
		
		image->setObjectDebugName("Nabla IMGUI extension Font Atlas");

		SIntendedSubmitInfo sInfo;
		{
			IQueue::SSubmitInfo::SCommandBufferInfo cmdInfo = { cmdBuffer };

			auto scratchSemaphore = m_device->createSemaphore(0);
			if (!scratchSemaphore)
			{
				logger->log("Could not create scratch semaphore", system::ILogger::ELL_ERROR);
				return IQueue::RESULT::OTHER_ERROR;
			}
			scratchSemaphore->setObjectDebugName("Nabla IMGUI extension Scratch Semaphore");

			sInfo.queue = transfer;
			sInfo.waitSemaphores = {};
			sInfo.commandBuffers = { &cmdInfo, 1 };
			sInfo.scratchSemaphore =
			{
				.semaphore = scratchSemaphore.get(),
				.value = 0u,
				.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
			};
			
			// we have no explicit source stage and access to sync against, brand new clean image.
			const asset::SMemoryBarrier toTransferDep = {
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
			if (!utilities->updateImageViaStagingBuffer(sInfo,pixels,image->getCreationParameters().format,image.get(),transferLayout,regions.range))
			{
				logger->log("Could not upload font image contents", system::ILogger::ELL_ERROR);
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
			if (transfer->submit(submit)!=IQueue::RESULT::SUCCESS)
			{
				logger->log("Could not submit workload for font texture upload.", system::ILogger::ELL_ERROR);
				return IQueue::RESULT::OTHER_ERROR;
			}
		}
		 
		{
			IGPUImageView::SCreationParams params;
			params.format = image->getCreationParameters().format;
			params.viewType = IImageView<IGPUImage>::ET_2D;
			params.subresourceRange = regions.subresource;
			params.image = core::smart_refctd_ptr(image);

			m_fontAtlasTexture = m_device->createImageView(std::move(params));
		}
		
        ISemaphore::future_t<IQueue::RESULT> retval(IQueue::RESULT::SUCCESS);
        retval.set({sInfo.scratchSemaphore.semaphore,sInfo.scratchSemaphore.value});
        return retval;
	}

	static void adjustGlobalFontScale()
	{
		ImGuiIO& io = ImGui::GetIO();
		io.FontGlobalScale = 1.0f;
	}

	void UI::handleMouseEvents(const nbl::hlsl::float32_t2& mousePosition, const core::SRange<const nbl::ui::SMouseEvent>& events) const
	{
		auto& io = ImGui::GetIO();

		const auto position = mousePosition - nbl::hlsl::float32_t2(m_window->getX(), m_window->getY());

		io.AddMousePosEvent(position.x, position.y);

		for (const auto& e : events)
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
				const auto wheel = nbl::hlsl::float32_t2(e.scrollEvent.horizontalScroll, e.scrollEvent.verticalScroll) * scalar;

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

	void UI::handleKeyEvents(const core::SRange<const nbl::ui::SKeyboardEvent>& events) const
	{
		auto& io = ImGui::GetIO();

		_NBL_STATIC_INLINE_CONSTEXPR auto keyMap = createKeyMap();

		const bool useBigLetters = [&]()  // TODO: we can later improve it to check for CAPS, etc
		{
			for (const auto& e : events)
				if (e.keyCode == EKC_LEFT_SHIFT && e.action == SKeyboardEvent::ECA_PRESSED)
					return true;

			return false;
		}();

		for (const auto& e : events)
		{
			const auto& bind = keyMap[e.keyCode];
			const auto& iCharacter = useBigLetters ? bind.physicalBig : bind.physicalSmall;

			if(bind.target == ImGuiKey_None)
				logger->log(std::string("Requested physical Nabla key \"") + iCharacter + std::string("\" has yet no mapping to IMGUI key!"), system::ILogger::ELL_ERROR);
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

	UI::UI(smart_refctd_ptr<ILogicalDevice> _device, core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> _descriptorSetLayout, uint32_t _maxFramesInFlight, video::IGPURenderpass* renderpass, IGPUPipelineCache* pipelineCache, smart_refctd_ptr<IWindow> window)
		: m_device(core::smart_refctd_ptr(_device)), m_window(core::smart_refctd_ptr(window)), maxFramesInFlight(_maxFramesInFlight)
	{
		createSystem();
		struct
		{
			struct
			{
				uint8_t transfer, graphics;
			} id;
		} families;

		const nbl::video::IPhysicalDevice* pDevice = m_device->getPhysicalDevice();
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
			return uint8_t(0); // silent warnings
		};

		// get & validate families' capabilities
		families.id.transfer = requestFamilyQueueId(IQueue::FAMILY_FLAGS::TRANSFER_BIT, "Could not find any queue with TRANSFER_BIT enabled!");
		families.id.graphics = requestFamilyQueueId(IQueue::FAMILY_FLAGS::GRAPHICS_BIT, "Could not find any queue with GRAPHICS_BIT enabled!");

		// allocate temporary command buffer
		auto* tQueue = m_device->getThreadSafeQueue(families.id.transfer, 0);

		if (!tQueue)
		{
			logger->log("Could not get queue!", system::ILogger::ELL_ERROR);
			assert(false);
		}

		smart_refctd_ptr<nbl::video::IGPUCommandBuffer> transistentCMD;
		{
			using pool_flags_t = IGPUCommandPool::CREATE_FLAGS;
			// need to be individually resettable such that we can form a valid SIntendedSubmit out of the commandbuffer allocated from the pool
			smart_refctd_ptr<nbl::video::IGPUCommandPool> pool = m_device->createCommandPool(families.id.transfer, pool_flags_t::RESET_COMMAND_BUFFER_BIT|pool_flags_t::TRANSIENT_BIT);
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

			createPipeline(core::smart_refctd_ptr(_descriptorSetLayout), renderpass, pipelineCache);
			createFontAtlasTexture(transistentCMD.get(), tQueue);
			adjustGlobalFontScale();
		}
		tQueue->endCapture();

		auto & io = ImGui::GetIO();
		io.DisplaySize = ImVec2(m_window->getWidth(), m_window->getHeight());
		io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);
		io.BackendUsingLegacyKeyArrays = 0; // 0: using AddKeyEvent() [new way of handling events in imgui]

		m_mdiBuffers.resize(maxFramesInFlight);
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

	bool UI::render(IGPUCommandBuffer* commandBuffer, const IGPUDescriptorSet* const descriptorSet, const uint32_t frameIndex)
	{
		const bool validFramesRange = frameIndex >= 0 && frameIndex < maxFramesInFlight;
		if (!validFramesRange)
		{
			logger->log("Requested frame index is OOB!", system::ILogger::ELL_ERROR);
			assert(false);
		}

		ImGuiIO& io = ImGui::GetIO();

		if (!io.Fonts->IsBuilt())
		{
			logger->log("Font atlas not built! It is generally built by the renderer backend. Missing call to renderer _NewFrame() function? e.g. ImGui_ImplOpenGL3_NewFrame().", system::ILogger::ELL_ERROR);
			assert(false);
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

				bool prepass(const ImDrawCmd* cmd) const
				{
					const auto rectangle = getClipRectangle(cmd);

					return prepass(rectangle);
				}

				bool prepass(const ImVec4& clipRectangle) const
				{
					return clipRectangle.x < framebuffer.x && clipRectangle.y < framebuffer.y && clipRectangle.z >= 0.0f && clipRectangle.w >= 0.0f;
				}

			} clip { .off = drawData->DisplayPos, .scale = drawData->FramebufferScale, .framebuffer = { frameBufferWidth, frameBufferHeight } };
			
			struct TRS
			{
				core::vector2df_SIMD scale;
				core::vector2df_SIMD translate;

				core::vector2df_SIMD toNDC(core::vector2df_SIMD in) const
				{
					return in * scale + translate;
				}
			};

			const TRS trs = [&]() 
			{
				TRS retV;

				retV.scale = core::vector2df_SIMD{ 2.0f / drawData->DisplaySize.x , 2.0f / drawData->DisplaySize.y };
				retV.translate = core::vector2df_SIMD { -1.0f, -1.0f } - core::vector2df_SIMD{ drawData->DisplayPos.x, drawData->DisplayPos.y } * trs.scale;

				return std::move(retV);
			}();
			
			struct
			{
				size_t vBufferSize = {}, iBufferSize = {},	// sizes in bytes
				vPadding = {};								// indicies can break our alignment so small padding may be required before vertices in memory, vertices' offset must be multiple of 4

				std::vector<VkDrawIndexedIndirectCommand> indirects;
				std::vector<PerObjectData> elements;

				size_t getMDIBufferSize() const				// buffer layout is [Draw Indirect structures] [Element Data] [All the Indices] [All the vertices]
				{
					return getVerticesOffset() + vBufferSize;
				}

				size_t getIndicesOffset() const
				{
					return getElementsOffset() + elements.size() * sizeof(PerObjectData);
				}

				size_t getVerticesOffset() const
				{
					return getIndicesOffset() + iBufferSize + vPadding;
				}

				size_t getVkDrawIndexedIndirectCommandOffset() const
				{
					return 0u;
				}

				size_t getElementsOffset() const
				{
					return elements.size() * sizeof(VkDrawIndexedIndirectCommand);
				}

				bool validate() const
				{
					return indirects.size() == elements.size();
				}

			} requestData;

			{
				/*
					IMGUI Render command lists. We merged all buffers into a single one so we 
					maintain our own offset into them, we pre-loop to get request data for 
					MDI buffer alocation request/update.
				*/

				size_t globalIOffset = {}, globalVOffset = {}, drawID = {};

				for (int n = 0; n < drawData->CmdListsCount; n++)
				{
					const auto* cmd_list = drawData->CmdLists[n];
					for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; cmd_i++)
					{
						const auto* pcmd = &cmd_list->CmdBuffer[cmd_i];

						const auto clipRectangle = clip.getClipRectangle(pcmd);

						// count draw invocations and allocate cpu indirect + element data for mdi buffer
						if (clip.prepass(clipRectangle))
						{
							// TODO: I guess we waste a lot of time on this emplace, we should reserve memory first or use some nice allocator probably?
							auto& indirect = requestData.indirects.emplace_back();
							auto& element = requestData.elements.emplace_back();

							indirect.firstIndex = pcmd->IdxOffset + globalIOffset;
							indirect.firstInstance = drawID; // use base instance as draw ID
							indirect.indexCount = pcmd->ElemCount;
							indirect.instanceCount = 1u;
							indirect.vertexOffset = pcmd->VtxOffset + globalVOffset;

							const auto scissor = clip.getScissor(clipRectangle);

							auto packSnorm16 = [](float ndc) -> int16_t
							{
								return std::round<int16_t>(std::clamp(ndc, -1.0f, 1.0f) * 32767.0f); // TODO: ok encodePixels<asset::EF_R16_SNORM, double>(void* _pix, const double* _input) but iirc we have issues with our encode/decode utils
							};

							const auto vMin = trs.toNDC(core::vector2df_SIMD(scissor.offset.x, scissor.offset.y));
							const auto vMax = trs.toNDC(core::vector2df_SIMD(scissor.offset.x + scissor.extent.width, scissor.offset.y + scissor.extent.height));

							element.aabbMin.x = packSnorm16(vMin.x);
							element.aabbMin.y = packSnorm16(vMin.y);
							element.aabbMax.x = packSnorm16(vMax.x);
							element.aabbMax.y = packSnorm16(vMax.y);
							element.texId = pcmd->TextureId;

							++drawID;
						}
					}

					globalIOffset += cmd_list->IdxBuffer.Size;
					globalVOffset += cmd_list->VtxBuffer.Size;
				}

				requestData.iBufferSize = drawData->TotalIdxCount * sizeof(ImDrawIdx);
				requestData.vBufferSize = drawData->TotalVtxCount * sizeof(ImDrawVert);
				requestData.vPadding = requestData.getVerticesOffset() % 4u;
				assert(requestData.validate());
			}
			
			IGPUBuffer::SCreationParams mdiCreationParams = {};
			mdiCreationParams.usage = nbl::core::bitflag(nbl::asset::IBuffer::EUF_INDIRECT_BUFFER_BIT) | nbl::asset::IBuffer::EUF_INDEX_BUFFER_BIT | nbl::asset::IBuffer::EUF_VERTEX_BUFFER_BIT | nbl::asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT  | nbl::asset::IBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF;
			mdiCreationParams.size = requestData.getMDIBufferSize();

			auto mdiBuffer = m_mdiBuffers[frameIndex];
			const uint32_t drawCount = requestData.elements.size();
			const auto mdicBSize = drawCount * sizeof(VkDrawIndexedIndirectCommand);
			const auto elementsBSize = drawCount * sizeof(PerObjectData);
			const uint64_t elementsBOffset = requestData.getElementsOffset();

			// create or resize the mdi buffer
			if (!static_cast<bool>(mdiBuffer) || mdiBuffer->getSize() < mdiCreationParams.size)
			{
				mdiBuffer = m_device->createBuffer(std::move(mdiCreationParams));

				auto mReqs = mdiBuffer->getMemoryReqs();
				mReqs.memoryTypeBits &= m_device->getPhysicalDevice()->getUpStreamingMemoryTypeBits();
				
				auto memOffset = m_device->allocate(mReqs, mdiBuffer.get(), core::bitflag<IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS>(IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_DEVICE_ADDRESS_BIT));
				assert(memOffset.isValid());
			}

			auto* rawPipeline = pipeline.get();
			commandBuffer->bindGraphicsPipeline(rawPipeline);
			commandBuffer->bindDescriptorSets(EPBP_GRAPHICS, rawPipeline->getLayout(), 0, 1, &descriptorSet);

			// update mdi buffer
			{
				auto binding = mdiBuffer->getBoundMemory();

				{
					if (!binding.memory->map({ 0ull, binding.memory->getAllocationSize() }, IDeviceMemoryAllocation::EMCAF_READ))
						logger->log("Could not map device memory!", system::ILogger::ELL_WARNING);

					assert(binding.memory->isCurrentlyMapped());
				}

				auto* mdiPointer = binding.memory->getMappedPointer();

				auto getMDIPointer = [&mdiPointer]<typename T>(auto bOffset)
				{
					return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(mdiPointer) + bOffset);
				};

				auto* const indirectPointer = getMDIPointer.template operator() < VkDrawIndexedIndirectCommand > (requestData.getVkDrawIndexedIndirectCommandOffset());
				auto* const elementPointer = getMDIPointer.template operator() < PerObjectData > (elementsBOffset);
				auto* indicesPointer = getMDIPointer.template operator() < ImDrawIdx > (requestData.getIndicesOffset());
				auto* verticesPointer = getMDIPointer.template operator() < ImDrawVert > (requestData.getVerticesOffset());

				// fill indirect/element data
				::memcpy(indirectPointer, requestData.indirects.data(), mdicBSize);
				::memcpy(elementPointer, requestData.elements.data(), elementsBSize);

				// flush elements data range if required
				{
					auto eData = mdiBuffer->getBoundMemory();

					if (eData.memory->haveToMakeVisible())
					{
						const ILogicalDevice::MappedMemoryRange range(eData.memory, eData.offset, mdiBuffer->getSize());
						m_device->flushMappedMemoryRanges(1, &range);
					}
				}

				// fill vertex/index data, no flush request
				for (int n = 0; n < drawData->CmdListsCount; n++)
				{
					const auto* cmd = drawData->CmdLists[n];

					::memcpy(verticesPointer, cmd->VtxBuffer.Data, cmd->VtxBuffer.Size * sizeof(ImDrawVert));
					::memcpy(indicesPointer, cmd->IdxBuffer.Data, cmd->IdxBuffer.Size * sizeof(ImDrawIdx));

					verticesPointer += cmd->VtxBuffer.Size;
					indicesPointer += cmd->IdxBuffer.Size;
				}

				binding.memory->unmap();
			}

			{
				const asset::SBufferBinding<const video::IGPUBuffer> binding =
				{
					.offset = requestData.getIndicesOffset(),
					.buffer = core::smart_refctd_ptr(mdiBuffer)
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
						.offset = requestData.getVerticesOffset(),
						.buffer = core::smart_refctd_ptr(mdiBuffer)
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
			{
				VkRect2D scissor[] = { {.offset = {(int32_t)viewport.x, (int32_t)viewport.y}, .extent = {(uint32_t)viewport.width, (uint32_t)viewport.height}} };
				commandBuffer->setScissor(scissor); // cover whole viewport (only to not throw validation errors)
			}

			/*
				Setup scale and translation, our visible imgui space lies from draw_data->DisplayPps (top left) to 
				draw_data->DisplayPos+data_data->DisplaySize (bottom right). DisplayPos is (0,0) for single viewport apps.
			*/

			{
				PushConstants constants
				{
					.elementBDA = { mdiBuffer->getDeviceAddress() + elementsBOffset },
					.elementCount = { drawCount },
					.scale = { trs.scale[0u], trs.scale[1u] },
					.translate = { trs.translate[0u], trs.translate[1u] },
					.viewport = { viewport.x, viewport.y, viewport.width, viewport.height }
				};

				commandBuffer->pushConstants(pipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_VERTEX | IShader::E_SHADER_STAGE::ESS_FRAGMENT, 0u, sizeof(constants), &constants);
			}

			const asset::SBufferBinding<const video::IGPUBuffer> binding =
			{
				.offset = requestData.getVkDrawIndexedIndirectCommandOffset(),
				.buffer = core::smart_refctd_ptr(mdiBuffer)
			};

			commandBuffer->drawIndexedIndirect(binding, drawCount, sizeof(VkDrawIndexedIndirectCommand));
		}

		return true;
	}

	void UI::update(float const deltaTimeInSec, const nbl::hlsl::float32_t2 mousePosition, const core::SRange<const nbl::ui::SMouseEvent> mouseEvents, const core::SRange<const nbl::ui::SKeyboardEvent> keyboardEvents)
	{
		auto & io = ImGui::GetIO();

		io.DeltaTime = deltaTimeInSec;
		io.DisplaySize = ImVec2(m_window->getWidth(), m_window->getHeight());

		handleMouseEvents(mousePosition, mouseEvents);
		handleKeyEvents(keyboardEvents);

		ImGui::NewFrame();

		for (auto const& subscriber : m_subscribers)
			subscriber.listener();

		ImGui::Render(); // note it doesn't touch GPU or graphics API at all, internal call for IMGUI cpu geometry buffers update
	}

	int UI::registerListener(std::function<void()> const& listener)
	{
		assert(listener != nullptr);
		static int NextId = 0;
		m_subscribers.emplace_back(NextId++, listener);
		return m_subscribers.back().id;
	}

	bool UI::unregisterListener(const uint32_t id)
	{
		for (int i = m_subscribers.size() - 1; i >= 0; --i)
		{
			if (m_subscribers[i].id == id)
			{
				m_subscribers.erase(m_subscribers.begin() + i);
				return true;
			}
		}
		return false;
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