#include "UI_System.hpp"

#include "imgui/imgui.h"
#include "imgui/imgui_stdlib.h"

#include <nbl/ui/CCursorControlWin32.h>

using namespace nbl::video;
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::ui;

namespace UI_System
{

	//-----------------------------------------------------------------------------
	// SHADERS
	//-----------------------------------------------------------------------------

	// glsl_shader.vert, compiled with:
	// # glslangValidator -V -x -o glsl_shader.vert.u32 glsl_shader.vert
	/*
	#version 450 core
	layout(location = 0) in vec2 aPos;
	layout(location = 1) in vec2 aUV;
	layout(location = 2) in vec4 aColor;
	layout(push_constant) uniform uPushConstant { vec2 uScale; vec2 uTranslate; } pc;

	out gl_PerVertex { vec4 gl_Position; };
	layout(location = 0) out struct { vec4 Color; vec2 UV; } Out;

	void main()
	{
		Out.Color = aColor;
		Out.UV = aUV;
		gl_Position = vec4(aPos * pc.uScale + pc.uTranslate, 0, 1);
	}
	*/
	static uint32_t vertexShaderSpv[] =
	{
		0x07230203,0x00010000,0x00080001,0x0000002e,0x00000000,0x00020011,0x00000001,0x0006000b,
		0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
		0x000a000f,0x00000000,0x00000004,0x6e69616d,0x00000000,0x0000000b,0x0000000f,0x00000015,
		0x0000001b,0x0000001c,0x00030003,0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,
		0x00000000,0x00030005,0x00000009,0x00000000,0x00050006,0x00000009,0x00000000,0x6f6c6f43,
		0x00000072,0x00040006,0x00000009,0x00000001,0x00005655,0x00030005,0x0000000b,0x0074754f,
		0x00040005,0x0000000f,0x6c6f4361,0x0000726f,0x00030005,0x00000015,0x00565561,0x00060005,
		0x00000019,0x505f6c67,0x65567265,0x78657472,0x00000000,0x00060006,0x00000019,0x00000000,
		0x505f6c67,0x7469736f,0x006e6f69,0x00030005,0x0000001b,0x00000000,0x00040005,0x0000001c,
		0x736f5061,0x00000000,0x00060005,0x0000001e,0x73755075,0x6e6f4368,0x6e617473,0x00000074,
		0x00050006,0x0000001e,0x00000000,0x61635375,0x0000656c,0x00060006,0x0000001e,0x00000001,
		0x61725475,0x616c736e,0x00006574,0x00030005,0x00000020,0x00006370,0x00040047,0x0000000b,
		0x0000001e,0x00000000,0x00040047,0x0000000f,0x0000001e,0x00000002,0x00040047,0x00000015,
		0x0000001e,0x00000001,0x00050048,0x00000019,0x00000000,0x0000000b,0x00000000,0x00030047,
		0x00000019,0x00000002,0x00040047,0x0000001c,0x0000001e,0x00000000,0x00050048,0x0000001e,
		0x00000000,0x00000023,0x00000000,0x00050048,0x0000001e,0x00000001,0x00000023,0x00000008,
		0x00030047,0x0000001e,0x00000002,0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,
		0x00030016,0x00000006,0x00000020,0x00040017,0x00000007,0x00000006,0x00000004,0x00040017,
		0x00000008,0x00000006,0x00000002,0x0004001e,0x00000009,0x00000007,0x00000008,0x00040020,
		0x0000000a,0x00000003,0x00000009,0x0004003b,0x0000000a,0x0000000b,0x00000003,0x00040015,
		0x0000000c,0x00000020,0x00000001,0x0004002b,0x0000000c,0x0000000d,0x00000000,0x00040020,
		0x0000000e,0x00000001,0x00000007,0x0004003b,0x0000000e,0x0000000f,0x00000001,0x00040020,
		0x00000011,0x00000003,0x00000007,0x0004002b,0x0000000c,0x00000013,0x00000001,0x00040020,
		0x00000014,0x00000001,0x00000008,0x0004003b,0x00000014,0x00000015,0x00000001,0x00040020,
		0x00000017,0x00000003,0x00000008,0x0003001e,0x00000019,0x00000007,0x00040020,0x0000001a,
		0x00000003,0x00000019,0x0004003b,0x0000001a,0x0000001b,0x00000003,0x0004003b,0x00000014,
		0x0000001c,0x00000001,0x0004001e,0x0000001e,0x00000008,0x00000008,0x00040020,0x0000001f,
		0x00000009,0x0000001e,0x0004003b,0x0000001f,0x00000020,0x00000009,0x00040020,0x00000021,
		0x00000009,0x00000008,0x0004002b,0x00000006,0x00000028,0x00000000,0x0004002b,0x00000006,
		0x00000029,0x3f800000,0x00050036,0x00000002,0x00000004,0x00000000,0x00000003,0x000200f8,
		0x00000005,0x0004003d,0x00000007,0x00000010,0x0000000f,0x00050041,0x00000011,0x00000012,
		0x0000000b,0x0000000d,0x0003003e,0x00000012,0x00000010,0x0004003d,0x00000008,0x00000016,
		0x00000015,0x00050041,0x00000017,0x00000018,0x0000000b,0x00000013,0x0003003e,0x00000018,
		0x00000016,0x0004003d,0x00000008,0x0000001d,0x0000001c,0x00050041,0x00000021,0x00000022,
		0x00000020,0x0000000d,0x0004003d,0x00000008,0x00000023,0x00000022,0x00050085,0x00000008,
		0x00000024,0x0000001d,0x00000023,0x00050041,0x00000021,0x00000025,0x00000020,0x00000013,
		0x0004003d,0x00000008,0x00000026,0x00000025,0x00050081,0x00000008,0x00000027,0x00000024,
		0x00000026,0x00050051,0x00000006,0x0000002a,0x00000027,0x00000000,0x00050051,0x00000006,
		0x0000002b,0x00000027,0x00000001,0x00070050,0x00000007,0x0000002c,0x0000002a,0x0000002b,
		0x00000028,0x00000029,0x00050041,0x00000011,0x0000002d,0x0000001b,0x0000000d,0x0003003e,
		0x0000002d,0x0000002c,0x000100fd,0x00010038
	};

	// glsl_shader.frag, compiled with:
	// # glslangValidator -V -x -o glsl_shader.frag.u32 glsl_shader.frag
	/*
	#version 450 core
	layout(location = 0) out vec4 fColor;
	layout(set=0, binding=0) uniform sampler2D sTexture;
	layout(location = 0) in struct { vec4 Color; vec2 UV; } In;
	void main()
	{
		fColor = In.Color * texture(sTexture, In.UV.st);
	}
	*/
	static uint32_t fragmentShaderSpv[] =
	{
		0x07230203,0x00010000,0x00080001,0x0000001e,0x00000000,0x00020011,0x00000001,0x0006000b,
		0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
		0x0007000f,0x00000004,0x00000004,0x6e69616d,0x00000000,0x00000009,0x0000000d,0x00030010,
		0x00000004,0x00000007,0x00030003,0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,
		0x00000000,0x00040005,0x00000009,0x6c6f4366,0x0000726f,0x00030005,0x0000000b,0x00000000,
		0x00050006,0x0000000b,0x00000000,0x6f6c6f43,0x00000072,0x00040006,0x0000000b,0x00000001,
		0x00005655,0x00030005,0x0000000d,0x00006e49,0x00050005,0x00000016,0x78655473,0x65727574,
		0x00000000,0x00040047,0x00000009,0x0000001e,0x00000000,0x00040047,0x0000000d,0x0000001e,
		0x00000000,0x00040047,0x00000016,0x00000022,0x00000000,0x00040047,0x00000016,0x00000021,
		0x00000000,0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,0x00030016,0x00000006,
		0x00000020,0x00040017,0x00000007,0x00000006,0x00000004,0x00040020,0x00000008,0x00000003,
		0x00000007,0x0004003b,0x00000008,0x00000009,0x00000003,0x00040017,0x0000000a,0x00000006,
		0x00000002,0x0004001e,0x0000000b,0x00000007,0x0000000a,0x00040020,0x0000000c,0x00000001,
		0x0000000b,0x0004003b,0x0000000c,0x0000000d,0x00000001,0x00040015,0x0000000e,0x00000020,
		0x00000001,0x0004002b,0x0000000e,0x0000000f,0x00000000,0x00040020,0x00000010,0x00000001,
		0x00000007,0x00090019,0x00000013,0x00000006,0x00000001,0x00000000,0x00000000,0x00000000,
		0x00000001,0x00000000,0x0003001b,0x00000014,0x00000013,0x00040020,0x00000015,0x00000000,
		0x00000014,0x0004003b,0x00000015,0x00000016,0x00000000,0x0004002b,0x0000000e,0x00000018,
		0x00000001,0x00040020,0x00000019,0x00000001,0x0000000a,0x00050036,0x00000002,0x00000004,
		0x00000000,0x00000003,0x000200f8,0x00000005,0x00050041,0x00000010,0x00000011,0x0000000d,
		0x0000000f,0x0004003d,0x00000007,0x00000012,0x00000011,0x0004003d,0x00000014,0x00000017,
		0x00000016,0x00050041,0x00000019,0x0000001a,0x0000000d,0x00000018,0x0004003d,0x0000000a,
		0x0000001b,0x0000001a,0x00050057,0x00000007,0x0000001c,0x00000017,0x0000001b,0x00050085,
		0x00000007,0x0000001d,0x00000012,0x0000001c,0x0003003e,0x00000009,0x0000001d,0x000100fd,
		0x00010038
	};

	struct State
	{
		smart_refctd_ptr<ILogicalDevice> device{};
		smart_refctd_ptr<IGPUSampler> fontSampler{};
		smart_refctd_ptr<IDescriptorPool> descriptorPool{};
		smart_refctd_ptr<IGPUDescriptorSet> gpuDescriptorSet{};
		smart_refctd_ptr<IGPURenderpassIndependentPipeline> independentPipeline{};
		smart_refctd_ptr<IGPUGraphicsPipeline> pipeline{};
		smart_refctd_ptr<IGPUImageView> fontTexture{};
		smart_refctd_ptr<CommonAPI::InputSystem> inputSystem{};
		smart_refctd_ptr<IWindow> window{};
		std::vector<smart_refctd_ptr<IGPUBuffer>> vertexBuffers{};
		std::vector<smart_refctd_ptr<IGPUBuffer>> indexBuffers{};
		bool hasFocus = false;

		// TODO: Use a signal class instead.//Signal<> UIRecordSignal{};
		struct Subscriber {
			int id = -1;
			std::function<void()> listener = nullptr;
		};
		std::vector<Subscriber> subscribers{};
	};

	static State* state = nullptr;

	struct PushConstants
	{
		float scale[2];
		float translate[2];
	};

	//-------------------------------------------------------------------------------------------------

	static smart_refctd_ptr<IGPUDescriptorSetLayout> createDescriptorSetLayout()
	{
		static constexpr int Count = 1;
		IGPUDescriptorSetLayout::SBinding bindings[1]{
			IGPUDescriptorSetLayout::SBinding {
				.binding = 0,
				.type = EDT_COMBINED_IMAGE_SAMPLER,
				.count = 1,
				.stageFlags = IShader::ESS_FRAGMENT,
				.samplers = &state->fontSampler,
			}
		};
		return state->device->createGPUDescriptorSetLayout(bindings, bindings + Count);
	}

	//-------------------------------------------------------------------------------------------------

	static void createPipeline(
		smart_refctd_ptr<IGPURenderpass>& renderPass,
		IGPUPipelineCache* pipelineCache
	)
	{
		// Constants: we are using 'vec2 offset' and 'vec2 scale' instead of a full 3d projection matrix
		static constexpr int PushConstantCount = 1;
		SPushConstantRange pushConstantRanges[PushConstantCount]{
			SPushConstantRange {
				.stageFlags = IShader::ESS_VERTEX,
				.offset = 0,
				.size = 4 * sizeof(float)
			}
		};

		auto descriptorSetLayout = createDescriptorSetLayout();

		// Create Descriptor Set:
		state->gpuDescriptorSet = state->device->createGPUDescriptorSet(
			state->descriptorPool.get(),
			descriptorSetLayout
		); // Original number was 1 , Now it creates as many as swap_chain_image_count

		auto pipelineLayout = state->device->createGPUPipelineLayout(
			pushConstantRanges,
			pushConstantRanges + PushConstantCount,
			std::move(descriptorSetLayout)
		);

		// Shaders ------------------------------------------
		// Vertex shader
		smart_refctd_ptr<ICPUBuffer> vertCpuBuffer = make_smart_refctd_ptr<ICPUBuffer>(sizeof(vertexShaderSpv));

		memcpy(vertCpuBuffer->getPointer(), vertexShaderSpv, vertCpuBuffer->getSize());   // TODO: Can we avoid this copy ?

		smart_refctd_ptr<ICPUShader> cpuVertShader = make_smart_refctd_ptr<ICPUShader>(std::move(vertCpuBuffer), IShader::ESS_VERTEX, "");

		auto const unSpecVertexShader = state->device->createGPUShader(std::move(cpuVertShader));

		auto const vertexShader = state->device->createGPUSpecializedShader(
			unSpecVertexShader.get(),
			IGPUSpecializedShader::SInfo(nullptr, nullptr, "main")
		);

		assert(vertexShader.isValid());

		// Fragment shader
		smart_refctd_ptr<ICPUBuffer> cpuFragBuffer = make_smart_refctd_ptr<ICPUBuffer>(sizeof(fragmentShaderSpv));

		memcpy(cpuFragBuffer->getPointer(), fragmentShaderSpv, cpuFragBuffer->getSize());   // TODO: Can we avoid this copy ?

		smart_refctd_ptr<ICPUShader> cpuFragShader = make_smart_refctd_ptr<ICPUShader>(std::move(cpuFragBuffer), IShader::ESS_FRAGMENT, "");

		auto const unSpecFragmentShader = state->device->createGPUShader(std::move(cpuFragShader));

		auto const fragmentShader = state->device->createGPUSpecializedShader(
			unSpecFragmentShader.get(),
			IGPUSpecializedShader::SInfo(nullptr, nullptr, "main")
		);

		assert(fragmentShader.isValid());

		IGPUSpecializedShader* shaders[2] = { vertexShader.get(), fragmentShader.get() };

		// Vertex input params --------------------------
		SVertexInputParams vertexInputParams{};
		vertexInputParams.enabledBindingFlags = 0b1u;
		vertexInputParams.enabledAttribFlags = 0b111u;
		vertexInputParams.bindings[0].inputRate = EVIR_PER_VERTEX;
		vertexInputParams.bindings[0].stride = sizeof(ImDrawVert);
		vertexInputParams.attributes[0].format = EF_R32G32_SFLOAT;//VK_FORMAT_R32G32_SFLOAT;
		vertexInputParams.attributes[0].relativeOffset = offsetof(ImDrawVert, pos);
		vertexInputParams.attributes[0].binding = 0u;
		vertexInputParams.attributes[1].format = EF_R32G32_SFLOAT;//VK_FORMAT_R32G32_SFLOAT;
		vertexInputParams.attributes[1].relativeOffset = offsetof(ImDrawVert, uv);
		vertexInputParams.attributes[1].binding = 0u;
		vertexInputParams.attributes[2].format = EF_R8G8B8A8_UNORM;//VK_FORMAT_R8G8B8A8_UNORM;
		vertexInputParams.attributes[2].relativeOffset = offsetof(ImDrawVert, col);
		vertexInputParams.attributes[2].binding = 0u;

		// Blend params ----------------------------------
		SBlendParams blendParams{};
		blendParams.logicOpEnable = false;
		blendParams.logicOp = ELO_NO_OP;
		blendParams.blendParams[0].blendEnable = VK_TRUE;
		blendParams.blendParams[0].srcColorFactor = EBF_SRC_ALPHA;//VK_BLEND_FACTOR_SRC_ALPHA;
		blendParams.blendParams[0].dstColorFactor = EBF_ONE_MINUS_SRC_ALPHA;//VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blendParams.blendParams[0].colorBlendOp = EBO_ADD;//VK_BLEND_OP_ADD;
		blendParams.blendParams[0].srcAlphaFactor = EBF_ONE_MINUS_SRC_ALPHA;//VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blendParams.blendParams[0].dstAlphaFactor = EBF_ZERO;//VK_BLEND_FACTOR_ZERO;
		blendParams.blendParams[0].alphaBlendOp = EBO_ADD;//VK_BLEND_OP_ADD;
		blendParams.blendParams[0].colorWriteMask = (1u << 0u) | (1u << 1u) | (1u << 2u) | (1u << 3u);//VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		// Rasterization params -------------------------
		SRasterizationParams rasterizationParams{};
		rasterizationParams.faceCullingMode = EFCM_NONE;
		rasterizationParams.depthTestEnable = false;
		rasterizationParams.depthWriteEnable = false;
		rasterizationParams.depthBoundsTestEnable = false;
		rasterizationParams.stencilTestEnable = false;
		rasterizationParams.rasterizationSamplesHint = IImage::ESCF_1_BIT;

		SPrimitiveAssemblyParams primitiveAssemblyParams{};
		primitiveAssemblyParams.primitiveType = EPT_TRIANGLE_LIST;

		state->independentPipeline = state->device->createGPURenderpassIndependentPipeline(
			pipelineCache,
			std::move(pipelineLayout),
			shaders,
			shaders + sizeof(shaders) / sizeof(IGPUSpecializedShader*),
			vertexInputParams,
			blendParams,
			primitiveAssemblyParams,
			rasterizationParams
		);

		IGPUGraphicsPipeline::SCreationParams creationParams{
			.renderpassIndependent = smart_refctd_ptr<IGPURenderpassIndependentPipeline>(state->independentPipeline.get()),
			.rasterizationSamplesHint = IImage::ESCF_1_BIT,
			.renderpass = renderPass,
		};
		state->pipeline = state->device->createGPUGraphicsPipeline(pipelineCache, std::move(creationParams));
	}

	//-------------------------------------------------------------------------------------------------

	static void createFontTexture(IGPUObjectFromAssetConverter::SParams& cpu2GpuParams)
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
		size_t const componentsCount = 4;
		size_t const image_size = width * height * componentsCount * sizeof(uint8_t);

		// Pixels buffer-----------------------
		smart_refctd_ptr<ICPUBuffer> imageCpuBuffer = make_smart_refctd_ptr<ICPUBuffer>(image_size);
		memcpy(imageCpuBuffer->getPointer(), pixels, imageCpuBuffer->getSize());   // TODO: Can we avoid this copy ?

		// Cpu image---------------------------
		ICPUImage::SCreationParams creationParams{};
		creationParams.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);
		creationParams.type = IImage::ET_2D;
		creationParams.format = EF_R8G8B8A8_UNORM;
		creationParams.extent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1u };
		creationParams.mipLevels = 1;
		creationParams.arrayLayers = 1u;
		creationParams.samples = IImage::ESCF_1_BIT;
		creationParams.tiling = IImage::ET_OPTIMAL;

		// TODO: Check usage
		creationParams.usage = bitflag(IImage::EUF_SAMPLED_BIT) | IImage::EUF_TRANSFER_DST_BIT;
		creationParams.sharingMode = ESM_EXCLUSIVE;
		creationParams.queueFamilyIndexCount = 1u;
		creationParams.queueFamilyIndices = nullptr;
		creationParams.initialLayout = EIL_UNDEFINED;

		auto const imageRegions = make_refctd_dynamic_array<smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1ull);
		imageRegions->begin()->bufferOffset = 0ull;
		imageRegions->begin()->bufferRowLength = creationParams.extent.width;
		imageRegions->begin()->bufferImageHeight = 0u;
		imageRegions->begin()->imageSubresource = {};
		imageRegions->begin()->imageSubresource.aspectMask = IImage::EAF_COLOR_BIT;
		imageRegions->begin()->imageSubresource.layerCount = 1u;
		imageRegions->begin()->imageOffset = { 0, 0, 0 };
		imageRegions->begin()->imageExtent = { creationParams.extent.width, creationParams.extent.height, 1u };

		auto const cpuImage = ICPUImage::create(std::move(creationParams));
		cpuImage->setBufferAndRegions(std::move(imageCpuBuffer), imageRegions);

		// Gpu image------------------------------

		// Creating gpu image then copying data into it which needs a transient command buffer
		cpu2GpuParams.beginCommandBuffers();
		IGPUObjectFromAssetConverter cpu2gpuConverter{};
		auto const gpuImage = cpu2gpuConverter.getGPUObjectsFromAssets(
			&cpuImage,
			&cpuImage + 1,
			cpu2GpuParams
		);
		cpu2GpuParams.waitForCreationToComplete(false);

		// Image view--------------------------
		IGPUImageView::SCreationParams viewParams;
		viewParams.format = cpuImage->getCreationParameters().format;
		viewParams.viewType = IImageView<IGPUImage>::ET_2D;
		viewParams.subresourceRange.aspectMask = IImage::EAF_COLOR_BIT;
		viewParams.subresourceRange.baseMipLevel = 0u;
		viewParams.subresourceRange.levelCount = 1u;
		viewParams.subresourceRange.baseArrayLayer = 0u;
		viewParams.subresourceRange.layerCount = 1u;
		viewParams.image = gpuImage->begin()[0];

		state->fontTexture = state->device->createGPUImageView(std::move(viewParams));
	}

	//-------------------------------------------------------------------------------------------------

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

	//-------------------------------------------------------------------------------------------------

	static void adjustGlobalFontScale()
	{
		ImGuiIO& io = ImGui::GetIO();
		io.FontGlobalScale = 1.0f;
	}

	//-------------------------------------------------------------------------------------------------

	static void updateDescriptorSets()
	{
		// Update the Descriptor Set:
		nbl::video::IGPUDescriptorSet::SDescriptorInfo info;
		{
			info.desc = state->fontTexture;
			info.image.sampler = state->fontSampler;
			info.image.imageLayout = nbl::asset::EIL_SHADER_READ_ONLY_OPTIMAL;
		}

		IGPUDescriptorSet::SWriteDescriptorSet writeDescriptorSet{};
		writeDescriptorSet.dstSet = state->gpuDescriptorSet.get();
		writeDescriptorSet.binding = 0;
		writeDescriptorSet.arrayElement = 0;
		writeDescriptorSet.count = 1;
		writeDescriptorSet.descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
		writeDescriptorSet.info = &info;

		state->device->updateDescriptorSets(
			1,
			&writeDescriptorSet,
			0,
			nullptr
		);
	}

	//-------------------------------------------------------------------------------------------------

	static void createFontSampler()
	{
		// TODO: Recheck this settings
		IGPUSampler::SParams params{};
		params.MinLod = -1000;
		params.MaxLod = 1000;
		params.AnisotropicFilter = 1.0f;
		params.TextureWrapU = ISampler::ETC_REPEAT;
		params.TextureWrapV = ISampler::ETC_REPEAT;
		params.TextureWrapW = ISampler::ETC_REPEAT;

		state->fontSampler = state->device->createGPUSampler(params);
	}

	//-------------------------------------------------------------------------------------------------

	static void createDescriptorPool()
	{
		static constexpr int TotalSetCount = 1;
		IDescriptorPool::SDescriptorPoolSize const poolSize{
			.type = EDT_COMBINED_IMAGE_SAMPLER,
			.count = TotalSetCount,
		};
		state->descriptorPool = state->device->createDescriptorPool(
			IDescriptorPool::E_CREATE_FLAGS::ECF_NONE,
			TotalSetCount,
			1,
			&poolSize
		);
	}

	//-------------------------------------------------------------------------------------------------

	void Init(
		nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> const& device,
		int maxFramesInFlight,
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass>& renderPass,
		nbl::video::IGPUPipelineCache* pipelineCache,
		nbl::video::IGPUObjectFromAssetConverter::SParams& cpu2GpuParams,
		nbl::core::smart_refctd_ptr<CommonAPI::InputSystem>& inputSystem,
		nbl::core::smart_refctd_ptr<nbl::ui::IWindow>& window
	)
	{
		state = new State();

		state->device = device;

		state->inputSystem = inputSystem;

		state->window = window;

		// Setup Dear ImGui context
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();

		createFontSampler();

		createDescriptorPool();

		createDescriptorSetLayout();

		createPipeline(renderPass, pipelineCache);

		createFontTexture(cpu2GpuParams);

		prepareKeyMapForDesktop();

		adjustGlobalFontScale();

		updateDescriptorSets();

		auto & io = ImGui::GetIO();
		io.DisplaySize = ImVec2(window->getWidth(), window->getHeight());
		io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);

		state->vertexBuffers.resize(maxFramesInFlight);
		state->indexBuffers.resize(maxFramesInFlight);}

	//-------------------------------------------------------------------------------------------------

	static void UpdateMousePositionAndButtons()
	{
		// TODO: Handle keyboard events
		//// Set OS mouse position if requested (rarely used, only when ImGuiConfigFlags_NavEnableSetMousePos is enabled by user)
		//if (io.WantSetMousePos)
		//{
		//    RF::WarpMouseInWindow(static_cast<int32_t>(io.MousePos.x), static_cast<int32_t>(io.MousePos.y));
		//}
		//else
		//{
		//    io.MousePos = ImVec2(-FLT_MAX, -FLT_MAX);
		//}
		//int mx, my;
		//uint32_t const mouse_buttons = MSDL::SDL_GetMouseState(&mx, &my);
		//io.MouseDown[0] = (mouse_buttons & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0;  // If a mouse press event came, always pass it as "mouse held this frame", so we don't miss click-release events that are shorter than 1 frame.
		//io.MouseDown[1] = (mouse_buttons & SDL_BUTTON(SDL_BUTTON_RIGHT)) != 0;
		//io.MouseDown[2] = (mouse_buttons & SDL_BUTTON(SDL_BUTTON_MIDDLE)) != 0;
		//if (RF::GetWindowFlags() & MSDL::SDL_WINDOW_INPUT_FOCUS)
		//{
		//    io.MousePos = ImVec2(static_cast<float>(mx), static_cast<float>(my));
		//}

		CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
		//CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

		state->inputSystem->getDefaultMouse(&mouse);
		//inputSystem->getDefaultKeyboard(&keyboard);

		static std::chrono::microseconds previousEventTimestamp {};

		mouse.consumeEvents([](const IMouseEventChannel::range_t& events) -> void {
			auto& io = ImGui::GetIO();
			for (auto event : events)
			{
				if (event.timeStamp < previousEventTimestamp)
				{
					continue;
				}
				previousEventTimestamp = event.timeStamp;

				if(event.type == nbl::ui::SMouseEvent::EET_CLICK)
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

					if(event.clickEvent.action == SMouseEvent::SClickEvent::EA_PRESSED) {
						io.MouseDown[buttonIndex] = true;
					} else if (event.clickEvent.action == SMouseEvent::SClickEvent::EA_RELEASED) {
						io.MouseDown[buttonIndex] = false;
					}
				}
			}
		});

		auto const position = state->window->getCursorControl()->getPosition();
		auto& io = ImGui::GetIO();
		io.MousePos.x = static_cast<float>(position.x) - state->window->getX();
		io.MousePos.y = static_cast<float>(position.y) - state->window->getY();
	}

	//-------------------------------------------------------------------------------------------------

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
		//    MSDL::SDL_SetCursor(state->mouseCursors[imgui_cursor] ? state->mouseCursors[imgui_cursor] : state->mouseCursors[ImGuiMouseCursor_Arrow]);
		//    MSDL::SDL_ShowCursor(MSDL::SDL_TRUE);
		//}
	//}

	//-------------------------------------------------------------------------------------------------

	bool Render(IGPUCommandBuffer& commandBuffer, int frameIndex)
	{
		ImGuiIO& io = ImGui::GetIO();
		assert(io.Fonts->IsBuilt() && "Font atlas not built! It is generally built by the renderer backend. Missing call to renderer _NewFrame() function? e.g. ImGui_ImplOpenGL3_NewFrame().");

		UpdateMousePositionAndButtons();
		
		// Setup desired Vulkan state
		// Bind pipeline and descriptor sets:
		commandBuffer.bindGraphicsPipeline(state->pipeline.get());
		commandBuffer.bindDescriptorSets(EPBP_GRAPHICS, state->independentPipeline->getLayout(), 0, 1, &state->gpuDescriptorSet.get());
		
		auto const* drawData = ImGui::GetDrawData();

		if (drawData == nullptr)
		{
			return false;
		}
		
		// Avoid rendering when minimized, scale coordinates for retina displays (screen coordinates != framebuffer coordinates)
		float const frameBufferWidth = drawData->DisplaySize.x * drawData->FramebufferScale.x;
		float const frameBufferHeight = drawData->DisplaySize.y * drawData->FramebufferScale.y;
		if (frameBufferWidth > 0 && frameBufferHeight > 0 && drawData->TotalVtxCount > 0)
		{
			// Create or resize the vertex/index buffers
			size_t const vertexSize = drawData->TotalVtxCount * sizeof(ImDrawVert);
			size_t const indexSize = drawData->TotalIdxCount * sizeof(ImDrawIdx);

			IGPUBuffer::SCreationParams vertexCreationParams = {};
			vertexCreationParams.usage = nbl::asset::IBuffer::EUF_VERTEX_BUFFER_BIT;
			vertexCreationParams.canUpdateSubRange = true;

			auto & vertexBuffer = state->vertexBuffers[frameIndex];

			if (vertexBuffer.isValid() == false || vertexBuffer->getSize() < vertexSize)
			{
				vertexBuffer = state->device->createCPUSideGPUVisibleGPUBufferOnDedMem(
					vertexCreationParams,
					vertexSize
				);
			}

			IGPUBuffer::SCreationParams indexCreationParams = {};
			indexCreationParams.usage = nbl::asset::IBuffer::EUF_INDEX_BUFFER_BIT;
			indexCreationParams.canUpdateSubRange = true;

			auto & indexBuffer = state->indexBuffers[frameIndex];

			if (indexBuffer.isValid() == false || indexBuffer->getSize() < indexSize)
			{
				indexBuffer = state->device->createCPUSideGPUVisibleGPUBufferOnDedMem(
					indexCreationParams,
					indexSize
				);
			}

			{
				IDriverMemoryAllocation::MappedMemoryRange const vertexMappedMemory(
					vertexBuffer->getBoundMemory(),
					vertexBuffer->getBoundMemoryOffset(),
					vertexBuffer->getSize()
				);

				IDriverMemoryAllocation::MappedMemoryRange const indexMappedMemory(
					indexBuffer->getBoundMemory(),
					indexBuffer->getBoundMemoryOffset(),
					indexBuffer->getSize()
				);

				state->device->mapMemory(vertexMappedMemory, IDriverMemoryAllocation::EMCAF_WRITE);
				state->device->mapMemory(indexMappedMemory, IDriverMemoryAllocation::EMCAF_WRITE);

				auto* vertex_ptr = static_cast<ImDrawVert*>(vertexMappedMemory.memory->getMappedPointer());
				auto* index_ptr = static_cast<ImDrawIdx*>(indexMappedMemory.memory->getMappedPointer());


				for (int n = 0; n < drawData->CmdListsCount; n++)
				{
					const ImDrawList* cmd = drawData->CmdLists[n];
					::memcpy(vertex_ptr, cmd->VtxBuffer.Data, cmd->VtxBuffer.Size * sizeof(ImDrawVert));
					::memcpy(index_ptr, cmd->IdxBuffer.Data, cmd->IdxBuffer.Size * sizeof(ImDrawIdx));
					vertex_ptr += cmd->VtxBuffer.Size;
					index_ptr += cmd->IdxBuffer.Size;
				}

				state->device->unmapMemory(vertexMappedMemory.memory);
				state->device->unmapMemory(indexMappedMemory.memory);
			}

			commandBuffer.bindIndexBuffer(
				indexBuffer.get(),
				0,
				sizeof(ImDrawIdx) == 2 ? EIT_16BIT : EIT_32BIT
			);

			static constexpr size_t offset = 0;
			commandBuffer.bindVertexBuffers(
				0,
				1,
				&vertexBuffer.get(),
				&offset
			);

			// Setup viewport:
			SViewport const viewport
			{
				.x = 0,
				.y = 0,
				.width = frameBufferWidth,
				.height = frameBufferHeight,
				.minDepth = 0.0f,
				.maxDepth = 1.0f,
			};
			commandBuffer.setViewport(0, 1, &viewport);

			// Setup scale and translation:
			// Our visible imgui space lies from draw_data->DisplayPps (top left) to draw_data->DisplayPos+data_data->DisplaySize (bottom right). DisplayPos is (0,0) for single viewport apps.
			{
				PushConstants constants{};
				constants.scale[0] = 2.0f / drawData->DisplaySize.x;
				constants.scale[1] = 2.0f / drawData->DisplaySize.y;
				constants.translate[0] = -1.0f - drawData->DisplayPos.x * constants.scale[0];
				constants.translate[1] = -1.0f - drawData->DisplayPos.y * constants.scale[1];

				commandBuffer.pushConstants(
					state->independentPipeline->getLayout(),
					IShader::ESS_VERTEX,
					0,
					sizeof(constants),
					&constants
				);
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
							commandBuffer.setScissor(0, 1, &scissor);
						}

						// Draw
						commandBuffer.drawIndexed(
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

	//-------------------------------------------------------------------------------------------------

	void Update(float const deltaTimeInSec)
	{
		auto & io = ImGui::GetIO();
		io.DeltaTime = deltaTimeInSec;
		io.DisplaySize = ImVec2(state->window->getWidth(), state->window->getHeight());
		ImGui::NewFrame();
		state->hasFocus = false;
		for (auto const& subscriber : state->subscribers)
		{
			subscriber.listener();
		}
		ImGui::Render();
	}

	//-------------------------------------------------------------------------------------------------

	void BeginWindow(char const* windowName)
	{
		ImGui::Begin(windowName);
	}

	//-------------------------------------------------------------------------------------------------

	void EndWindow()
	{
		if (ImGui::IsWindowFocused())
		{
			state->hasFocus = true;
		}
		ImGui::End();
	}

	//-------------------------------------------------------------------------------------------------

	int Register(std::function<void()> const& listener)
	{
		assert(listener != nullptr);
		static int NextId = 0;
		state->subscribers.emplace_back(NextId++, listener);
		return state->subscribers.back().id;
	}

	//-------------------------------------------------------------------------------------------------

	bool UnRegister(int const listenerId)
	{
		for (int i = state->subscribers.size() - 1; i >= 0; --i)
		{
			if (state->subscribers[i].id == listenerId)
			{
				state->subscribers.erase(state->subscribers.begin() + i);
				return true;
			}
		}
		return false;
	}

	//-------------------------------------------------------------------------------------------------

	void SetNextItemWidth(float const nextItemWidth)
	{
		ImGui::SetNextItemWidth(nextItemWidth);
	}

	//-------------------------------------------------------------------------------------------------

	void SetWindowSize(float const width, float const height)
	{
		ImGui::SetWindowSize(ImVec2(width, height));
	}

	//-------------------------------------------------------------------------------------------------

	void Text(char const* label, ...)
	{
		va_list args;
		va_start(args, label);
		ImGui::TextV(label, args);
		va_end(args);
	}

	//-------------------------------------------------------------------------------------------------

	void InputFloat(char const* label, float* value)
	{
		ImGui::InputFloat(label, value);
	}

	//-------------------------------------------------------------------------------------------------

	void InputFloat2(char const* label, float* value)
	{
		ImGui::InputFloat2(label, value);
	}

	//-------------------------------------------------------------------------------------------------

	void InputFloat3(char const* label, float* value)
	{
		ImGui::InputFloat3(label, value);
	}

	//-------------------------------------------------------------------------------------------------

	void InputFloat4(char const* label, float* value)
	{
		ImGui::InputFloat3(label, value);
	}

	//-------------------------------------------------------------------------------------------------

	void InputFloat3(char const* label, glm::vec3& value)
	{
		float tempValue[3]{ value.x, value.y, value.z };
		UI::InputFloat3(label, tempValue);

		if (memcmp(tempValue, &value[0], sizeof(float) * 3) != 0)
		{
			memcpy(&value[0], tempValue, sizeof(float) * 3);
		}
	}

	//-------------------------------------------------------------------------------------------------

	// TODO Maybe we could cache unchanged vertices
	bool Combo(
		char const* label,
		int32_t* selectedItemIndex,
		char const** items,
		int32_t const itemsCount
	)
	{
		return ImGui::Combo(
			label,
			selectedItemIndex,
			items,
			itemsCount
		);
	}

	//-------------------------------------------------------------------------------------------------
	// Based on https://eliasdaler.github.io/using-imgui-with-sfml-pt2/
	static auto vector_getter = [](void* vec, int idx, const char** out_text)
	{
		auto& vector = *static_cast<std::vector<std::string>*>(vec);
		if (idx < 0 || idx >= static_cast<int>(vector.size())) { return false; }
		*out_text = vector.at(idx).c_str();
		return true;
	};

	bool Combo(
		const char* label,
		int* selectedItemIndex,
		std::vector<std::string>& values
	)
	{
		if (values.empty())
		{
			return false;
		}
		return ImGui::Combo(
			label,
			selectedItemIndex,
			vector_getter,
			&values,
			static_cast<int>(values.size())
		);
	}

	//-------------------------------------------------------------------------------------------------

	void SliderInt(
		char const* label,
		int* value,
		int const minValue,
		int const maxValue
	)
	{
		ImGui::SliderInt(
			label,
			value,
			minValue,
			maxValue
		);
	}

	//-------------------------------------------------------------------------------------------------

	void SliderFloat(
		char const* label,
		float* value,
		float const minValue,
		float const maxValue
	)
	{
		ImGui::SliderFloat(
			label,
			value,
			minValue,
			maxValue
		);
	}

	//-------------------------------------------------------------------------------------------------

	void Checkbox(char const* label, bool* value)
	{
		ImGui::Checkbox(label, value);
	}

	//-------------------------------------------------------------------------------------------------

	void Spacing()
	{
		ImGui::Spacing();
	}

	//-------------------------------------------------------------------------------------------------

	void Button(char const* label, std::function<void()> const& onPress)
	{
		if (ImGui::Button(label))
		{
			assert(onPress != nullptr);
			//SceneManager::AssignMainThreadTask([onPress]()->void{
			onPress();
			//});
		}
	}

	//-------------------------------------------------------------------------------------------------

	void InputText(char const* label, std::string& outValue)
	{
		ImGui::InputText(label, &outValue);
	}

	//-------------------------------------------------------------------------------------------------

	bool HasFocus()
	{
		return state->hasFocus;
	}

	//-------------------------------------------------------------------------------------------------

	void Shutdown()
	{
		// TODO: Unregister from resize event

		delete state;
		state = nullptr;
	}

	//-------------------------------------------------------------------------------------------------

	bool IsItemActive()
	{
		return ImGui::IsItemActive();
	}

	//-------------------------------------------------------------------------------------------------

	bool TreeNode(char const* name)
	{
		return ImGui::TreeNode(name);
	}

	//-------------------------------------------------------------------------------------------------

	void TreePop()
	{
		ImGui::TreePop();
	}

	//-------------------------------------------------------------------------------------------------

}
