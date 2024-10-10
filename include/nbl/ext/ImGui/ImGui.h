#ifndef _NBL_EXT_IMGUI_UI_H_
#define _NBL_EXT_IMGUI_UI_H_

#include "nbl/video/declarations.h"
#include "nbl/asset/IAssetManager.h"

namespace nbl::ext::imgui
{
class UI final : public core::IReferenceCounted
{
	public:
		//! Reserved font atlas indicies for default backend textures & samplers descriptor binding's array
		static constexpr auto FontAtlasTexId = 0u, FontAtlasSamplerId = 0u;

		//! Reserved indexes for default backend samplers descriptor binding's array - use only if you created your pipeline layout with createDefaultPipelineLayout. If you need more or custom samplers then create the pipeline layout yourself
		enum class DefaultSamplerIx : uint16_t
		{
			FONT_ATLAS = FontAtlasSamplerId,
			USER,

			COUNT,
		};

		struct SResourceParameters
		{
			//! for a given pipeline layout we need to know what is intended for UI resources
			struct SBindingInfo
			{
				//! descriptor set index for a resource
				uint32_t setIx,

				//! binding index for a given resource
				bindingIx;
			};

			using binding_flags_t = video::IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS;

			//! required textures binding creation flags
			static constexpr auto TexturesRequiredCreateFlags = core::bitflag(binding_flags_t::ECF_UPDATE_AFTER_BIND_BIT) | binding_flags_t::ECF_PARTIALLY_BOUND_BIT | binding_flags_t::ECF_UPDATE_UNUSED_WHILE_PENDING_BIT;

			//! required samplers binding creation flags
			static constexpr auto SamplersRequiredCreateFlags = core::bitflag(binding_flags_t::ECF_UPDATE_AFTER_BIND_BIT);

			//! required shader stage flags
			static constexpr auto RequiredShaderStageFlags = asset::IShader::E_SHADER_STAGE::ESS_FRAGMENT;

			//! required, fill the info to instruct the backend about the required UI resources
			SBindingInfo texturesInfo, samplersInfo;

			private:
				uint32_t texturesCount = {}, samplersCount = {};

			friend class UI;
		};

		struct SCachedCreationParams
		{
			using streaming_buffer_t = video::StreamingTransientDataBufferST<core::allocator<uint8_t>>;

			//! required buffer allocate flags
			static constexpr auto RequiredAllocateFlags = core::bitflag<video::IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS>(video::IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);

			//! required buffer usage flags
			static constexpr auto RequiredUsageFlags = core::bitflag(asset::IBuffer::EUF_INDIRECT_BUFFER_BIT) | asset::IBuffer::EUF_INDEX_BUFFER_BIT | asset::IBuffer::EUF_VERTEX_BUFFER_BIT | asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;

			//! required, you provide us information about your required UI binding resources which we validate at creation time
			SResourceParameters resources;

			//! required
			core::smart_refctd_ptr<video::IUtilities> utilities;

			//! optional, default MDI buffer allocated if not provided
			core::smart_refctd_ptr<streaming_buffer_t> streamingBuffer = nullptr;

			//! optional, default single one
			uint32_t viewportCount = 1u;
		};

		struct SCreationParameters : public SCachedCreationParams
		{
			//! required
			video::IQueue* transfer = nullptr;

			//! required, must declare required UI resources such as textures (required font atlas + optional user defined textures) & samplers
			core::smart_refctd_ptr<video::IGPUPipelineLayout> pipelineLayout;

			//! required
			core::smart_refctd_ptr<asset::IAssetManager> assetManager = nullptr;		

			//! required
			core::smart_refctd_ptr<video::IGPURenderpass> renderpass = nullptr;		

			//! optional, default value used if not provided
			uint32_t subpassIx = 0u;														

			//! optional, no cache used if not provided
			core::smart_refctd_ptr<video::IGPUPipelineCache> pipelineCache = nullptr;
		};

		//! parameters which may change every frame, used with the .update call to interact with ImGuiIO; we require a very *required* minimum - if you need to cover more IO options simply get the IO with ImGui::GetIO() to customize them (they all have default values you can change before calling the .update)
		struct SUpdateParameters
		{
			//! what we pass to ImGuiIO::AddMousePosEvent 
			hlsl::float32_t2 mousePosition,

			//! main display size in pixels
			displaySize;

			//! Nabla mouse events you want to be handled with the backend
			std::span<const ui::SMouseEvent> mouseEvents = {};

			//! Nabla keyboard events you want to be handled with the backend
			std::span<const ui::SKeyboardEvent> keyboardEvents = {};
		};

		UI(SCreationParameters&& params);
		~UI() override;

		//! updates ImGuiIO & records ImGUI *cpu* draw command lists, you have to call it before .render
		bool update(const SUpdateParameters& params);

		//! updates mapped mdi buffer & records *gpu* draw command, you are required to bind UI's graphics pipeline & descriptor sets before calling this function - use getPipeline() to get the pipeline & getCreationParameters() to get info about your set resources
		bool render(video::IGPUCommandBuffer* const commandBuffer, video::ISemaphore::SWaitInfo waitInfo, const std::chrono::steady_clock::time_point waitPoint = std::chrono::steady_clock::now() + std::chrono::milliseconds(1u), const std::span<const VkRect2D> scissors = {});

		//! registers lambda listener in which ImGUI calls should be recorded, use the returned id to unregister the listener
		size_t registerListener(std::function<void()> const& listener);

		//! unregisters listener with the given id
		std::optional<size_t> unregisterListener(size_t id);

		//! sets ImGUI context, you are supposed to pass valid ImGuiContext* context
		void setContext(void* imguiContext);

		//! creates default pipeline layout for the UI resources, "texturesCount" argument is textures descriptor binding's array size. Samplers are immutable and part of the created layout, SResourceParameters::DefaultSamplerIx::COUNT is the size of the samplers descriptor binding's array
		static core::smart_refctd_ptr<video::IGPUPipelineLayout> createDefaultPipelineLayout(video::IUtilities* const utilities, const SResourceParameters::SBindingInfo texturesInfo, const SResourceParameters::SBindingInfo samplersInfo, uint32_t texturesCount = 0x45);

		//! mounts the extension's archive to given system - useful if you want to create your own shaders with common header included
		static const core::smart_refctd_ptr<system::IFileArchive> mount(core::smart_refctd_ptr<system::ILogger> logger, system::ISystem* system, const std::string_view archiveAlias = "");

		//! creation cached parametrs
		inline const SCachedCreationParams& getCreationParameters() const { return m_cachedCreationParams; }

		//! ImGUI graphics pipeline
		inline const video::IGPUGraphicsPipeline* getPipeline() const { return m_pipeline.get(); }

		//! image view default font texture
		//! TODO: we cannot expose immutable view of our default font texture because user MUST be able to write a descriptor, 
		//! we can have mutable getter or we can decide to not create any default font texture at all and force users 
		//! to do it externally (have a static helper to do it which gives you mutable view)
		inline video::IGPUImageView* getFontAtlasView() const { return m_fontAtlasTexture.get(); }

		//! mdi streaming buffer
		inline const auto* getStreamingBuffer() const {return m_cachedCreationParams.streamingBuffer.get();}

		//! ImGUI context, you are supposed to cast it, eg. reinterpret_cast<ImGuiContext*>(this->getContext());
		void* getContext();
	private:
		void createPipeline(SCreationParameters& creationParams);
		void createMDIBuffer(SCreationParameters& creationParams);
		void handleMouseEvents(const SUpdateParameters& params) const;
		void handleKeyEvents(const SUpdateParameters& params) const;
		video::ISemaphore::future_t<video::IQueue::RESULT> createFontAtlasTexture(video::IGPUCommandBuffer* cmdBuffer, SCreationParameters& creationParams);

		SCachedCreationParams m_cachedCreationParams;

		core::smart_refctd_ptr<video::IGPUGraphicsPipeline> m_pipeline;
		core::smart_refctd_ptr<video::IGPUImageView> m_fontAtlasTexture;

		std::vector<std::function<void()>> m_subscribers {};
};
}

#endif	// NBL_EXT_IMGUI_UI_H
