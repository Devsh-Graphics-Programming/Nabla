#ifndef _NBL_EXT_IMGUI_UI_H_
#define _NBL_EXT_IMGUI_UI_H_

#include "nbl/video/declarations.h"
#include "nbl/asset/IAssetManager.h"

namespace nbl::ext::imgui
{
class UI final : public core::IReferenceCounted
{
	public:
		//! Reserved font atlas indicies for backend textures & samplers descriptor binding's array, any attempt to hook user defined texture ID == FontAtlasTexId will result in undefined behaviour
		static constexpr auto FontAtlasTexId = 0u, FontAtlasSamplerId = 0u;

		struct SMdiBuffer
		{
			//! composes memory available for the general purpose allocator to suballocate memory ranges
			using compose_t = video::StreamingTransientDataBufferST<core::allocator<uint8_t>>;

			//! traits for MDI buffer suballocator - fills the data given the mdi allocator memory request
			using suballocator_traits_t = core::address_allocator_traits<core::LinearAddressAllocatorST<uint32_t>>;

			enum class Content : uint16_t
			{
				INDIRECT_STRUCTURES,
				ELEMENT_STRUCTURES,
				INDEX_BUFFERS,
				VERTEX_BUFFERS,

				COUNT,
			};

			//! streaming mdi buffer
			core::smart_refctd_ptr<typename compose_t> compose;

			//! required buffer allocate flags
			static constexpr auto RequiredAllocateFlags = core::bitflag<video::IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS>(video::IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);

			//! required buffer usage flags
			static constexpr auto RequiredUsageFlags = core::bitflag(asset::IBuffer::EUF_INDIRECT_BUFFER_BIT) | asset::IBuffer::EUF_INDEX_BUFFER_BIT | asset::IBuffer::EUF_VERTEX_BUFFER_BIT | asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
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

			//! Reserved indexes for default backend samplers descriptor binding's array - use only if you created your pipeline layout with createDefaultPipelineLayout. If you need more or custom samplers then create the pipeline layout yourself
			enum class DefaultSamplerIx : uint16_t
			{
				FONT_ATLAS = FontAtlasSamplerId,
				USER,

				COUNT,
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
				uint32_t texturesCount, samplersCount;

			friend class UI;
		};

		struct SCachedCreationParams
		{
			//! required, you provide us information about your required UI binding resources which we validate at creation time
			SResourceParameters resources;

			//! required
			core::smart_refctd_ptr<video::IUtilities> utilities;

			//! optional, default MDI buffer allocated if not provided	
			core::smart_refctd_ptr<typename SMdiBuffer::compose_t> const streamingBuffer = nullptr;
		};

		struct SCreationParameters : public SCachedCreationParams
		{
			//! required
			video::IQueue* const transfer = nullptr;

			//! required, must declare required UI resources such as textures (required font atlas + optional user defined textures) & samplers
			core::smart_refctd_ptr<video::IGPUPipelineLayout> pipelineLayout;

			//! required
			core::smart_refctd_ptr<asset::IAssetManager> assetManager = nullptr;		

			//! required
			core::smart_refctd_ptr<video::IGPURenderpass> renderpass = nullptr;		

			//! optional, default value used if not provided
			uint32_t subpassIx = 0u;														

			//! optional, no cache used if not provided
			core::smart_refctd_ptr<video::IGPUPipelineCache> const pipelineCache = nullptr;
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
		bool render(video::IGPUCommandBuffer* const commandBuffer, video::ISemaphore::SWaitInfo waitInfo, const std::span<const VkRect2D> scissors = {});

		//! registers lambda listener in which ImGUI calls should be recorded, use the returned id to unregister the listener
		size_t registerListener(std::function<void()> const& listener);

		//! unregisters listener with the given id
		std::optional<size_t> unregisterListener(size_t id);

		//! sets ImGUI context, you are supposed to pass valid ImGuiContext* context
		void setContext(void* imguiContext);

		//! creates default pipeline layout for the UI resources, "texturesCount" argument is textures descriptor binding's array size. Samplers are immutable and part of the created layout, SResourceParameters::DefaultSamplerIx::COUNT is the size of the samplers descriptor binding's array
		static core::smart_refctd_ptr<video::IGPUPipelineLayout> createDefaultPipelineLayout(video::IUtilities* const utilities, const SResourceParameters::SBindingInfo texturesInfo = { .setIx = 0u, .bindingIx = 0u }, const SResourceParameters::SBindingInfo samplersInfo = { .setIx = 0u, .bindingIx = 1u }, uint32_t texturesCount = 0x45);

		//! creation cached parametrs
		inline const SCachedCreationParams& getCreationParameters() const { return m_cachedCreationParams; }

		//! ImGUI graphics pipeline
		inline const video::IGPUGraphicsPipeline* getPipeline() const { return m_pipeline.get(); }

		//! image view default font texture
		inline const video::IGPUImageView* getFontAtlasView() const { return m_fontAtlasTexture.get(); }

		//! mdi streaming buffer
		inline const typename SMdiBuffer::compose_t* getStreamingBuffer() const { return m_mdi.compose.get(); }

		//! ImGUI context, you are supposed to cast it, eg. reinterpret_cast<ImGuiContext*>(this->getContext());
		void* getContext();
	private:
		void createPipeline();
		void createMDIBuffer();
		void handleMouseEvents(const SUpdateParameters& params) const;
		void handleKeyEvents(const SUpdateParameters& params) const;
		video::ISemaphore::future_t<video::IQueue::RESULT> createFontAtlasTexture(video::IGPUCommandBuffer* cmdBuffer);

		SCachedCreationParams m_cachedCreationParams;

		core::smart_refctd_ptr<video::IGPUGraphicsPipeline> m_pipeline;
		core::smart_refctd_ptr<video::IGPUImageView> m_fontAtlasTexture;

		SMdiBuffer m_mdi;
		std::vector<std::function<void()>> m_subscribers {};
};
}

#endif	// NBL_EXT_IMGUI_UI_H
