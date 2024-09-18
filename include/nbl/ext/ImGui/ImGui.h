#ifndef _NBL_EXT_IMGUI_UI_H_
#define _NBL_EXT_IMGUI_UI_H_

#include "nbl/video/declarations.h"
#include "nbl/asset/IAssetManager.h"

namespace nbl::ext::imgui
{
class UI final : public core::IReferenceCounted
{
	public:
		struct MDI
		{
			using COMPOSE_T = nbl::video::StreamingTransientDataBufferST<nbl::core::allocator<uint8_t>>;

			enum E_BUFFER_CONTENT : uint8_t
			{
				EBC_DRAW_INDIRECT_STRUCTURES,
				EBC_ELEMENT_STRUCTURES,
				EBC_INDEX_BUFFERS,
				EBC_VERTEX_BUFFERS,

				EBC_COUNT,
			};

			nbl::core::smart_refctd_ptr<typename COMPOSE_T> streamingTDBufferST; //! composed buffer layout is [EBC_DRAW_INDIRECT_STRUCTURES] [EBC_ELEMENT_STRUCTURES] [EBC_INDEX_BUFFERS] [EBC_VERTEX_BUFFERS]

			static constexpr auto MDI_BUFFER_REQUIRED_ALLOCATE_FLAGS = nbl::core::bitflag<nbl::video::IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS>(nbl::video::IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT); //! required flags
			static constexpr auto MDI_BUFFER_REQUIRED_USAGE_FLAGS = nbl::core::bitflag(nbl::asset::IBuffer::EUF_INDIRECT_BUFFER_BIT) | nbl::asset::IBuffer::EUF_INDEX_BUFFER_BIT | nbl::asset::IBuffer::EUF_VERTEX_BUFFER_BIT | nbl::asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT; //! required flags
		};

		struct S_CREATION_PARAMETERS
		{
			struct S_RESOURCE_PARAMETERS
			{
				nbl::video::IGPUPipelineLayout* const pipelineLayout = nullptr;				//! optional, default layout used if not provided declaring required UI resources such as textures (required font atlas + optional user defined textures) & corresponding samplers
				uint32_t count = 0x45u;														//! amount of total UI textures (and corresponding samplers)

				struct S_BINDING_REQUEST_INFO												//! for a given pipeline layout we need to know what is intended for UI resources
				{
					uint32_t setIx,															//! descriptor set index for a resource	
					bindingIx;																//! binding index for a given resource
				};

				const S_BINDING_REQUEST_INFO textures = { .setIx = 0u, .bindingIx = 0u },	//! optional, default texture binding request info used if not provided (set & binding index)
				samplers = { .setIx = 0u, .bindingIx = 1u };								//! optional, default sampler binding request info used if not provided (set & binding index)		

				using binding_flags_t = nbl::video::IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS;
				static constexpr auto TEXTURES_REQUIRED_CREATE_FLAGS = nbl::core::bitflag(binding_flags_t::ECF_UPDATE_AFTER_BIND_BIT) | binding_flags_t::ECF_PARTIALLY_BOUND_BIT | binding_flags_t::ECF_UPDATE_UNUSED_WHILE_PENDING_BIT; //! required flags
				static constexpr auto SAMPLERS_REQUIRED_CREATE_FLAGS = nbl::core::bitflag(binding_flags_t::ECF_NONE); //! required flags
				static constexpr auto RESOURCES_REQUIRED_STAGE_FLAGS = nbl::asset::IShader::E_SHADER_STAGE::ESS_FRAGMENT; //! required stage
			};

			nbl::asset::IAssetManager* const assetManager;									//! required		
			nbl::video::IUtilities* const utilities;										//! required
			nbl::video::IQueue* const transfer;												//! required
			nbl::video::IGPURenderpass* const renderpass;									//! required
			uint32_t subpassIx = 0u;														//! optional, default value used if not provided
			S_RESOURCE_PARAMETERS resources;												//! optional, default parameters used if not provided
			nbl::video::IGPUPipelineCache* const pipelineCache = nullptr;					//! optional, no cache used if not provided
			typename MDI::COMPOSE_T* const streamingMDIBuffer = nullptr;					//! optional, default MDI buffer allocated if not provided	
		};

		//! parameters which may change every frame, used with the .update call to interact with ImGuiIO; we require a very *required* minimum - if you need to cover more IO options simply get the IO with ImGui::GetIO() to customize them (they all have default values you can change before calling the .update)
		struct S_UPDATE_PARAMETERS
		{
			//! what we pass to ImGuiIO::AddMousePosEvent 
			nbl::hlsl::float32_t2 mousePosition,

			//! main display size in pixels
			displaySize;

			//! Nabla events you want to be handled with the backend
			struct S_EVENTS
			{
				core::SRange<const nbl::ui::SMouseEvent> mouse;
				core::SRange<const nbl::ui::SKeyboardEvent> keyboard;
			};

			S_EVENTS events;
		};

		UI(S_CREATION_PARAMETERS&& params);
		~UI() override;

		//! Nabla ImGUI backend reserves this index for font atlas, any attempt to hook user defined texture within the index will result in undefined behaviour
		static constexpr auto NBL_FONT_ATLAS_TEX_ID = 0u;

		//! updates ImGuiIO & records ImGUI *cpu* draw command lists, you have to call it before .render
		bool update(const S_UPDATE_PARAMETERS& params);

		//! updates mapped mdi buffer & records *gpu* draw command, you are required to bind UI's graphics pipeline & descriptor sets before calling this function - use getPipeline() to get the pipeline & getCreationParameters() to get info about your set resources
		bool render(nbl::video::IGPUCommandBuffer* commandBuffer, nbl::video::ISemaphore::SWaitInfo waitInfo, const std::span<const VkRect2D> scissors = {});

		//! registers lambda listener in which ImGUI calls should be recorded
		size_t registerListener(std::function<void()> const& listener);
		std::optional<size_t> unregisterListener(size_t id);

		//! sets ImGUI context, you are supposed to pass valid ImGuiContext* context
		void setContext(void* imguiContext);

		//! creation parametrs
		inline const S_CREATION_PARAMETERS& getCreationParameters() const { return m_creationParams; }

		//! ImGUI graphics pipeline
		inline nbl::video::IGPUGraphicsPipeline* getPipeline() { return pipeline.get(); }

		//! image view default font texture
		inline nbl::video::IGPUImageView* getFontAtlasView() { return m_fontAtlasTexture.get(); }

		//! mdi streaming buffer
		inline typename MDI::COMPOSE_T* getStreamingBuffer() { return m_mdi.streamingTDBufferST.get(); }

		//! ImGUI context, you are supposed to cast it, eg. reinterpret_cast<ImGuiContext*>(this->getContext());
		void* getContext();
	private:
		void createPipeline();
		void createMDIBuffer();
		void handleMouseEvents(const S_UPDATE_PARAMETERS& params) const;
		void handleKeyEvents(const S_UPDATE_PARAMETERS& params) const;
		video::ISemaphore::future_t<video::IQueue::RESULT> createFontAtlasTexture(video::IGPUCommandBuffer* cmdBuffer);

		S_CREATION_PARAMETERS m_creationParams;

		core::smart_refctd_ptr<video::IGPUGraphicsPipeline> pipeline;
		core::smart_refctd_ptr<video::IGPUImageView> m_fontAtlasTexture;

		MDI m_mdi;
		std::vector<std::function<void()>> m_subscribers {};
};
}

#endif	// NBL_EXT_IMGUI_UI_H
