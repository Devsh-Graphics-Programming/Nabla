#ifndef _NBL_EXT_IMGUI_UI_H_
#define _NBL_EXT_IMGUI_UI_H_

#include "nbl/video/declarations.h"

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
		};

		struct S_CREATION_PARAMETERS
		{
			video::IUtilities* const utilities;										//! required
			video::IQueue* const transfer;											//! required
			video::IGPURenderpass* const renderpass;								//! required
			uint32_t subpassIx = 0u;												//! optional, default value if not provided
			video::IGPUDescriptorSetLayout* const descriptorSetLayout = nullptr;	//! optional, default layout used if not provided [STILL TODO, currently its assumed its not nullptr!]
			video::IGPUPipelineCache* const pipelineCache = nullptr;				//! optional, no cache used if not provided
			typename MDI::COMPOSE_T* const streamingMDIBuffer = nullptr;			//! optional, default MDI buffer allocated if not provided
		};

		//! parameters which may change every frame, used with the .update call to interact with ImGuiIO; we require a very *required* minimum - if you need to cover more IO options simply get the IO with ImGui::GetIO() to customize them (they all have default values you can change before calling the .update)
		struct S_UPDATE_PARAMETERS
		{
			//! what we pass to ImGuiIO::AddMousePosEvent 
			nbl::hlsl::float32_t2 mousePosition,

			//! main display size in pixels (generally == GetMainViewport()->Size)
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

		//! Nabla ImGUI backend reserves this index for font atlas, any attempt to hook user defined texture within the index will cause runtime error [TODO: could have a setter & getter to control the default & currently hooked font texture ID and init 0u by default]
		_NBL_STATIC_INLINE_CONSTEXPR auto NBL_FONT_ATLAS_TEX_ID = 0u;

		//! update ImGuiIO & record ImGUI *cpu* draw command lists, call it before .render
		bool update(const S_UPDATE_PARAMETERS& params);

		//! updates mapped mdi buffer & records *gpu* draw commands, handles overflows for mdi allocation failure cases (pop & submit)
		bool render(nbl::video::SIntendedSubmitInfo& info, const nbl::video::IGPUDescriptorSet* const descriptorSet, const std::span<const VkRect2D> scissors = {});

		//! registers lambda listener in which ImGUI calls should be recorded
		size_t registerListener(std::function<void()> const& listener);
		std::optional<size_t> unregisterListener(size_t id);

		//! sets ImGUI context, you are supposed to pass valid ImGuiContext* context
		void setContext(void* imguiContext);

		//! image view default font texture
		inline nbl::video::IGPUImageView* getFontAtlasView() { return m_fontAtlasTexture.get(); }

		//! mdi streaming buffer
		inline typename MDI::COMPOSE_T* getStreamingBuffer() { return m_mdi.streamingTDBufferST.get(); }

		//! ImGUI context getter, you are supposed to cast it, eg. reinterpret_cast<ImGuiContext*>(this->getContext());
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
