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

			nbl::core::smart_refctd_ptr<typename COMPOSE_T> streamingTDBufferST; // composed buffer layout is [EBC_DRAW_INDIRECT_STRUCTURES] [EBC_ELEMENT_STRUCTURES] [EBC_INDEX_BUFFERS] [EBC_VERTEX_BUFFERS]
		};

		UI(core::smart_refctd_ptr<video::ILogicalDevice> _device, core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> _descriptorSetLayout, video::IGPURenderpass* renderpass, uint32_t subpassIx, video::IGPUPipelineCache* pipelineCache = nullptr, nbl::core::smart_refctd_ptr<typename MDI::COMPOSE_T> _streamingMDIBuffer = nullptr);
		~UI() override;

		//! Nabla ImGUI backend reserves this index for font atlas, any attempt to hook user defined texture within the index will cause runtime error [TODO: could have a setter & getter to control the default & currently hooked font texture ID and init 0u by default]
		_NBL_STATIC_INLINE_CONSTEXPR auto NBL_FONT_ATLAS_TEX_ID = 0u;

		//! update ImGUI internal state & cpu draw command lists, call it before this->render
		bool update(const ui::IWindow* window, float deltaTimeInSec, const core::SRange<const nbl::ui::SMouseEvent> mouseEvents, const core::SRange<const nbl::ui::SKeyboardEvent> keyboardEvents);

		//! updates mapped mdi buffer & records draw calls
		bool render(nbl::video::SIntendedSubmitInfo& info, const nbl::video::IGPUDescriptorSet* const descriptorSet);

		//! registers lambda listener in which ImGUI calls should be recorded
		int registerListener(std::function<void()> const& listener);
		bool unregisterListener(uint32_t id);

		//! sets ImGUI context, you are supposed to pass valid ImGuiContext* context
		void setContext(void* imguiContext);

		//! image view default font texture
		inline nbl::video::IGPUImageView* getFontAtlasView() { return m_fontAtlasTexture.get(); }

		//! mdi streaming buffer
		inline typename MDI::COMPOSE_T* getStreamingBuffer() { return m_mdi.streamingTDBufferST.get(); }

		//! ImGUI context getter, you are supposed to cast it, eg. reinterpret_cast<ImGuiContext*>(this->getContext());
		void* getContext();
	private:
		void createSystem();
		void createPipeline(core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> descriptorSetLayout, video::IGPURenderpass* renderpass, uint32_t subpassIx, video::IGPUPipelineCache* pipelineCache);
		void createMDIBuffer(nbl::core::smart_refctd_ptr<typename MDI::COMPOSE_T> _streamingMDIBuffer);
		video::ISemaphore::future_t<video::IQueue::RESULT> createFontAtlasTexture(video::IGPUCommandBuffer* cmdBuffer, video::IQueue* queue);
		void handleMouseEvents(const core::SRange<const nbl::ui::SMouseEvent>& events, const ui::IWindow* window) const;
		void handleKeyEvents(const core::SRange<const nbl::ui::SKeyboardEvent>& events) const;

		core::smart_refctd_ptr<system::ISystem> system;
		core::smart_refctd_ptr<system::ILogger> logger;
		core::smart_refctd_ptr<video::IUtilities> utilities;
		core::smart_refctd_ptr<video::ILogicalDevice> m_device;

		core::smart_refctd_ptr<video::IGPUGraphicsPipeline> pipeline;
		core::smart_refctd_ptr<video::IGPUImageView> m_fontAtlasTexture;

		MDI m_mdi;

		// TODO: Use a signal class instead like Signal<> UIRecordSignal{};
		struct Subscriber 
		{
			uint32_t id = 0;
			std::function<void()> listener = nullptr;
		};
		std::vector<Subscriber> m_subscribers{};
};
}

#endif	// NBL_EXT_IMGUI_UI_H
