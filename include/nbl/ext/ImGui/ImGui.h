#ifndef _NBL_EXT_IMGUI_UI_H_
#define _NBL_EXT_IMGUI_UI_H_

#include "nbl/video/declarations.h"

namespace nbl::ext::imgui
{

class UI final : public core::IReferenceCounted
{
	public:
		// Nabla IMGUI backend reserves this index for font atlas, any attempt to hook user defined texture within the index will cause runtime error
		_NBL_STATIC_INLINE_CONSTEXPR auto NBL_FONT_ATLAS_TEX_ID = 0u;

		UI(core::smart_refctd_ptr<video::ILogicalDevice> _device, core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> _descriptorSetLayout, video::IGPURenderpass* renderpass, video::IGPUPipelineCache* pipelineCache, core::smart_refctd_ptr<ui::IWindow> window);
		~UI() override;

		bool render(nbl::video::SIntendedSubmitInfo& info, const nbl::video::IGPUDescriptorSet* const descriptorSet);
		void update(float deltaTimeInSec, const nbl::hlsl::float32_t2 mousePosition, const core::SRange<const nbl::ui::SMouseEvent> mouseEvents, const core::SRange<const nbl::ui::SKeyboardEvent> keyboardEvents);
		int registerListener(std::function<void()> const& listener);
		bool unregisterListener(uint32_t id);
		void setContext(void* imguiContext);

		inline nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> getFontAtlasView() { return m_fontAtlasTexture; }
		inline auto getStreamingBuffer() -> decltype(auto) { return (std::as_const(m_mdi.streamingTDBufferST)); }
		void* getContext();

	private:
		void createPipeline(core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> descriptorSetLayout, video::IGPURenderpass* renderpass, video::IGPUPipelineCache* pipelineCache);

		// TODO: just take an intended next submit instead of queue and cmdbuf, so we're consistent across utilities

		void createMDIBuffer(const uint32_t totalByteSize);
		video::ISemaphore::future_t<video::IQueue::RESULT> createFontAtlasTexture(video::IGPUCommandBuffer* cmdBuffer, video::IQueue* queue);
		void createSystem();
		void handleMouseEvents(const nbl::hlsl::float32_t2& mousePosition, const core::SRange<const nbl::ui::SMouseEvent>& events) const;
		void handleKeyEvents(const core::SRange<const nbl::ui::SKeyboardEvent>& events) const;

		core::smart_refctd_ptr<system::ISystem> system;
		core::smart_refctd_ptr<system::ILogger> logger;
		core::smart_refctd_ptr<video::IUtilities> utilities;
		core::smart_refctd_ptr<video::ILogicalDevice> m_device;

		core::smart_refctd_ptr<video::IGPUGraphicsPipeline> pipeline;
		core::smart_refctd_ptr<video::IGPUImageView> m_fontAtlasTexture;
		core::smart_refctd_ptr<ui::IWindow> m_window;

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
