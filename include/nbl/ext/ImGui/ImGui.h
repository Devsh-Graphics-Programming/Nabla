#ifndef _NBL_EXT_IMGUI_UI_H_
#define _NBL_EXT_IMGUI_UI_H_

namespace nbl::ext::imgui
{

class UI final : public core::IReferenceCounted
{
	public:
		UI(core::smart_refctd_ptr<video::ILogicalDevice> device, uint32_t _maxFramesInFlight, video::IGPURenderpass* renderpass, video::IGPUPipelineCache* pipelineCache, core::smart_refctd_ptr<ui::IWindow> window);
		~UI() override;

		bool render(nbl::video::IGPUCommandBuffer* commandBuffer, const uint32_t frameIndex);
		void update(float deltaTimeInSec, const nbl::hlsl::float32_t2 mousePosition, const core::SRange<const nbl::ui::SMouseEvent> mouseEvents, const core::SRange<const nbl::ui::SKeyboardEvent> keyboardEvents);
		int registerListener(std::function<void()> const& listener);
		bool unregisterListener(uint32_t id);
		
		void* getContext();
		void setContext(void* imguiContext);

	private:
		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createDescriptorSetLayout();
		void createPipeline(video::IGPURenderpass* renderpass, video::IGPUPipelineCache* pipelineCache);
		void createFontAtlas2DArrayTexture(video::IGPUCommandBuffer* cmdBuffer, video::IQueue* queue);
		void updateDescriptorSets();
		void createSystem();
		void createFontAtlasSampler();
		void createDescriptorPool();
		void handleMouseEvents(const nbl::hlsl::float32_t2& mousePosition, const core::SRange<const nbl::ui::SMouseEvent>& events) const;
		void handleKeyEvents(const core::SRange<const nbl::ui::SKeyboardEvent>& events) const;

		core::smart_refctd_ptr<system::ISystem> system;
		core::smart_refctd_ptr<system::ILogger> logger;
		core::smart_refctd_ptr<video::IUtilities> utilities;
		core::smart_refctd_ptr<video::ILogicalDevice> m_device;
		core::smart_refctd_ptr<video::IGPUSampler> m_fontSampler;
		core::smart_refctd_ptr<video::IDescriptorPool> m_descriptorPool;
		core::smart_refctd_ptr<video::IGPUDescriptorSet> m_gpuDescriptorSet;
		core::smart_refctd_ptr<video::IGPUGraphicsPipeline> pipeline;
		core::smart_refctd_ptr<video::IGPUImageView> m_fontAtlas2DArrayTexture;
		core::smart_refctd_ptr<ui::IWindow> m_window;
		std::vector<core::smart_refctd_ptr<video::IGPUBuffer>> m_mdiBuffers;
		const uint32_t maxFramesInFlight;

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
