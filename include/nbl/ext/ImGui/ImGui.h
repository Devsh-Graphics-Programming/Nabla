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
		void update(float deltaTimeInSec, float mousePosX, float mousePosY, size_t mouseEventsCount, ui::SMouseEvent const * mouseEvents); // TODO: Keyboard events
		int registerListener(std::function<void()> const& listener);
		bool unregisterListener(int listenerId);
		
		void* getContext();
		void setContext(void* imguiContext);

	private:
		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createDescriptorSetLayout();
		void createPipeline(video::IGPURenderpass* renderpass, video::IGPUPipelineCache* pipelineCache);
		void createFontTexture(video::IGPUCommandBuffer* cmdBuffer, video::IQueue* queue);
		void updateDescriptorSets();
		void createSystem();
		void createFontSampler();
		void createDescriptorPool();
		void handleMouseEvents(float mousePosX, float mousePosY, size_t mouseEventsCount, ui::SMouseEvent const * mouseEvents) const;

		core::smart_refctd_ptr<system::ISystem> system;
		core::smart_refctd_ptr<system::ILogger> logger;
		core::smart_refctd_ptr<video::IUtilities> utilities;
		core::smart_refctd_ptr<video::ILogicalDevice> m_device;
		core::smart_refctd_ptr<video::IGPUSampler> m_fontSampler;
		core::smart_refctd_ptr<video::IDescriptorPool> m_descriptorPool;
		core::smart_refctd_ptr<video::IGPUDescriptorSet> m_gpuDescriptorSet;
		core::smart_refctd_ptr<video::IGPUGraphicsPipeline> pipeline;
		core::smart_refctd_ptr<video::IGPUImageView> m_fontTexture;
		core::smart_refctd_ptr<ui::IWindow> m_window;
		std::vector<core::smart_refctd_ptr<video::IGPUBuffer>> m_mdiBuffers;
		const uint32_t maxFramesInFlight;

		// TODO: Use a signal class instead like Signal<> UIRecordSignal{};
		struct Subscriber 
		{
			int id = -1;
			std::function<void()> listener = nullptr;
		};
		std::vector<Subscriber> m_subscribers{};

};
}

#endif	// NBL_EXT_IMGUI_UI_H
