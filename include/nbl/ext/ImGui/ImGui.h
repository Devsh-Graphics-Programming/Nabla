#ifndef _NBL_EXT_IMGUI_UI_H_
#define _NBL_EXT_IMGUI_UI_H_

namespace nbl::ext::imgui
{
class UI final : public core::IReferenceCounted
{
	public:
		UI(core::smart_refctd_ptr<video::ILogicalDevice> device, int maxFramesInFlight, video::IGPURenderpass* renderpass, video::IGPUPipelineCache* pipelineCache, core::smart_refctd_ptr<ui::IWindow> window);
		~UI() override;

		bool Render(nbl::video::IGPUCommandBuffer* commandBuffer, int frameIndex);
		void Update(float deltaTimeInSec, float mousePosX, float mousePosY, size_t mouseEventsCount, ui::SMouseEvent const * mouseEvents); // TODO: Keyboard events
		void BeginWindow(char const* windowName);
		void EndWindow();
		int Register(std::function<void()> const& listener);
		bool UnRegister(int listenerId);
		void SetNextItemWidth(float nextItemWidth);
		void SetWindowSize(float width, float height);
		void Text(char const* label, ...);
		void InputFloat(char const* label, float* value);
		void InputFloat2(char const* label, float* value);
		void InputFloat3(char const* label, float* value);
		void InputFloat4(char const* label, float* value);
		void InputFloat3(char const* label, nbl::core::vector3df& value);
		bool Combo(char const* label, int32_t* selectedItemIndex, char const** items, int32_t itemsCount);
		bool Combo(const char* label, int* selectedItemIndex, std::vector<std::string>& values);
		void SliderInt(char const* label, int* value, int minValue, int maxValue);
		void SliderFloat(char const* label, float* value, float minValue, float maxValue);
		void Checkbox(char const* label, bool* value);
		void Spacing();
		void Button(char const* label, std::function<void()> const& onPress);
		void InputText(char const* label, std::string& outValue);
		[[nodiscard]] bool HasFocus();		
		[[nodiscard]] bool IsItemActive();
		[[nodiscard]] bool TreeNode(char const* name);
		void TreePop();

	private:

		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> CreateDescriptorSetLayout();

		void CreatePipeline(video::IGPURenderpass* renderpass, video::IGPUPipelineCache* pipelineCache);
		void CreateFontTexture(video::IGPUCommandBuffer* cmdBuffer, video::IQueue* queue);
		void UpdateDescriptorSets();
		void createSystem();
		void CreateFontSampler();
		void CreateDescriptorPool();
		void HandleMouseEvents(float mousePosX, float mousePosY, size_t mouseEventsCount, ui::SMouseEvent const * mouseEvents) const;

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
		std::vector<core::smart_refctd_ptr<video::IGPUBuffer>> m_vertexBuffers, m_indexBuffers;
		bool hasFocus = false;

		// TODO: Use a signal class instead like Signal<> UIRecordSignal{};
		struct Subscriber {
			int id = -1;
			std::function<void()> listener = nullptr;
		};
		std::vector<Subscriber> m_subscribers{};

};
}

#endif	// NBL_EXT_IMGUI_UI_H
