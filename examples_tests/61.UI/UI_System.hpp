#pragma once

#include <nabla.h>

#include <glm/vec3.hpp>

#include "../common/CommonAPI.h"

namespace UI_System
{

	// TODO Support for custom font
	void Init(
		nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> const& device,
		float screenWidth, float screenHeight,
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass>& renderPass,
		nbl::video::IGPUPipelineCache* pipelineCache,
		nbl::video::IGPUObjectFromAssetConverter::SParams& cpu2GpuParams,
		nbl::core::smart_refctd_ptr<CommonAPI::InputSystem>& inputSystem
	);

	bool Render(float deltaTimeInSec, nbl::video::IGPUCommandBuffer& commandBuffer);

	void Update(float deltaTimeInSec);

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

	void InputFloat3(char const* label, glm::vec3& value);

	bool Combo(
		char const* label,
		int32_t* selectedItemIndex,
		char const** items,
		int32_t itemsCount
	);

	bool Combo(
		const char* label,
		int* selectedItemIndex,
		std::vector<std::string>& values
	);

	void SliderInt(
		char const* label,
		int* value,
		int minValue,
		int maxValue
	);

	void SliderFloat(
		char const* label,
		float* value,
		float minValue,
		float maxValue
	);

	void Checkbox(
		char const* label,
		bool* value
	);

	void Spacing();

	void Button(
		char const* label,
		std::function<void()> const& onPress
	);

	void InputText(
		char const* label,
		std::string& outValue
	);

	[[nodiscard]]
	bool HasFocus();

	void Shutdown();

	[[nodiscard]]
	bool IsItemActive();

	[[nodiscard]]
	bool TreeNode(char const* name);

	void TreePop();

	//#ifdef __ANDROID__
	//    void SetAndroidApp(android_app * pApp);
	//#endif

}

namespace UI = UI_System;
