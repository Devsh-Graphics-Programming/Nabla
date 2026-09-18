#ifndef _NBL_EXT_CAMERAS_C_INPUT_CODE_NAMES_HPP_
#define _NBL_EXT_CAMERAS_C_INPUT_CODE_NAMES_HPP_

#include <array>
#include <string_view>

#include "nbl/ui/KeyCodes.h"

// Stable string names for key codes and mouse buttons, used by persisted camera bindings.
// These used to live in `nbl/ui/KeyCodes.h` but nothing in `nbl::ui` produces or consumes them, only the camera extension does.

namespace nbl::ext::cameras
{

namespace impl
{

template<typename Code>
struct SNamedCode final
{
	std::string_view name;
	Code code;
};

template<typename Code, size_t N>
constexpr Code lookupNamedCode(std::string_view str, const std::array<SNamedCode<Code>, N>& table, const Code fallback)
{
	for (const auto& entry : table)
	{
		if (str == entry.name)
			return entry.code;
	}

	return fallback;
}

template<typename Code, size_t N>
constexpr std::string_view lookupCodeName(const Code code, const std::array<SNamedCode<Code>, N>& table, const std::string_view fallback)
{
	for (const auto& entry : table)
	{
		if (code == entry.code)
			return entry.name;
	}

	return fallback;
}

constexpr char asciiToUpper(const char c)
{
	return (c >= 'a' && c <= 'z') ? static_cast<char>(c - ('a' - 'A')) : c;
}

// single character key codes (`EKC_A`..`EKC_Z`, `EKC_0`..`EKC_9`) equal their ASCII value, this string provides stable storage for their names
inline constexpr std::string_view SingleCharacterKeyNames = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

inline constexpr auto NamedKeyCodes = std::to_array<SNamedCode<ui::E_KEY_CODE>>({
	{ "BACKSPACE", ui::E_KEY_CODE::EKC_BACKSPACE },
	{ "TAB", ui::E_KEY_CODE::EKC_TAB },
	{ "CLEAR", ui::E_KEY_CODE::EKC_CLEAR },
	{ "ENTER", ui::E_KEY_CODE::EKC_ENTER },
	{ "LEFT_SHIFT", ui::E_KEY_CODE::EKC_LEFT_SHIFT },
	{ "RIGHT_SHIFT", ui::E_KEY_CODE::EKC_RIGHT_SHIFT },
	{ "LEFT_CONTROL", ui::E_KEY_CODE::EKC_LEFT_CONTROL },
	{ "RIGHT_CONTROL", ui::E_KEY_CODE::EKC_RIGHT_CONTROL },
	{ "LEFT_ALT", ui::E_KEY_CODE::EKC_LEFT_ALT },
	{ "RIGHT_ALT", ui::E_KEY_CODE::EKC_RIGHT_ALT },
	{ "PAUSE", ui::E_KEY_CODE::EKC_PAUSE },
	{ "CAPS_LOCK", ui::E_KEY_CODE::EKC_CAPS_LOCK },
	{ "ESCAPE", ui::E_KEY_CODE::EKC_ESCAPE },
	{ "SPACE", ui::E_KEY_CODE::EKC_SPACE },
	{ "PAGE_UP", ui::E_KEY_CODE::EKC_PAGE_UP },
	{ "PAGE_DOWN", ui::E_KEY_CODE::EKC_PAGE_DOWN },
	{ "END", ui::E_KEY_CODE::EKC_END },
	{ "HOME", ui::E_KEY_CODE::EKC_HOME },
	{ "LEFT_ARROW", ui::E_KEY_CODE::EKC_LEFT_ARROW },
	{ "RIGHT_ARROW", ui::E_KEY_CODE::EKC_RIGHT_ARROW },
	{ "DOWN_ARROW", ui::E_KEY_CODE::EKC_DOWN_ARROW },
	{ "UP_ARROW", ui::E_KEY_CODE::EKC_UP_ARROW },
	{ "SELECT", ui::E_KEY_CODE::EKC_SELECT },
	{ "PRINT", ui::E_KEY_CODE::EKC_PRINT },
	{ "EXECUTE", ui::E_KEY_CODE::EKC_EXECUTE },
	{ "PRINT_SCREEN", ui::E_KEY_CODE::EKC_PRINT_SCREEN },
	{ "INSERT", ui::E_KEY_CODE::EKC_INSERT },
	{ "DELETE", ui::E_KEY_CODE::EKC_DELETE },
	{ "HELP", ui::E_KEY_CODE::EKC_HELP },
	{ "LEFT_WIN", ui::E_KEY_CODE::EKC_LEFT_WIN },
	{ "RIGHT_WIN", ui::E_KEY_CODE::EKC_RIGHT_WIN },
	{ "APPS", ui::E_KEY_CODE::EKC_APPS },
	{ "COMMA", ui::E_KEY_CODE::EKC_COMMA },
	{ "PERIOD", ui::E_KEY_CODE::EKC_PERIOD },
	{ "SEMICOLON", ui::E_KEY_CODE::EKC_SEMICOLON },
	{ "OPEN_BRACKET", ui::E_KEY_CODE::EKC_OPEN_BRACKET },
	{ "CLOSE_BRACKET", ui::E_KEY_CODE::EKC_CLOSE_BRACKET },
	{ "BACKSLASH", ui::E_KEY_CODE::EKC_BACKSLASH },
	{ "APOSTROPHE", ui::E_KEY_CODE::EKC_APOSTROPHE },
	{ "ADD", ui::E_KEY_CODE::EKC_ADD },
	{ "SUBTRACT", ui::E_KEY_CODE::EKC_SUBTRACT },
	{ "MULTIPLY", ui::E_KEY_CODE::EKC_MULTIPLY },
	{ "DIVIDE", ui::E_KEY_CODE::EKC_DIVIDE },
	{ "F1", ui::E_KEY_CODE::EKC_F1 },
	{ "F2", ui::E_KEY_CODE::EKC_F2 },
	{ "F3", ui::E_KEY_CODE::EKC_F3 },
	{ "F4", ui::E_KEY_CODE::EKC_F4 },
	{ "F5", ui::E_KEY_CODE::EKC_F5 },
	{ "F6", ui::E_KEY_CODE::EKC_F6 },
	{ "F7", ui::E_KEY_CODE::EKC_F7 },
	{ "F8", ui::E_KEY_CODE::EKC_F8 },
	{ "F9", ui::E_KEY_CODE::EKC_F9 },
	{ "F10", ui::E_KEY_CODE::EKC_F10 },
	{ "F11", ui::E_KEY_CODE::EKC_F11 },
	{ "F12", ui::E_KEY_CODE::EKC_F12 },
	{ "F13", ui::E_KEY_CODE::EKC_F13 },
	{ "F14", ui::E_KEY_CODE::EKC_F14 },
	{ "F15", ui::E_KEY_CODE::EKC_F15 },
	{ "F16", ui::E_KEY_CODE::EKC_F16 },
	{ "F17", ui::E_KEY_CODE::EKC_F17 },
	{ "F18", ui::E_KEY_CODE::EKC_F18 },
	{ "F19", ui::E_KEY_CODE::EKC_F19 },
	{ "F20", ui::E_KEY_CODE::EKC_F20 },
	{ "F21", ui::E_KEY_CODE::EKC_F21 },
	{ "F22", ui::E_KEY_CODE::EKC_F22 },
	{ "F23", ui::E_KEY_CODE::EKC_F23 },
	{ "F24", ui::E_KEY_CODE::EKC_F24 },
	{ "NUMPAD_0", ui::E_KEY_CODE::EKC_NUMPAD_0 },
	{ "NUMPAD_1", ui::E_KEY_CODE::EKC_NUMPAD_1 },
	{ "NUMPAD_2", ui::E_KEY_CODE::EKC_NUMPAD_2 },
	{ "NUMPAD_3", ui::E_KEY_CODE::EKC_NUMPAD_3 },
	{ "NUMPAD_4", ui::E_KEY_CODE::EKC_NUMPAD_4 },
	{ "NUMPAD_5", ui::E_KEY_CODE::EKC_NUMPAD_5 },
	{ "NUMPAD_6", ui::E_KEY_CODE::EKC_NUMPAD_6 },
	{ "NUMPAD_7", ui::E_KEY_CODE::EKC_NUMPAD_7 },
	{ "NUMPAD_8", ui::E_KEY_CODE::EKC_NUMPAD_8 },
	{ "NUMPAD_9", ui::E_KEY_CODE::EKC_NUMPAD_9 },
	{ "NUM_LOCK", ui::E_KEY_CODE::EKC_NUM_LOCK },
	{ "SCROLL_LOCK", ui::E_KEY_CODE::EKC_SCROLL_LOCK },
	{ "VOLUME_MUTE", ui::E_KEY_CODE::EKC_VOLUME_MUTE },
	{ "VOLUME_UP", ui::E_KEY_CODE::EKC_VOLUME_UP },
	{ "VOLUME_DOWN", ui::E_KEY_CODE::EKC_VOLUME_DOWN }
});

// one table for both directions so the two mappings cannot drift apart
inline constexpr auto NamedMouseButtons = std::to_array<SNamedCode<ui::E_MOUSE_BUTTON>>({
	{ "LEFT_BUTTON", ui::EMB_LEFT_BUTTON },
	{ "RIGHT_BUTTON", ui::EMB_RIGHT_BUTTON },
	{ "MIDDLE_BUTTON", ui::EMB_MIDDLE_BUTTON },
	{ "BUTTON_4", ui::EMB_BUTTON_4 },
	{ "BUTTON_5", ui::EMB_BUTTON_5 }
});

} // namespace impl

constexpr ui::E_KEY_CODE stringToKeyCode(std::string_view str)
{
	if (str.size() == 1u)
	{
		const char upper = impl::asciiToUpper(str.front());
		if ((upper >= 'A' && upper <= 'Z') || (upper >= '0' && upper <= '9'))
			return static_cast<ui::E_KEY_CODE>(upper);
	}

	return impl::lookupNamedCode(str, impl::NamedKeyCodes, ui::E_KEY_CODE::EKC_NONE);
}

constexpr std::string_view keyCodeToString(const ui::E_KEY_CODE code)
{
	const auto single = impl::SingleCharacterKeyNames.find(static_cast<char>(code));
	if (single != std::string_view::npos)
		return impl::SingleCharacterKeyNames.substr(single, 1u);

	return impl::lookupCodeName(code, impl::NamedKeyCodes, "NONE");
}

/// @brief Mouse button named by `str`, or `EMB_COUNT` when no button has that name.
constexpr ui::E_MOUSE_BUTTON stringToMouseButton(std::string_view str)
{
	return impl::lookupNamedCode(str, impl::NamedMouseButtons, ui::EMB_COUNT);
}

constexpr std::string_view mouseButtonToString(const ui::E_MOUSE_BUTTON button)
{
	return impl::lookupCodeName(button, impl::NamedMouseButtons, "NONE");
}

} // namespace nbl::ext::cameras

#endif // _NBL_EXT_CAMERAS_C_INPUT_CODE_NAMES_HPP_
