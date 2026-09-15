#ifndef _NBL_UI_KEYCODES_H_INCLUDED_
#define _NBL_UI_KEYCODES_H_INCLUDED_

#include <array>
#include <string_view>

namespace nbl::ui
{

enum E_KEY_CODE : uint8_t
{
	EKC_NONE = 0,
	EKC_BACKSPACE,
	EKC_TAB,
	EKC_CLEAR,
	EKC_ENTER,
	EKC_LEFT_SHIFT,
	EKC_RIGHT_SHIFT,
	EKC_LEFT_CONTROL,
	EKC_RIGHT_CONTROL,
	EKC_LEFT_ALT,
	EKC_RIGHT_ALT,
	EKC_PAUSE,
	EKC_CAPS_LOCK,
	EKC_ESCAPE,
	EKC_SPACE,
	EKC_PAGE_UP,
	EKC_PAGE_DOWN,
	EKC_END, 
	EKC_HOME,
	EKC_LEFT_ARROW,
	EKC_RIGHT_ARROW,
	EKC_DOWN_ARROW,
	EKC_UP_ARROW,
	EKC_SELECT,
	EKC_PRINT,
	EKC_EXECUTE,
	EKC_PRINT_SCREEN,
	EKC_INSERT,
	EKC_DELETE,
	EKC_HELP,
		
	EKC_LEFT_WIN,
	EKC_RIGHT_WIN,
	EKC_APPS,

	EKC_COMMA,
	EKC_PERIOD,
	EKC_SEMICOLON,
	EKC_OPEN_BRACKET,
	EKC_CLOSE_BRACKET,
	EKC_BACKSLASH,
	EKC_APOSTROPHE,

	EKC_ADD = '+',
	EKC_SUBTRACT = '-',
	EKC_MULTIPLY = '*',
	EKC_DIVIDE = '/',

	EKC_0 = '0',
	EKC_1,
	EKC_2,
	EKC_3,
	EKC_4,
	EKC_5,
	EKC_6,
	EKC_7,
	EKC_8,
	EKC_9,

	EKC_A = 'A',
	EKC_B,
	EKC_C,
	EKC_D,
	EKC_E,
	EKC_F,
	EKC_G,
	EKC_H,
	EKC_I,
	EKC_J,
	EKC_K,
	EKC_L,
	EKC_M,
	EKC_N,
	EKC_O,
	EKC_P,
	EKC_Q,
	EKC_R,
	EKC_S,
	EKC_T,
	EKC_U,
	EKC_V,
	EKC_W,
	EKC_X,
	EKC_Y,
	EKC_Z,

	EKC_NUMPAD_0,
	EKC_NUMPAD_1,
	EKC_NUMPAD_2,
	EKC_NUMPAD_3,
	EKC_NUMPAD_4,
	EKC_NUMPAD_5,
	EKC_NUMPAD_6,
	EKC_NUMPAD_7,
	EKC_NUMPAD_8,
	EKC_NUMPAD_9,

	EKC_F1,
	EKC_F2,
	EKC_F3,
	EKC_F4,
	EKC_F5,
	EKC_F6,
	EKC_F7,
	EKC_F8,
	EKC_F9,
	EKC_F10,
	EKC_F11,
	EKC_F12,
	EKC_F13,
	EKC_F14,
	EKC_F15,
	EKC_F16,
	EKC_F17,
	EKC_F18,
	EKC_F19,
	EKC_F20,
	EKC_F21,
	EKC_F22,
	EKC_F23,
	EKC_F24,

	EKC_NUM_LOCK,
	EKC_SCROLL_LOCK,

	EKC_VOLUME_MUTE,
	EKC_VOLUME_UP,
	EKC_VOLUME_DOWN,


	EKC_COUNT,
};

constexpr char keyCodeToChar(E_KEY_CODE code, bool shiftPressed)
{
	char result = 0;
	if (!shiftPressed)
	{
		switch (code)
		{
			case EKC_0: [[fallthrough]];
			case EKC_NUMPAD_0: result = '0'; break;
			case EKC_1: [[fallthrough]];
			case EKC_NUMPAD_1: result = '1'; break;
			case EKC_2: [[fallthrough]];
			case EKC_NUMPAD_2: result = '2'; break;
			case EKC_3: [[fallthrough]];
			case EKC_NUMPAD_3: result = '3'; break;
			case EKC_4: [[fallthrough]];
			case EKC_NUMPAD_4: result = '4'; break;
			case EKC_5: [[fallthrough]];
			case EKC_NUMPAD_5: result = '5'; break;
			case EKC_6: [[fallthrough]];
			case EKC_NUMPAD_6: result = '6'; break;
			case EKC_7: [[fallthrough]];
			case EKC_NUMPAD_7: result = '7'; break;
			case EKC_8: [[fallthrough]];
			case EKC_NUMPAD_8: result = '8'; break;
			case EKC_9: [[fallthrough]];
			case EKC_NUMPAD_9: result = '9'; break;

			case EKC_A: result = 'a'; break;
			case EKC_B: result = 'b'; break;
			case EKC_C: result = 'c'; break;
			case EKC_D: result = 'd'; break;
			case EKC_E: result = 'e'; break;
			case EKC_F: result = 'f'; break;
			case EKC_G: result = 'g'; break;
			case EKC_H: result = 'h'; break;
			case EKC_I: result = 'i'; break;
			case EKC_J: result = 'j'; break;
			case EKC_K: result = 'k'; break;
			case EKC_L: result = 'l'; break;
			case EKC_M: result = 'm'; break;
			case EKC_N: result = 'n'; break;
			case EKC_O: result = 'o'; break;
			case EKC_P: result = 'p'; break;
			case EKC_Q: result = 'q'; break;
			case EKC_R: result = 'r'; break;
			case EKC_S: result = 's'; break;
			case EKC_T: result = 't'; break;
			case EKC_U: result = 'u'; break;
			case EKC_V: result = 'v'; break;
			case EKC_W: result = 'w'; break;
			case EKC_X: result = 'x'; break;
			case EKC_Y: result = 'y'; break;
			case EKC_Z: result = 'z'; break;

			case EKC_TAB: result = '\t'; break;
			case EKC_ENTER: result = '\n'; break;
			case EKC_SPACE: result = ' '; break;
			case EKC_COMMA: result = ','; break;
			case EKC_PERIOD: result = '.'; break;
			case EKC_SEMICOLON: result = ';'; break;
			case EKC_ADD: result = '='; break;
			case EKC_SUBTRACT: result = '-'; break;
			case EKC_DIVIDE: result = '/'; break;
			case EKC_OPEN_BRACKET: result = '['; break;
			case EKC_CLOSE_BRACKET: result = ']'; break;
			case EKC_BACKSLASH: result = '\\'; break;
			case EKC_APOSTROPHE: result = '\''; break;
		}
	}
	else
	{
		switch (code)
		{
			case EKC_0: result = ')'; break;
			case EKC_1: result = '!'; break;
			case EKC_2: result = '@'; break;
			case EKC_3: result = '#'; break;
			case EKC_4: result = '$'; break;
			case EKC_5: result = '%'; break;
			case EKC_6: result = '^'; break;
			case EKC_7: result = '&'; break;
			case EKC_8: result = '*'; break;
			case EKC_9: result = '('; break;

			case EKC_A: result = 'A'; break;
			case EKC_B: result = 'B'; break;
			case EKC_C: result = 'C'; break;
			case EKC_D: result = 'D'; break;
			case EKC_E: result = 'E'; break;
			case EKC_F: result = 'F'; break;
			case EKC_G: result = 'G'; break;
			case EKC_H: result = 'H'; break;
			case EKC_I: result = 'I'; break;
			case EKC_J: result = 'J'; break;
			case EKC_K: result = 'K'; break;
			case EKC_L: result = 'L'; break;
			case EKC_M: result = 'M'; break;
			case EKC_N: result = 'N'; break;
			case EKC_O: result = 'O'; break;
			case EKC_P: result = 'P'; break;
			case EKC_Q: result = 'Q'; break;
			case EKC_R: result = 'R'; break;
			case EKC_S: result = 'S'; break;
			case EKC_T: result = 'T'; break;
			case EKC_U: result = 'U'; break;
			case EKC_V: result = 'V'; break;
			case EKC_W: result = 'W'; break;
			case EKC_X: result = 'X'; break;
			case EKC_Y: result = 'Y'; break;
			case EKC_Z: result = 'Z'; break;

			case EKC_COMMA: result = '<'; break;
			case EKC_PERIOD: result = '>'; break;
			case EKC_SEMICOLON: result = ':'; break;
			case EKC_ADD: result = '+'; break;
			case EKC_SUBTRACT: result = '_'; break;
			case EKC_DIVIDE: result = '/'; break;
			case EKC_OPEN_BRACKET: result = '{'; break;
			case EKC_CLOSE_BRACKET: result = '}'; break;
			case EKC_BACKSLASH: result = '|'; break;
			case EKC_APOSTROPHE: result = '\"'; break;
		}
	}
	return result;
}

enum E_MOUSE_BUTTON : uint8_t
{
	EMB_LEFT_BUTTON,
	EMB_RIGHT_BUTTON,
	EMB_MIDDLE_BUTTON,
	EMB_BUTTON_4,
	EMB_BUTTON_5,
	EMB_COUNT,
};

// Unambiguous set of "codes" to represent various mouse actions we support with Nabla - equivalent of E_KEY_CODE
enum E_MOUSE_CODE : uint8_t
{
	EMC_NONE = 0,

	// I know its E_MOUSE_BUTTON, this enum *must* be more abstract to standardize mouse
	EMC_LEFT_BUTTON,
	EMC_RIGHT_BUTTON,
	EMC_MIDDLE_BUTTON,
	EMC_BUTTON_4,
	EMC_BUTTON_5,

	// and this is kinda SMouseEvent::E_EVENT_TYPE::EET_SCROLL
	EMC_VERTICAL_POSITIVE_SCROLL,
	EMC_VERTICAL_NEGATIVE_SCROLL,
	EMC_HORIZONTAL_POSITIVE_SCROLL,
	EMC_HORIZONTAL_NEGATIVE_SCROLL,

	// SMouseEvent::E_EVENT_TYPE::EET_MOVEMENT
	EMC_RELATIVE_POSITIVE_MOVEMENT_X,
	EMC_RELATIVE_POSITIVE_MOVEMENT_Y,
	EMC_RELATIVE_NEGATIVE_MOVEMENT_X,
	EMC_RELATIVE_NEGATIVE_MOVEMENT_Y,

	EMC_COUNT,
};

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

constexpr char asciiToUpper(const char c)
{
	return (c >= 'a' && c <= 'z') ? static_cast<char>(c - ('a' - 'A')) : c;
}

static constexpr auto NamedKeyCodes = std::to_array<SNamedCode<E_KEY_CODE>>({
	{ "BACKSPACE", E_KEY_CODE::EKC_BACKSPACE },
	{ "TAB", E_KEY_CODE::EKC_TAB },
	{ "CLEAR", E_KEY_CODE::EKC_CLEAR },
	{ "ENTER", E_KEY_CODE::EKC_ENTER },
	{ "LEFT_SHIFT", E_KEY_CODE::EKC_LEFT_SHIFT },
	{ "RIGHT_SHIFT", E_KEY_CODE::EKC_RIGHT_SHIFT },
	{ "LEFT_CONTROL", E_KEY_CODE::EKC_LEFT_CONTROL },
	{ "RIGHT_CONTROL", E_KEY_CODE::EKC_RIGHT_CONTROL },
	{ "LEFT_ALT", E_KEY_CODE::EKC_LEFT_ALT },
	{ "RIGHT_ALT", E_KEY_CODE::EKC_RIGHT_ALT },
	{ "PAUSE", E_KEY_CODE::EKC_PAUSE },
	{ "CAPS_LOCK", E_KEY_CODE::EKC_CAPS_LOCK },
	{ "ESCAPE", E_KEY_CODE::EKC_ESCAPE },
	{ "SPACE", E_KEY_CODE::EKC_SPACE },
	{ "PAGE_UP", E_KEY_CODE::EKC_PAGE_UP },
	{ "PAGE_DOWN", E_KEY_CODE::EKC_PAGE_DOWN },
	{ "END", E_KEY_CODE::EKC_END },
	{ "HOME", E_KEY_CODE::EKC_HOME },
	{ "LEFT_ARROW", E_KEY_CODE::EKC_LEFT_ARROW },
	{ "RIGHT_ARROW", E_KEY_CODE::EKC_RIGHT_ARROW },
	{ "DOWN_ARROW", E_KEY_CODE::EKC_DOWN_ARROW },
	{ "UP_ARROW", E_KEY_CODE::EKC_UP_ARROW },
	{ "SELECT", E_KEY_CODE::EKC_SELECT },
	{ "PRINT", E_KEY_CODE::EKC_PRINT },
	{ "EXECUTE", E_KEY_CODE::EKC_EXECUTE },
	{ "PRINT_SCREEN", E_KEY_CODE::EKC_PRINT_SCREEN },
	{ "INSERT", E_KEY_CODE::EKC_INSERT },
	{ "DELETE", E_KEY_CODE::EKC_DELETE },
	{ "HELP", E_KEY_CODE::EKC_HELP },
	{ "LEFT_WIN", E_KEY_CODE::EKC_LEFT_WIN },
	{ "RIGHT_WIN", E_KEY_CODE::EKC_RIGHT_WIN },
	{ "APPS", E_KEY_CODE::EKC_APPS },
	{ "COMMA", E_KEY_CODE::EKC_COMMA },
	{ "PERIOD", E_KEY_CODE::EKC_PERIOD },
	{ "SEMICOLON", E_KEY_CODE::EKC_SEMICOLON },
	{ "OPEN_BRACKET", E_KEY_CODE::EKC_OPEN_BRACKET },
	{ "CLOSE_BRACKET", E_KEY_CODE::EKC_CLOSE_BRACKET },
	{ "BACKSLASH", E_KEY_CODE::EKC_BACKSLASH },
	{ "APOSTROPHE", E_KEY_CODE::EKC_APOSTROPHE },
	{ "ADD", E_KEY_CODE::EKC_ADD },
	{ "SUBTRACT", E_KEY_CODE::EKC_SUBTRACT },
	{ "MULTIPLY", E_KEY_CODE::EKC_MULTIPLY },
	{ "DIVIDE", E_KEY_CODE::EKC_DIVIDE },
	{ "F1", E_KEY_CODE::EKC_F1 },
	{ "F2", E_KEY_CODE::EKC_F2 },
	{ "F3", E_KEY_CODE::EKC_F3 },
	{ "F4", E_KEY_CODE::EKC_F4 },
	{ "F5", E_KEY_CODE::EKC_F5 },
	{ "F6", E_KEY_CODE::EKC_F6 },
	{ "F7", E_KEY_CODE::EKC_F7 },
	{ "F8", E_KEY_CODE::EKC_F8 },
	{ "F9", E_KEY_CODE::EKC_F9 },
	{ "F10", E_KEY_CODE::EKC_F10 },
	{ "F11", E_KEY_CODE::EKC_F11 },
	{ "F12", E_KEY_CODE::EKC_F12 },
	{ "F13", E_KEY_CODE::EKC_F13 },
	{ "F14", E_KEY_CODE::EKC_F14 },
	{ "F15", E_KEY_CODE::EKC_F15 },
	{ "F16", E_KEY_CODE::EKC_F16 },
	{ "F17", E_KEY_CODE::EKC_F17 },
	{ "F18", E_KEY_CODE::EKC_F18 },
	{ "F19", E_KEY_CODE::EKC_F19 },
	{ "F20", E_KEY_CODE::EKC_F20 },
	{ "F21", E_KEY_CODE::EKC_F21 },
	{ "F22", E_KEY_CODE::EKC_F22 },
	{ "F23", E_KEY_CODE::EKC_F23 },
	{ "F24", E_KEY_CODE::EKC_F24 },
	{ "NUMPAD_0", E_KEY_CODE::EKC_NUMPAD_0 },
	{ "NUMPAD_1", E_KEY_CODE::EKC_NUMPAD_1 },
	{ "NUMPAD_2", E_KEY_CODE::EKC_NUMPAD_2 },
	{ "NUMPAD_3", E_KEY_CODE::EKC_NUMPAD_3 },
	{ "NUMPAD_4", E_KEY_CODE::EKC_NUMPAD_4 },
	{ "NUMPAD_5", E_KEY_CODE::EKC_NUMPAD_5 },
	{ "NUMPAD_6", E_KEY_CODE::EKC_NUMPAD_6 },
	{ "NUMPAD_7", E_KEY_CODE::EKC_NUMPAD_7 },
	{ "NUMPAD_8", E_KEY_CODE::EKC_NUMPAD_8 },
	{ "NUMPAD_9", E_KEY_CODE::EKC_NUMPAD_9 },
	{ "NUM_LOCK", E_KEY_CODE::EKC_NUM_LOCK },
	{ "SCROLL_LOCK", E_KEY_CODE::EKC_SCROLL_LOCK },
	{ "VOLUME_MUTE", E_KEY_CODE::EKC_VOLUME_MUTE },
	{ "VOLUME_UP", E_KEY_CODE::EKC_VOLUME_UP },
	{ "VOLUME_DOWN", E_KEY_CODE::EKC_VOLUME_DOWN }
});

static constexpr auto NamedMouseCodes = std::to_array<SNamedCode<E_MOUSE_CODE>>({
	{ "LEFT_BUTTON", E_MOUSE_CODE::EMC_LEFT_BUTTON },
	{ "RIGHT_BUTTON", E_MOUSE_CODE::EMC_RIGHT_BUTTON },
	{ "MIDDLE_BUTTON", E_MOUSE_CODE::EMC_MIDDLE_BUTTON },
	{ "BUTTON_4", E_MOUSE_CODE::EMC_BUTTON_4 },
	{ "BUTTON_5", E_MOUSE_CODE::EMC_BUTTON_5 },
	{ "VERTICAL_POSITIVE_SCROLL", E_MOUSE_CODE::EMC_VERTICAL_POSITIVE_SCROLL },
	{ "VERTICAL_NEGATIVE_SCROLL", E_MOUSE_CODE::EMC_VERTICAL_NEGATIVE_SCROLL },
	{ "HORIZONTAL_POSITIVE_SCROLL", E_MOUSE_CODE::EMC_HORIZONTAL_POSITIVE_SCROLL },
	{ "HORIZONTAL_NEGATIVE_SCROLL", E_MOUSE_CODE::EMC_HORIZONTAL_NEGATIVE_SCROLL },
	{ "RELATIVE_POSITIVE_MOVEMENT_X", E_MOUSE_CODE::EMC_RELATIVE_POSITIVE_MOVEMENT_X },
	{ "RELATIVE_POSITIVE_MOVEMENT_Y", E_MOUSE_CODE::EMC_RELATIVE_POSITIVE_MOVEMENT_Y },
	{ "RELATIVE_NEGATIVE_MOVEMENT_X", E_MOUSE_CODE::EMC_RELATIVE_NEGATIVE_MOVEMENT_X },
	{ "RELATIVE_NEGATIVE_MOVEMENT_Y", E_MOUSE_CODE::EMC_RELATIVE_NEGATIVE_MOVEMENT_Y }
});

} // namespace impl

constexpr E_KEY_CODE stringToKeyCode(std::string_view str)
{
	if (str.size() == 1u)
	{
		const char upper = impl::asciiToUpper(str.front());
		if (upper >= 'A' && upper <= 'Z')
			return static_cast<E_KEY_CODE>(upper);
		if (upper >= '0' && upper <= '9')
			return static_cast<E_KEY_CODE>(upper);
	}

	return impl::lookupNamedCode(str, impl::NamedKeyCodes, E_KEY_CODE::EKC_NONE);
}

constexpr std::string_view mouseCodeToString(E_MOUSE_CODE code)
{
	switch (code)
	{
	case EMC_LEFT_BUTTON: return "LEFT_BUTTON";
	case EMC_RIGHT_BUTTON: return "RIGHT_BUTTON";
	case EMC_MIDDLE_BUTTON: return "MIDDLE_BUTTON";
	case EMC_BUTTON_4: return "BUTTON_4";
	case EMC_BUTTON_5: return "BUTTON_5";

	case EMC_VERTICAL_POSITIVE_SCROLL: return "VERTICAL_POSITIVE_SCROLL";
	case EMC_VERTICAL_NEGATIVE_SCROLL: return "VERTICAL_NEGATIVE_SCROLL";
	case EMC_HORIZONTAL_POSITIVE_SCROLL: return "HORIZONTAL_POSITIVE_SCROLL";
	case EMC_HORIZONTAL_NEGATIVE_SCROLL: return "HORIZONTAL_NEGATIVE_SCROLL";

	case EMC_RELATIVE_POSITIVE_MOVEMENT_X: return "RELATIVE_POSITIVE_MOVEMENT_X";
	case EMC_RELATIVE_POSITIVE_MOVEMENT_Y: return "RELATIVE_POSITIVE_MOVEMENT_Y";
	case EMC_RELATIVE_NEGATIVE_MOVEMENT_X: return "RELATIVE_NEGATIVE_MOVEMENT_X";
	case EMC_RELATIVE_NEGATIVE_MOVEMENT_Y: return "RELATIVE_NEGATIVE_MOVEMENT_Y";

	default: return "NONE";
	}
}

constexpr E_MOUSE_CODE stringToMouseCode(std::string_view str)
{
	return impl::lookupNamedCode(str, impl::NamedMouseCodes, E_MOUSE_CODE::EMC_NONE);
}

}
#endif
