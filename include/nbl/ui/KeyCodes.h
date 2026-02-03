#ifndef _NBL_UI_KEYCODES_H_INCLUDED_
#define _NBL_UI_KEYCODES_H_INCLUDED_

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

constexpr E_KEY_CODE stringToKeyCode(std::string_view str)
{
	if (str == "BACKSPACE") return EKC_BACKSPACE;
	if (str == "TAB") return EKC_TAB;
	if (str == "CLEAR") return EKC_CLEAR;
	if (str == "ENTER") return EKC_ENTER;
	if (str == "LEFT_SHIFT") return EKC_LEFT_SHIFT;
	if (str == "RIGHT_SHIFT") return EKC_RIGHT_SHIFT;
	if (str == "LEFT_CONTROL") return EKC_LEFT_CONTROL;
	if (str == "RIGHT_CONTROL") return EKC_RIGHT_CONTROL;
	if (str == "LEFT_ALT") return EKC_LEFT_ALT;
	if (str == "RIGHT_ALT") return EKC_RIGHT_ALT;
	if (str == "PAUSE") return EKC_PAUSE;
	if (str == "CAPS_LOCK") return EKC_CAPS_LOCK;
	if (str == "ESCAPE") return EKC_ESCAPE;
	if (str == "SPACE") return EKC_SPACE;
	if (str == "PAGE_UP") return EKC_PAGE_UP;
	if (str == "PAGE_DOWN") return EKC_PAGE_DOWN;
	if (str == "END") return EKC_END;
	if (str == "HOME") return EKC_HOME;
	if (str == "LEFT_ARROW") return EKC_LEFT_ARROW;
	if (str == "RIGHT_ARROW") return EKC_RIGHT_ARROW;
	if (str == "DOWN_ARROW") return EKC_DOWN_ARROW;
	if (str == "UP_ARROW") return EKC_UP_ARROW;
	if (str == "SELECT") return EKC_SELECT;
	if (str == "PRINT") return EKC_PRINT;
	if (str == "EXECUTE") return EKC_EXECUTE;
	if (str == "PRINT_SCREEN") return EKC_PRINT_SCREEN;
	if (str == "INSERT") return EKC_INSERT;
	if (str == "DELETE") return EKC_DELETE;
	if (str == "HELP") return EKC_HELP;
	if (str == "LEFT_WIN") return EKC_LEFT_WIN;
	if (str == "RIGHT_WIN") return EKC_RIGHT_WIN;
	if (str == "APPS") return EKC_APPS;
	if (str == "COMMA") return EKC_COMMA;
	if (str == "PERIOD") return EKC_PERIOD;
	if (str == "SEMICOLON") return EKC_SEMICOLON;
	if (str == "OPEN_BRACKET") return EKC_OPEN_BRACKET;
	if (str == "CLOSE_BRACKET") return EKC_CLOSE_BRACKET;
	if (str == "BACKSLASH") return EKC_BACKSLASH;
	if (str == "APOSTROPHE") return EKC_APOSTROPHE;
	if (str == "ADD") return EKC_ADD;
	if (str == "SUBTRACT") return EKC_SUBTRACT;
	if (str == "MULTIPLY") return EKC_MULTIPLY;
	if (str == "DIVIDE") return EKC_DIVIDE;

	if (str == "A" || str == "a") return EKC_A;
	if (str == "B" || str == "b") return EKC_B;
	if (str == "C" || str == "c") return EKC_C;
	if (str == "D" || str == "d") return EKC_D;
	if (str == "E" || str == "e") return EKC_E;
	if (str == "F" || str == "f") return EKC_F;
	if (str == "G" || str == "g") return EKC_G;
	if (str == "H" || str == "h") return EKC_H;
	if (str == "I" || str == "i") return EKC_I;
	if (str == "J" || str == "j") return EKC_J;
	if (str == "K" || str == "k") return EKC_K;
	if (str == "L" || str == "l") return EKC_L;
	if (str == "M" || str == "m") return EKC_M;
	if (str == "N" || str == "n") return EKC_N;
	if (str == "O" || str == "o") return EKC_O;
	if (str == "P" || str == "p") return EKC_P;
	if (str == "Q" || str == "q") return EKC_Q;
	if (str == "R" || str == "r") return EKC_R;
	if (str == "S" || str == "s") return EKC_S;
	if (str == "T" || str == "t") return EKC_T;
	if (str == "U" || str == "u") return EKC_U;
	if (str == "V" || str == "v") return EKC_V;
	if (str == "W" || str == "w") return EKC_W;
	if (str == "X" || str == "x") return EKC_X;
	if (str == "Y" || str == "y") return EKC_Y;
	if (str == "Z" || str == "z") return EKC_Z;

	if (str == "0") return EKC_0;
	if (str == "1") return EKC_1;
	if (str == "2") return EKC_2;
	if (str == "3") return EKC_3;
	if (str == "4") return EKC_4;
	if (str == "5") return EKC_5;
	if (str == "6") return EKC_6;
	if (str == "7") return EKC_7;
	if (str == "8") return EKC_8;
	if (str == "9") return EKC_9;

	if (str == "F1") return EKC_F1;
	if (str == "F2") return EKC_F2;
	if (str == "F3") return EKC_F3;
	if (str == "F4") return EKC_F4;
	if (str == "F5") return EKC_F5;
	if (str == "F6") return EKC_F6;
	if (str == "F7") return EKC_F7;
	if (str == "F8") return EKC_F8;
	if (str == "F9") return EKC_F9;
	if (str == "F10") return EKC_F10;
	if (str == "F11") return EKC_F11;
	if (str == "F12") return EKC_F12;
	if (str == "F13") return EKC_F13;
	if (str == "F14") return EKC_F14;
	if (str == "F15") return EKC_F15;
	if (str == "F16") return EKC_F16;
	if (str == "F17") return EKC_F17;
	if (str == "F18") return EKC_F18;
	if (str == "F19") return EKC_F19;
	if (str == "F20") return EKC_F20;
	if (str == "F21") return EKC_F21;
	if (str == "F22") return EKC_F22;
	if (str == "F23") return EKC_F23;
	if (str == "F24") return EKC_F24;

	if (str == "NUMPAD_0") return EKC_NUMPAD_0;
	if (str == "NUMPAD_1") return EKC_NUMPAD_1;
	if (str == "NUMPAD_2") return EKC_NUMPAD_2;
	if (str == "NUMPAD_3") return EKC_NUMPAD_3;
	if (str == "NUMPAD_4") return EKC_NUMPAD_4;
	if (str == "NUMPAD_5") return EKC_NUMPAD_5;
	if (str == "NUMPAD_6") return EKC_NUMPAD_6;
	if (str == "NUMPAD_7") return EKC_NUMPAD_7;
	if (str == "NUMPAD_8") return EKC_NUMPAD_8;
	if (str == "NUMPAD_9") return EKC_NUMPAD_9;

	if (str == "NUM_LOCK") return EKC_NUM_LOCK;
	if (str == "SCROLL_LOCK") return EKC_SCROLL_LOCK;

	if (str == "VOLUME_MUTE") return EKC_VOLUME_MUTE;
	if (str == "VOLUME_UP") return EKC_VOLUME_UP;
	if (str == "VOLUME_DOWN") return EKC_VOLUME_DOWN;

	return EKC_NONE;
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
	if (str == "LEFT_BUTTON") return EMC_LEFT_BUTTON;
	if (str == "RIGHT_BUTTON") return EMC_RIGHT_BUTTON;
	if (str == "MIDDLE_BUTTON") return EMC_MIDDLE_BUTTON;
	if (str == "BUTTON_4") return EMC_BUTTON_4;
	if (str == "BUTTON_5") return EMC_BUTTON_5;
	if (str == "VERTICAL_POSITIVE_SCROLL") return EMC_VERTICAL_POSITIVE_SCROLL;
	if (str == "VERTICAL_NEGATIVE_SCROLL") return EMC_VERTICAL_NEGATIVE_SCROLL;
	if (str == "HORIZONTAL_POSITIVE_SCROLL") return EMC_HORIZONTAL_POSITIVE_SCROLL;
	if (str == "HORIZONTAL_NEGATIVE_SCROLL") return EMC_HORIZONTAL_NEGATIVE_SCROLL;
	if (str == "RELATIVE_POSITIVE_MOVEMENT_X") return EMC_RELATIVE_POSITIVE_MOVEMENT_X;
	if (str == "RELATIVE_POSITIVE_MOVEMENT_Y") return EMC_RELATIVE_POSITIVE_MOVEMENT_Y;
	if (str == "RELATIVE_NEGATIVE_MOVEMENT_X") return EMC_RELATIVE_NEGATIVE_MOVEMENT_X;
	if (str == "RELATIVE_NEGATIVE_MOVEMENT_Y") return EMC_RELATIVE_NEGATIVE_MOVEMENT_Y;

	return EMC_NONE;
}

}
#endif
