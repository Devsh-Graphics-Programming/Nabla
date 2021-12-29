#ifndef _MNL_UI_KEYCODES_H_INCLUDED_
#define _MNL_UI_KEYCODES_H_INCLUDED_
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

		EKC_SEPARATOR,
		EKC_NUM_LOCK,
		EKC_SCROLL_LOCK,

		EKC_VOLUME_MUTE,
		EKC_VOLUME_UP,
		EKC_VOLUME_DOWN,


		EKC_COUNT,
	};

	inline char keyCodeToChar(E_KEY_CODE code, bool shiftPressed)
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
};
#endif
