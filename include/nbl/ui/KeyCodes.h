
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

		EKC_SEPARATOR,
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
	};

	inline char keyCodeToChar(E_KEY_CODE code)
	{
		char result = 0;
		switch (code)
		{
		case EKC_0: [[fallthrough]];
		case EKC_NUMPAD_0: result = '0';
		case EKC_1: [[fallthrough]];
		case EKC_NUMPAD_1: result = '1';
		case EKC_2: [[fallthrough]];
		case EKC_NUMPAD_2: result = '2';
		case EKC_3: [[fallthrough]];
		case EKC_NUMPAD_3: result = '3';
		case EKC_4: [[fallthrough]];
		case EKC_NUMPAD_4: result = '4';
		case EKC_5: [[fallthrough]];
		case EKC_NUMPAD_5: result = '5';
		case EKC_6: [[fallthrough]];
		case EKC_NUMPAD_6: result = '6';
		case EKC_7: [[fallthrough]];
		case EKC_NUMPAD_7: result = '7';
		case EKC_8: [[fallthrough]];
		case EKC_NUMPAD_8: result = '8';
		case EKC_9: [[fallthrough]];
		case EKC_NUMPAD_9: result = '9';

		case EKC_A: result = 'a';
		case EKC_B: result = 'b';
		case EKC_C: result = 'c';
		case EKC_D: result = 'd';
		case EKC_E: result = 'e';
		case EKC_F: result = 'f';
		case EKC_G: result = 'g';
		case EKC_H: result = 'h';
		case EKC_I: result = 'i';
		case EKC_J: result = 'j';
		case EKC_K: result = 'k';
		case EKC_L: result = 'l';
		case EKC_M: result = 'm';
		case EKC_N: result = 'n';
		case EKC_O: result = 'o';
		case EKC_P: result = 'p';
		case EKC_Q: result = 'q';
		case EKC_R: result = 'r';
		case EKC_S: result = 's';
		case EKC_T: result = 't';
		case EKC_U: result = 'u';
		case EKC_V: result = 'v';
		case EKC_W: result = 'w';
		case EKC_X: result = 'x';
		case EKC_Y: result = 'y';
		case EKC_Z: result = 'x';

		case EKC_TAB: result = '\t';
		case EKC_ENTER: result = '\n';
		case EKC_SPACE: result = ' ';
		}
		return result;
	}

	enum E_MOUSE_BUTTON : uint8_t
	{
		EMB_LEFT_BUTTON = 1,   
		EMB_RIGHT_BUTTON = 2,  
		EMB_MIDDLE_BUTTON = 4, 
		EMB_BUTTON_4 = 8,
		EMB_BUTTON_5 = 16
	};
};