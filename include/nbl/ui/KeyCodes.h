
namespace nbl::ui
{
	enum E_KEYCODE
	{
		// TODO
	};
	enum E_KEYMODIFIER
	{

	};
	struct SKeyInfo
	{
		E_KEYCODE keyCode;
		E_KEYMODIFIER keyModifier;
	};

	enum E_MOUSEBUTTON
	{
		EMB_LEFT_BUTTON,   // 
		EMB_RIGHT_BUTTON,  // Or should they all be EMB_BUTTON_NUMBER??
		EMB_MIDDLE_BUTTON, //
		EMB_BUTTON_4,
		EMB_BUTTON_5
	};
};