#ifndef _NBL_UI_I_GRAPHICAL_APPLICATION_FRAMEWORK_H_INCLUDED_
#define _NBL_UI_I_GRAPHICAL_APPLICATION_FRAMEWORK_H_INCLUDED_

namespace nbl::ui
{
	class IGraphicalApplicationFramework
	{
	public:
		virtual void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& window) = 0;
		virtual nbl::ui::IWindow* getWindow() = 0;
	};
}
#endif