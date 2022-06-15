#ifndef __NBL_SYSTEM_I_CURSOR_CONTROL_H_INCLUDED__
#define __NBL_SYSTEM_I_CURSOR_CONTROL_H_INCLUDED__

#include "nbl/core/declarations.h"
#include "nbl/core/decl/Types.h"

#include "nbl/ui/IWindow.h"

namespace nbl::ui
{

class ICursorControl : public core::IReferenceCounted
{
	public:
		struct SPosition
		{
			int32_t x, y;
		};
		struct SRelativePosition
		{
			float x, y;
		};
		NBL_API2 virtual void setVisible(bool visible) = 0;
		NBL_API2 virtual bool isVisible() const = 0;

		// Native OS screen position
		NBL_API2 virtual void setPosition(SPosition pos) = 0;

		NBL_API2 virtual void setRelativePosition(IWindow* window, SRelativePosition pos) = 0;
		
		// Native OS screen position
		NBL_API2 virtual SPosition getPosition() = 0;

		NBL_API2 virtual SRelativePosition getRelativePosition(IWindow* window) = 0;

		NBL_API2 virtual ~ICursorControl() = default;
};

}
#endif