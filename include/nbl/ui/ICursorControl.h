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
		virtual void setVisible(bool visible) = 0;
		virtual bool isVisible() const = 0;
		virtual void setPosition(int32_t x, int32_t y) = 0;
		virtual void setPosition(const core::vector2d<int32_t>& pos) = 0;
		//TODO something instead of core::vector2d ??
		// Native OS screen position
		virtual core::vector2di32_SIMD getPosition() = 0;

		// NDC vulkan-like coordinates
		virtual core::vector2df_SIMD getRelativePosition() = 0;

		virtual ~ICursorControl() = default;
	};
}
#endif