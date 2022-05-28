#ifndef _NBL_UI_C_CLIPBOARD_MANAGER_WIN32_INCLUDED_
#define _NBL_UI_C_CLIPBOARD_MANAGER_WIN32_INCLUDED_

#include "nbl/ui/IClipboardManager.h"

#ifdef _NBL_PLATFORM_WINDOWS_
namespace nbl::ui
{

class NBL_API2 CClipboardManagerWin32 final : public IClipboardManager
{
		using base_t = IClipboardManager;
	public:
		inline CClipboardManagerWin32() = default;

		virtual std::string getClipboardText() override;
		virtual bool setClipboardText(const std::string_view& data) override;

		//virtual core::smart_refctd_ptr<asset::ICPUImage> getClipboardImage() override;
		//virtual bool setClipboardImage(asset::ICPUImage* image, asset::ICPUImage::SImageCopy data) override;
	private:
		HGLOBAL* CPUImageToClipboardImage(asset::ICPUImage* image);
};

}


#endif
#endif