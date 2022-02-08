#ifndef C_CLIPBOARD_MANAGER_WIN32
#define C_CLIPBOARD_MANAGER_WIN32
#include "nbl/ui/IClipboardManager.h"
#include <WinUser.h>

namespace nbl::ui
{
class CClipboardManagerWin32 final : public IClipboardManager
{
	using base_t = IClipboardManager;
public:
	CClipboardManagerWin32() = default;
	virtual std::string getClipboardText() override;
	virtual bool setClipboardText(const std::string_view& data) override;
	//virtual core::smart_refctd_ptr<asset::ICPUImage> getClipboardImage() override;
	//virtual bool setClipboardImage(asset::ICPUImage* image, asset::ICPUImage::SImageCopy data) override;
private:
	HGLOBAL* CPUImageToClipboardImage(asset::ICPUImage* image);
};

}


#endif