#ifndef __IRR_I_IMAGE_LOADER_H_INCLUDED__
#define __IRR_I_IMAGE_LOADER_H_INCLUDED__

#include "irr/core/core.h"

#include "irr/asset/IAssetLoader.h"
#include "irr/asset/IImageAssetHandlerBase.h"
#include "irr/asset/ICPUImageView.h"

namespace irr
{
namespace asset
{

class IImageLoader : public IAssetLoader, public IImageAssetHandlerBase
{
	public:

	protected:

		IImageLoader() = default;
		virtual ~IImageLoader() = 0;

	private:
};

}
}

#endif // __IRR_I_IMAGE_LOADER_H_INCLUDED__
