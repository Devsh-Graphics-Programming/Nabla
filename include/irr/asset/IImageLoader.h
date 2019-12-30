#ifndef __IRR_I_IMAGE_LOADER_H_INCLUDED__
#define __IRR_I_IMAGE_LOADER_H_INCLUDED__

#include "irr/core/core.h"
#include "IAssetLoader.h"
#include "IImageAssetHandlerBase.h"

namespace irr
{
	namespace asset
	{

		class IImageLoader : public IAssetLoader, public IImageAssetHandlerBase
		{
			public:

				IImageLoader() = default;
				virtual ~IImageLoader() = 0;

			protected:

			private:
		};
	}
}

#endif // __IRR_I_IMAGE_LOADER_H_INCLUDED__
