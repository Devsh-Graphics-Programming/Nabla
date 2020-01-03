#ifndef __IRR_I_IMAGE_ASSET_HANDLER_BASE_H_INCLUDED__
#define __IRR_I_IMAGE_ASSET_HANDLER_BASE_H_INCLUDED__

#include "irr/core/core.h"

namespace irr
{
	namespace asset
	{

		class IImageAssetHandlerBase : public virtual core::IReferenceCounted
		{
			public:

			protected:

				IImageAssetHandlerBase() = default;
				virtual ~IImageAssetHandlerBase() = 0;

				static const uint32_t MAX_PITCH_ALIGNMENT = 8u;										             // OpenGL cannot transfer rows with arbitrary padding
				static inline uint32_t calcPitchInBlocks(uint32_t width, uint32_t blockByteSize)                 // try with largest alignment first
				{
					auto rowByteSize = width * blockByteSize;
					for (uint32_t _alignment = MAX_PITCH_ALIGNMENT; _alignment > 1u; _alignment >>= 1u)
					{
						auto paddedSize = core::alignUp(rowByteSize, _alignment);
						if (paddedSize % blockByteSize)
							continue;
						return paddedSize / blockByteSize;
					}
					return width;
				}

			private:
		};

	}
}

#endif // __IRR_I_IMAGE_ASSET_HANDLER_BASE_H_INCLUDED__
