#ifndef __IRR_I_IMAGE_WRITER_H_INCLUDED__
#define __IRR_I_IMAGE_WRITER_H_INCLUDED__

#include "IImage.h"
#include "irr/core/core.h"

#include "irr/asset/IAssetWriter.h"
#include "irr/asset/ICPUImageView.h"
#include "irr/asset/IImageAssetHandlerBase.h"

#include "irr/asset/filters/CFlattenRegionsImageFilter.h"

namespace irr
{
namespace asset
{

class IImageWriter : public IAssetWriter, public IImageAssetHandlerBase
{
	public:

	protected:

		IImageWriter() = default;
		virtual ~IImageWriter() = 0;

	private:
};

}
}

#endif // __IRR_I_IMAGE_WRITER_H_INCLUDED__
