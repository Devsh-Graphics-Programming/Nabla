#ifndef __IRR_I_CPU_IMAGE_VIEW_H_INCLUDED__
#define __IRR_I_CPU_IMAGE_VIEW_H_INCLUDED__

#include "irr/asset/IAsset.h"
#include "irr/asset/ICPUImage.h"
#include "irr/asset/IImageView.h"

namespace irr
{
namespace asset
{

class ICPUImageView final : public IImageView<ICPUImage>, public IAsset
{
	public:
		ICPUImageView(SCreationParams&& _params) : IImageView<ICPUImage>(std::move(_params)) {}

		//!
		size_t conservativeSizeEstimate() const override
		{
			return sizeof(uint32_t)*3u+sizeof(void*)+sizeof(SComponentMapping);
		}
		//!
		void convertToDummyObject() override { }
		//!
		IAsset::E_TYPE getAssetType() const override { return ET_IMAGE_VIEW; }


		//!
		SComponentMapping&	getComponents() { return params.components; }

	protected:
		virtual ~ICPUImageView() = default;
};

}
}

#endif