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
		static core::smart_refctd_ptr<ICPUImageView> create(SCreationParams&& params)
		{
			if (!validateCreationParameters(params))
				return nullptr;

			return core::make_smart_refctd_ptr<ICPUImageView>(std::move(params));
		}
		ICPUImageView(SCreationParams&& _params) : IImageView<ICPUImage>(std::move(_params)) {}

		//!
		size_t conservativeSizeEstimate() const override
		{
			return sizeof(SCreationParams);
		}
		//!
		void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
		{
			if (referenceLevelsBelowToConvert--)
				params.image->convertToDummyObject(referenceLevelsBelowToConvert);
		}
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