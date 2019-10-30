#ifndef __IRR_I_IMAGE_VIEW_H_INCLUDED__
#define __IRR_I_IMAGE_VIEW_H_INCLUDED__

#include "irr/asset/IImage.h"

namespace irr
{
namespace asset
{

template<class ImageType>
class IImageView : public IDescriptor
{
	public:
		// no flags for now, yet
		enum E_IMAGE_VIEW_CREATE_FLAGS
		{
		};
		enum E_IMAGE_VIEW_TYPE
		{
			EIVT_1D = 0,
			EIVT_2D,
			EIVT_3D,
			EIVT_CUBE_MAP,
			EIVT_1D_ARRAY,
			EIVT_2D_ARRAY,
			EIVT_CUBE_MAP_ARRAY,
			EIVT_COUNT
		};
		enum E_CUBE_MAP_FACE
		{
			ECMF_POSITIVE_X = 0,
			ECMF_NEGATIVE_X,
			ECMF_POSITIVE_Y,
			ECMF_NEGATIVE_Y,
			ECMF_POSITIVE_Z,
			ECMF_NEGATIVE_Z,
			ECMF_COUNT
		};
		struct SComponentMapping
		{
			enum E_SWIZZLE
			{
				ES_IDENTITY = 0u,
				ES_ZERO		= 1u,
				ES_ONE		= 2u,
				ES_R		= 3u,
				ES_G		= 4u,
				ES_B		= 5u,
				ES_A		= 6u
			};
			E_SWIZZLE r = ES_R;
			E_SWIZZLE g = ES_G;
			E_SWIZZLE b = ES_B;
			E_SWIZZLE a = ES_A;
		};


		//!
		E_CATEGORY	getTypeCategory() const override { return EC_IMAGE; }


		//!
		const E_IMAGE_VIEW_CREATE_FLAGS&	getFlags() const { return flags; }

		//!
		const ImageType*					getImage() const { return image.get(); }

		//!
		const E_IMAGE_VIEW_TYPE&			getViewType() const { return viewType; }

		//!
		const E_FORMAT&						getFormat() const { return format; }

		//!
		const SComponentMapping&			getComponents() const { return components; }

	protected:
		IImageView() = default;
		virtual ~IImageView() = default;

		E_IMAGE_VIEW_CREATE_FLAGS			flags;
		core::smart_refctd_ptr<ImageType>	image;
		E_IMAGE_VIEW_TYPE					viewType;
		E_FORMAT							format;
		SComponentMapping					components;
};

}
}

#endif