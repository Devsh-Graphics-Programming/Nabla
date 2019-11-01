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
		enum E_CREATE_FLAGS
		{
		};
		enum E_TYPE
		{
			ET_1D = 0,
			ET_2D,
			ET_3D,
			ET_CUBE_MAP,
			ET_1D_ARRAY,
			ET_2D_ARRAY,
			ET_CUBE_MAP_ARRAY,
			ET_COUNT
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
		struct SCreationParams
		{
			E_CREATE_FLAGS						flags;
			core::smart_refctd_ptr<ImageType>	image;
			E_TYPE								viewType;
			E_FORMAT							format;
			SComponentMapping					components;
			IImage::SSubresourceRange			subresourceRange;
		};
		//!
		inline static bool validateCreationParameters(const SCreationParams& _params)
		{
			if ()
				return false;

			return true;
		}

		//!
		E_CATEGORY	getTypeCategory() const override { return EC_IMAGE; }


		//!
		const SCreationParams&	getCreationParameters() const { return params; }

	protected:
		IImageView() : params{static_cast<E_CREATE_FLAGS>(0u),nullptr,ET_COUNT,EF_UNKNOWN,{}} {}
		IImageView(SCreationParams&& _params) : params(_params) {}
		virtual ~IImageView() = default;

		SCreationParams params;
};

}
}

#endif