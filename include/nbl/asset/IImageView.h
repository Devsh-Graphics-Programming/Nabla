// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_IMAGE_VIEW_H_INCLUDED__
#define __NBL_ASSET_I_IMAGE_VIEW_H_INCLUDED__

#include "nbl/asset/IImage.h"

namespace nbl
{
namespace asset
{
template<class ImageType>
class IImageView : public IDescriptor
{
public:
    _NBL_STATIC_INLINE_CONSTEXPR size_t remaining_mip_levels = ~static_cast<size_t>(0u);
    _NBL_STATIC_INLINE_CONSTEXPR size_t remaining_array_layers = ~static_cast<size_t>(0u);

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
            ES_ZERO = 1u,
            ES_ONE = 2u,
            ES_R = 3u,
            ES_G = 4u,
            ES_B = 5u,
            ES_A = 6u,
            ES_COUNT
        };
        E_SWIZZLE r = ES_R;
        E_SWIZZLE g = ES_G;
        E_SWIZZLE b = ES_B;
        E_SWIZZLE a = ES_A;

        E_SWIZZLE& operator[](const uint32_t ix)
        {
            assert(ix < 4u);
            return (&r)[ix];
        }

        bool operator==(const SComponentMapping& rhs) const
        {
            return r == rhs.r && g == rhs.g && b == rhs.b && a == rhs.a;
        }
        bool operator!=(const SComponentMapping& rhs) const
        {
            return !operator==(rhs);
        }
    };
    struct SCreationParams
    {
        E_CREATE_FLAGS flags = static_cast<E_CREATE_FLAGS>(0);
        core::smart_refctd_ptr<ImageType> image;
        E_TYPE viewType;
        E_FORMAT format;
        SComponentMapping components;
        IImage::SSubresourceRange subresourceRange;
    };
    //!
    inline static bool validateCreationParameters(const SCreationParams& _params)
    {
        if(!_params.image)
            return false;

        if(_params.flags)
            return false;

        const auto& imgParams = _params.image->getCreationParameters();
        bool mutableFormat = imgParams.flags & IImage::ECF_MUTABLE_FORMAT_BIT;
        if(mutableFormat)
        {
            //if (!isFormatCompatible(_params.format,imgParams.format))
            //return false;

            /*
				TODO: if the format of the image is a multi-planar format, and if subresourceRange.aspectMask
				is one of VK_IMAGE_ASPECT_PLANE_0_BIT, VK_IMAGE_ASPECT_PLANE_1_BIT, or VK_IMAGE_ASPECT_PLANE_2_BIT,
				then format must be compatible with the VkFormat for the plane of the image format indicated by subresourceRange.aspectMask,
				as defined in Compatible formats of planes of multi-planar formats
				*/
        }

        const auto& subresourceRange = _params.subresourceRange;

        if(imgParams.flags & IImage::ECF_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT)
        {
            /*
				TODO: format must be compatible with, or must be an uncompressed format that is size-compatible with,
				the format used to create image.
				*/
            if(subresourceRange.levelCount != 1u || subresourceRange.layerCount != 1u)
                return false;
        }
        else
        {
            if(mutableFormat)
            {
                /*
					TODO: if the format of the image is not a multi-planar format,
					format must be compatible with the format used to create image,
					as defined in Format Compatibility Classes
					*/
            }
        }

        if(!mutableFormat || asset::isPlanarFormat(imgParams.format))
        {
            /*
				TODO: format must be compatible with, or must be an uncompressed format that is size-compatible with,
				the format used to create image.
				*/
        }

        /*
			image must have been created with a usage value containing at least one of VK_IMAGE_USAGE_SAMPLED_BIT,
			VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
			VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, VK_IMAGE_USAGE_SHADING_RATE_IMAGE_BIT_NV, or VK_IMAGE_USAGE_FRAGMENT_DENSITY_MAP_BIT_EXT
			if (imgParams.)
				return false;
			*/

        if(subresourceRange.baseMipLevel >= imgParams.mipLevels)
            return false;
        if(subresourceRange.levelCount != remaining_mip_levels &&
            (subresourceRange.levelCount == 0u ||
                subresourceRange.baseMipLevel + subresourceRange.levelCount > imgParams.mipLevels))
            return false;
        auto mipExtent = _params.image->getMipSize(subresourceRange.baseMipLevel);

        if(subresourceRange.layerCount == 0u)
            return false;

        bool sourceIs3D = imgParams.type == IImage::ET_3D;
        bool sourceIs2DCompat = (imgParams.flags & IImage::ECF_2D_ARRAY_COMPATIBLE_BIT) && (_params.viewType == ET_2D || _params.viewType == ET_2D_ARRAY);
        auto actualLayerCount = subresourceRange.layerCount != remaining_array_layers ? subresourceRange.layerCount :
                                                                                        ((sourceIs3D && sourceIs2DCompat ? mipExtent.z : imgParams.arrayLayers) - subresourceRange.baseArrayLayer);
        bool checkLayers = true;
        auto hasCubemapProporties = [&](bool isItACubemapArray = false) {
            if(!(imgParams.flags & IImage::ECF_CUBE_COMPATIBLE_BIT))
                return false;
            if(imgParams.samples > 1u)
                return false;
            if(imgParams.extent.height != imgParams.extent.width)
                return false;
            if(imgParams.extent.depth > 1u)
                return false;
            if(actualLayerCount % 6u)
                return false;
            else if(isItACubemapArray)
            {
                if(imgParams.arrayLayers < 6u)
                    return false;
            }
            else if(imgParams.arrayLayers != 6u)
                return false;

            if(subresourceRange.baseArrayLayer + actualLayerCount > imgParams.arrayLayers)
                return false;
            return true;
        };

        switch(_params.viewType)
        {
            case ET_1D:
                if(imgParams.type != IImage::ET_1D)
                    return false;
                if(actualLayerCount > 1u)
                    return false;
                [[fallthrough]];
            case ET_1D_ARRAY:
                if(imgParams.extent.height > 1u || imgParams.extent.depth > 1u)
                    return false;
                break;
            case ET_2D:
                if(imgParams.type == IImage::ET_1D)
                    return false;
                if(actualLayerCount > 1u)
                    return false;
                [[fallthrough]];
            case ET_2D_ARRAY:
                if(sourceIs3D)
                {
                    if(!sourceIs2DCompat)
                        return false;
                    checkLayers = false;  // has compatible flag

                    if(imgParams.flags & (IImage::ECF_SPARSE_BINDING_BIT | IImage::ECF_SPARSE_RESIDENCY_BIT | IImage::ECF_SPARSE_ALIASED_BIT))
                        return false;

                    if(subresourceRange.levelCount > 1u)
                        return false;
                    if(subresourceRange.baseArrayLayer >= mipExtent.z)
                        return false;
                    if(subresourceRange.baseArrayLayer + actualLayerCount > mipExtent.z)
                        return false;
                }
                break;
            case ET_CUBE_MAP_ARRAY:
                if(!hasCubemapProporties(true))
                    return false;
                break;
            case ET_CUBE_MAP:
                if(!hasCubemapProporties(false))
                    return false;
                break;
            case ET_3D:
                // checkLayers will chack the appropriate stuff
                break;
            default:
                return false;
                break;
        }
        if(checkLayers)
        {
            if(subresourceRange.baseArrayLayer >= imgParams.arrayLayers)
                return false;
            if(subresourceRange.layerCount != remaining_array_layers &&
                (subresourceRange.baseArrayLayer + subresourceRange.layerCount > imgParams.arrayLayers))
                return false;
        }

        return true;
    }

    //!
    E_CATEGORY getTypeCategory() const override { return EC_IMAGE; }

    //!
    const SCreationParams& getCreationParameters() const { return params; }

protected:
    IImageView()
        : params{static_cast<E_CREATE_FLAGS>(0u), nullptr, ET_COUNT, EF_UNKNOWN, {}} {}
    IImageView(SCreationParams&& _params)
        : params(_params) {}
    virtual ~IImageView() = default;

    SCreationParams params;
};

}
}

#endif