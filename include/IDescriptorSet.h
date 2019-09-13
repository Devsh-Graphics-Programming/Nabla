// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_DESCRIPTOR_SET_H_INCLUDED__
#define __I_DESCRIPTOR_SET_H_INCLUDED__

#include "stdint.h"
#include "IVulkanPrebaked.h"
#include "IDescriptorSetLayout.h"
#include "IGPUBuffer.h"
#include "STextureSamplingParams.h"
#include "IRenderableVirtualTexture.h"

namespace irr
{
namespace video
{

/**
It's IVulkanPrebaked because IGPUBuffer, IStorageImage and texture views are IVulkanPrebaked
**/
class IDescriptorSet : public virtual IVulkanPrebaked
{
    public:
    protected:
        IDescriptorSet(IDescriptorSetLayout* layout)
        {
            for (auto i=0; i<_IRR_MATERIAL_MAX_IMAGES_; i++)
                storageImages[i] = nullptr;
        }
        virtual ~IDescriptorSet()
        {
            for (auto i=0; i<_IRR_MATERIAL_MAX_IMAGES_; i++)
            {
                if (storageImages[i])
                    storageImages->drop();
            }
        }
        /* Old and wrong design
        IStorageImage* storageImages[_IRR_MATERIAL_MAX_IMAGES_]; //for all types except EVTT_BUFFER_OBJECT
        IStorageImageBuffer* storageTexelBuffers[_IRR_MATERIAL_MAX_IMAGES_];
        SCombinedImageSampler texturesAndSamplers[_IRR_MATERIAL_MAX_TEXTURES_]; //for all types except EVTT_BUFFER_OBJECT
        ITextureBufferObject* uniformTexelObjects[_IRR_MATERIAL_MAX_TEXTURES_];
        */
        IStorageImage* storageImages[_IRR_MATERIAL_MAX_IMAGES_];
        class CombinedImageSampler
        {
            public:
                CombinedImageSampler() : texture(NULL)
                {
                }
                CombinedImageSampler(IRenderableVirtualTexture* inTexture, STextureSamplingParams inSampler) //0 means whole range
                    : texture(NULL)
                {
                    if (!inTexture)
                        return;

                    //! Valid Usage: https://www.khronos.org/registry/vulkan/specs/

                    texture = inTexture;
                    sampler = inSampler;

                    texture->grab();
                }
                ~CombinedImageSampler()
                {
                    if (texture)
                        texture->drop();
                }

                IRenderableVirtualTexture* texture;
                STextureSamplingParams sampler;
        };
        CombinedImageSampler texturesAndSamplers[_IRR_MATERIAL_MAX_TEXTURES_];


        class RangedBufferMapping
        {
            public:
                RangedBufferMapping() : buffer(NULL), baseOffset(0), size(0)
                {
                }
                RangedBufferMapping(IGPUBuffer* inBuffer, const uint32_t& inBaseOffset, const uint32_t& inSize=0) //0 means whole range
                    : buffer(NULL), baseOffset(0), size(0)
                {
                    if (!inBuffer)
                        return;

                    //! Valid Usage: https://www.khronos.org/registry/vulkan/specs/1.1-extensions/man/html/VkDescriptorBufferInfo.html
                    uint32_t bufSz = inBuffer->getSize();
                    if (inBaseOffset>=bufSz)
                        return;
                    if (inSize&&inSize>bufSz-offset)
                        return;

                    buffer      = inBuffer;
                    baseOffset  = inBaseOffset;
                    size        = inSize;

                    buffer->grab();
                }
                ~RangedBufferMapping()
                {
                    if (buffer)
                        buffer->drop();
                }

                IGPUBuffer* buffer;
                uint32_t    baseOffset,size;
        };
        RangedBufferMapping SSBOs[_IRR_MATERIAL_MAX_SHADER_STORAGE_OBJECTS_];
        RangedBufferMapping UBOs[_IRR_MATERIAL_MAX_UNIFORM_BUFFER_OBJECTS_];

        typedef RangedBufferMapping DynamicRangedBufferMapping;
        DynamicRangedBufferMapping dynRangeSSBOs[_IRR_MATERIAL_MAX_DYNAMIC_SHADER_STORAGE_OBJECTS_];
        DynamicRangedBufferMapping dynRangeUBOs[_IRR_MATERIAL_MAX_DYNAMIC_UNIFORM_BUFFER_OBJECTS_];
};


} // end namespace video
} // end namespace irr

#endif




