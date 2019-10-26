// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_GRAPHICS_PIPELINE_H_INCLUDED__
#define __I_GRAPHICS_PIPELINE_H_INCLUDED__

#include "stdint.h"
#include "IVulkanPrebaked.h"
#include "IShader.h"
#include "irr/asset/IMeshBuffer.h"
#include "IDescriptorSetLayout.h"
///#include "SRasterState.h"

namespace irr
{
namespace video
{

/**
class IRR_FORCE_EBO SFixedFuncLayout
{
    public:
        //some of this shit will be dynamic
        video::SRasterState rasterState; //vkPipelineRasterizationStateCreateInfo
        video::SDepthStencilState depthStencilState; //vkPipelineDepthStencilStateCreateInfo
        video::SMultisampleState multisampleState; //vkPipelineMultisampleStateCreateInfo
        video::SGlobalBlendState globalBlendState; //vkPipelineColorBlendStateCreateInfo
        video::SSeparateBlendState separateBlendState[OGL_STATE_maxDRAW_BUFFERS]; //vkPipelineColorBlendAttachmentState
        video::SPipelineLayout pipelineLayout; //VkPipelineLayoutCreateInfo

        //! Descriptor Sets
        video::SCombinedImageSamplers combinedImageSampler[MATERIAL_MAX_TEXTURES];
        video::SStorageImage storageImages[MATERIAL_maxIMAGES];
        video::SInputAttachment inputAttachments[OGL_STATE_maxDRAW_BUFFERS];
        video::SUniformBuffer uniformBuffers[MATERIAL_maxUNIFORM_BUFFER_OBJECTS];
        video::SStorageBuffer storageBuffers[MATERIAL_maxSSBOs];
        video::SUniformTexelBuffer uniformTexelBuffers[MATERIAL_MAX_TEXTURES];
};
*/

class IGraphicsPipeline : public virtual IVulkanPrebaked
{
    public:
        //
    protected:
        IGraphicsPipeline() : meshFormatDesc(NULL), shader(NULL), descLayout(NULL)
        {
        }

        video::IShader* shader;
        scene::IGPUMeshDataFormatDesc* meshFormatDesc; //vkPipelineVertexInputStateCreateInfo,vkPipelineInputAssemblyStateCreateInfo
        video::IDescriptorSetLayout* descLayout;
        ///video::SFixedFuncLayout ffLayout;
};


} // end namespace video
} // end namespace irr

#endif



