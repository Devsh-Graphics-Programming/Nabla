// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/DebugDraw/CDraw3DLine.h"

namespace nbl
{
  namespace ext
  {
    namespace DebugDraw
    {
      CDraw3DLine::CDraw3DLine(const LogicalDevicePtr& device, bool isDepthEnabled)
      : m_device(LogicalDevicePtr(device))
      {
        shared_ptr<video::IGPUPipelineLayout> layout;
        {
          asset::SPushConstantRange range;
          range.offset = 0u;
          range.size = sizeof(core::matrix4SIMD);
          range.stageFlags = asset::IShader::ESS_VERTEX;
          layout = device->createGPUPipelineLayout(&range, &range + 1);
        }
        assert(layout);
        {
          auto vs_unspec = m_device->createGPUShader(core::make_smart_refctd_ptr<asset::ICPUShader>(Draw3DLineVertexShader, asset::IShader::ESS_VERTEX, "vs"));
          auto fs_unspec = m_device->createGPUShader(core::make_smart_refctd_ptr<asset::ICPUShader>(Draw3DLineFragmentShader, asset::IShader::ESS_FRAGMENT, "fs"));

          asset::ISpecializedShader::SInfo vsinfo(nullptr, nullptr, "main");
          auto vs = m_device->createGPUSpecializedShader(vs_unspec.get(), vsinfo);
          asset::ISpecializedShader::SInfo fsinfo(nullptr, nullptr, "main");
          auto fs = m_device->createGPUSpecializedShader(fs_unspec.get(), fsinfo);

          video::IGPUSpecializedShader* shaders[2]{ vs.get(), fs.get() };

          asset::SVertexInputParams vtxinput;
          vtxinput.attributes[0].binding = 0;
          vtxinput.attributes[0].format = asset::EF_R32G32_SFLOAT;
          vtxinput.attributes[0].relativeOffset = offsetof(S3DLineVertex, Position[0]);

          vtxinput.attributes[1].binding = 0;
          vtxinput.attributes[1].format = asset::EF_R32G32B32_SFLOAT;
          vtxinput.attributes[1].relativeOffset = offsetof(S3DLineVertex, Color[0]);

          vtxinput.bindings[0].inputRate = asset::EVIR_PER_VERTEX;
          vtxinput.bindings[0].stride = sizeof(S3DLineVertex);

          vtxinput.enabledAttribFlags = 0b0011;
          vtxinput.enabledBindingFlags = 0b0001;

          asset::SRasterizationParams raster;
          raster.depthTestEnable = isDepthEnabled;
          raster.depthWriteEnable = isDepthEnabled;
          raster.faceCullingMode = asset::EFCM_NONE;

          asset::SPrimitiveAssemblyParams primitive;
          primitive.primitiveType = asset::EPT_LINE_LIST;

          asset::SBlendParams blendParams;
          blendParams.logicOpEnable = false;
          blendParams.logicOp = nbl::asset::ELO_NO_OP;

          m_rpindependent_pipeline = m_device->createGPURenderpassIndependentPipeline(nullptr, core::smart_refctd_ptr(layout), shaders, shaders + 2, vtxinput, blendParams, primitive, raster);
          assert(m_rpindependent_pipeline);
        }
      }

      void CDraw3DLine::updateVertexBuffer(
        video::IUtilities* utilities,
        video::IGPUQueue* queue,
        shared_ptr<video::IGPUFence>* fence
      ) noexcept
      {
        size_t buffSize = m_linesBuffer.get() != nullptr ? m_linesBuffer->getSize() : 0;
        size_t minimalBuffSize = m_lines.size() * sizeof(std::pair<S3DLineVertex, S3DLineVertex>);

        if (buffSize < minimalBuffSize)
        {
          video::IGPUBuffer::SCreationParams creationParams;
          creationParams.usage = static_cast<asset::IBuffer::E_USAGE_FLAGS>(asset::IBuffer::EUF_VERTEX_BUFFER_BIT | asset::IBuffer::EUF_TRANSFER_DST_BIT);
          creationParams.sharingMode = asset::E_SHARING_MODE::ESM_EXCLUSIVE;
          creationParams.queueFamilyIndices = nullptr;

          m_linesBuffer = m_device->createDeviceLocalGPUBufferOnDedMem(creationParams, minimalBuffSize);
        }

        asset::SBufferRange<video::IGPUBuffer> range;
        range.buffer = m_linesBuffer;
        range.offset = 0;
        range.size = minimalBuffSize;

        if (!fence)
        { utilities->updateBufferRangeViaStagingBuffer(queue, range, m_lines.data()); }
        else
        {
          *fence = m_device->createFence(video::IGPUFence::ECF_UNSIGNALED);
          utilities->updateBufferRangeViaStagingBuffer(fence->get(), queue, range, m_lines.data());
        }
      }

      void CDraw3DLine::recordToCommandBuffer(
        video::IGPUCommandBuffer* cmdBuffer,
        video::IGPUGraphicsPipeline* graphics_pipeline,
        uint32_t pushConstOffset
      ) noexcept
      {
        auto cb = cmdBuffer;
        assert(cb->getState() == video::IGPUCommandBuffer::ES_RECORDING);
        size_t offset = 0u;
        cb->bindVertexBuffers(0u, 1u, const_cast<const video::IGPUBuffer**>(&m_linesBuffer.get()), &offset);
        cb->pushConstants(const_cast<video::IGPUPipelineLayout*>(m_rpindependent_pipeline->getLayout()), asset::IShader::ESS_VERTEX, pushConstOffset, sizeof(m_viewProj), &m_viewProj);
        cb->bindGraphicsPipeline(graphics_pipeline);
        cb->draw(m_lines.size() * 2, 1u, 0u, 0u);
      }
    } // namespace DebugDraw
  } // namespace ext
} // namespace nbl