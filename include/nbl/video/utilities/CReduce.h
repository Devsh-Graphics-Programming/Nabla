// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_VIDEO_C_REDUCE_H_INCLUDED_
#define _NBL_VIDEO_C_REDUCE_H_INCLUDED_

#include "nbl/video/utilities/CArithmeticOps.h"

namespace nbl::video
{

    class CReduce final : public CArithmeticOps
    {

    public:
        CReduce(core::smart_refctd_ptr<ILogicalDevice>&& device) : CReduce(std::move(device), core::roundDownToPoT(device->getPhysicalDevice()->getLimits().maxOptimallyResidentWorkgroupInvocations)) {}
        CReduce(core::smart_refctd_ptr<ILogicalDevice>&& device, const uint32_t workgroupSize) : CArithmeticOps(core::smart_refctd_ptr(device), workgroupSize) {}
        asset::ICPUShader* getDefaultShader(const CArithmeticOps::E_DATA_TYPE dataType, const CArithmeticOps::E_OPERATOR op, const uint32_t scratchElCount);
        IGPUShader* getDefaultSpecializedShader(const CArithmeticOps::E_DATA_TYPE dataType, const CArithmeticOps::E_OPERATOR op, const uint32_t scratchElCount);
        IGPUComputePipeline* getDefaultPipeline(const CArithmeticOps::E_DATA_TYPE dataType, const CArithmeticOps::E_OPERATOR op, const uint32_t scratchElCount);
        core::smart_refctd_ptr<asset::ICPUShader> createShader(const CArithmeticOps::E_DATA_TYPE dataType, const CArithmeticOps::E_OPERATOR op, const uint32_t scratchElCount) const;

    protected:
        ~CReduce()
        {
            // all drop themselves automatically
        }

        core::smart_refctd_ptr<asset::ICPUShader> m_shaders[EDT_COUNT][EO_COUNT];
        core::smart_refctd_ptr < IGPUShader > m_specialized_shaders[EDT_COUNT][EO_COUNT];
        core::smart_refctd_ptr<IGPUComputePipeline> m_pipelines[EDT_COUNT][EO_COUNT];
    };
}

#endif