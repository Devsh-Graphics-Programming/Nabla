// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_GPU_OBJECT_FROM_ASSET_CONVERTER_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_OBJECT_FROM_ASSET_CONVERTER_H_INCLUDED_

#include "nbl/core/declarations.h"
#include "nbl/core/alloc/LinearAddressAllocator.h"

#include "nbl/video/ISemaphore.h"
#include "nbl/video/ILogicalDevice.h"

#if 0
auto IGPUObjectFromAssetConverter::create(const asset::ICPUAccelerationStructure** _begin, const asset::ICPUAccelerationStructure** _end, SParams& _params) -> created_gpu_object_array<asset::ICPUAccelerationStructure>
{
	const size_t assetCount = std::distance(_begin, _end);
	auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUAccelerationStructure> >(assetCount);
	auto toCreateAndBuild = std::vector<const asset::ICPUAccelerationStructure*>();
    auto buildRangeInfos = std::vector<IGPUAccelerationStructure::BuildRangeInfo*>();
    toCreateAndBuild.reserve(assetCount);
    buildRangeInfos.reserve(assetCount);
    // Lambda function: creates the acceleration structure and It's buffer
    auto allocateBufferAndCreateAccelerationStructure = [&](size_t asSize, const asset::ICPUAccelerationStructure* cpuas)
    {
        // Create buffer with cpuas->getAccelerationStructureSize
        IGPUBuffer::SCreationParams gpuBufParams = {};
        gpuBufParams.size = asSize;
        gpuBufParams.usage = core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | asset::IBuffer::EUF_ACCELERATION_STRUCTURE_STORAGE_BIT;
        auto gpubuf = _params.device->createBuffer(std::move(gpuBufParams));
        auto mreqs = gpubuf->getMemoryReqs();
        mreqs.memoryTypeBits &= _params.device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
        auto gpubufMem = _params.device->allocate(mreqs, gpubuf.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
        assert(gpubufMem.isValid());

        // Create GPUAccelerationStructure with that buffer
        IGPUAccelerationStructure::SCreationParams creatationParams = {};
        creatationParams.bufferRange.buffer = gpubuf;
        creatationParams.bufferRange.offset = 0;
        creatationParams.bufferRange.size = asSize;
        creatationParams.flags = cpuas->getCreationParameters().flags;
        creatationParams.type = cpuas->getCreationParameters().type;
        return _params.device->createAccelerationStructure(std::move(creatationParams));
    };

    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        const asset::ICPUAccelerationStructure* cpuas = _begin[i];

        if(cpuas->hasBuildInfo())
        {
            // Add to toBuild vector of ICPUAccelerationStructure
            toCreateAndBuild.push_back(cpuas);
            buildRangeInfos.push_back(const_cast<IGPUAccelerationStructure::BuildRangeInfo*>(cpuas->getBuildRanges().begin()));
        }
        else if(cpuas->getAccelerationStructureSize() > 0)
        {
            res->operator[](i) = allocateBufferAndCreateAccelerationStructure(cpuas->getAccelerationStructureSize(), cpuas);
        }
    }

    if(toCreateAndBuild.empty() == false)
    {
        bool hostBuildCommands = false; // get from SFeatures
        if(hostBuildCommands)
        {
            _NBL_TODO();
        }
        else
        {
            core::vector<const asset::ICPUBuffer*> cpuBufferDeps;
            constexpr uint32_t MaxGeometryPerBuildInfo = 16;
            constexpr uint32_t MaxBuffersPerGeometry = 3; // TrianglesData ->  vertex+index+transformation
            cpuBufferDeps.reserve(assetCount * MaxGeometryPerBuildInfo * MaxBuffersPerGeometry);

            // Get CPUBuffer Dependencies
            for (ptrdiff_t i = 0u; i < toCreateAndBuild.size(); ++i)
            {
                const asset::ICPUAccelerationStructure* cpuas = toCreateAndBuild[i];
            
                auto buildInfo = cpuas->getBuildInfo();
                assert(buildInfo != nullptr);

                auto geoms = buildInfo->getGeometries().begin();
                auto geomsCount = buildInfo->getGeometries().size();
                if(geomsCount == 0)
                {
                    assert(false);
                    continue;
                }

                for(uint32_t g = 0; g < geomsCount; ++g) 
                {
                    const auto& geom = geoms[g];
                    if(geom.type == asset::IAccelerationStructure::EGT_TRIANGLES)
                    {
                        if(geom.data.triangles.indexData.isValid())
                        {
                            auto cpuBuf = geom.data.triangles.indexData.buffer.get();
                            cpuBuf->addUsageFlags(core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | asset::IBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
                            cpuBufferDeps.push_back(cpuBuf);
                        }
                        if(geom.data.triangles.vertexData.isValid())
                        {
                            auto cpuBuf = geom.data.triangles.vertexData.buffer.get();
                            cpuBuf->addUsageFlags(core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | asset::IBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
                            cpuBufferDeps.push_back(cpuBuf);
                        }
                        if(geom.data.triangles.transformData.isValid())
                        {
                            auto cpuBuf = geom.data.triangles.transformData.buffer.get();
                            cpuBuf->addUsageFlags(core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | asset::IBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
                            cpuBufferDeps.push_back(cpuBuf);
                        }
                    }
                    else if(geom.type == asset::IAccelerationStructure::EGT_AABBS)
                    {
                        if(geom.data.aabbs.data.isValid())
                        {
                            auto cpuBuf = geom.data.aabbs.data.buffer.get();
                            cpuBuf->addUsageFlags(core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | asset::IBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
                            cpuBufferDeps.push_back(cpuBuf);
                        }
                    }
                    else if(geom.type == asset::IAccelerationStructure::EGT_INSTANCES)
                    {
                        if(geom.data.instances.data.isValid())
                        {
                            auto cpuBuf = geom.data.instances.data.buffer.get();
                            cpuBuf->addUsageFlags(core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | asset::IBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
                            cpuBufferDeps.push_back(cpuBuf);
                        }
                    }
                }
            }

            // Convert CPUBuffer Deps to GPUBuffers
            core::vector<size_t> redirs = eliminateDuplicatesAndGenRedirs(cpuBufferDeps);
            auto gpuBufs = getGPUObjectsFromAssets<asset::ICPUBuffer>(cpuBufferDeps.data(), cpuBufferDeps.data()+cpuBufferDeps.size(), _params);
            _params.waitForCreationToComplete();
            _params.beginCommandBuffers();
            size_t bufIter = 0ull;

            // Fill buildGeomInfos partially (to later ge Get AS Size before build command)
            std::vector<IGPUAccelerationStructure::DeviceBuildGeometryInfo> buildGeomInfos(toCreateAndBuild.size());
     
            using GPUGeometry = IGPUAccelerationStructure::Geometry<IGPUAccelerationStructure::DeviceAddressType>;
            std::vector<GPUGeometry> gpuGeoms;
            gpuGeoms.reserve(assetCount * MaxGeometryPerBuildInfo);

            for (ptrdiff_t i = 0u; i < toCreateAndBuild.size(); ++i)
            {
                const asset::ICPUAccelerationStructure* cpuas = toCreateAndBuild[i];
            
                auto cpuBuildInfo = cpuas->getBuildInfo();
                auto & gpuBuildInfo = buildGeomInfos[i];

                gpuBuildInfo.type = cpuBuildInfo->type;
                gpuBuildInfo.buildFlags = cpuBuildInfo->buildFlags;
                gpuBuildInfo.buildMode = cpuBuildInfo->buildMode;
                assert(cpuBuildInfo->buildMode == asset::IAccelerationStructure::EBM_BUILD);

                // Fill Later:
                gpuBuildInfo.srcAS = nullptr;
                gpuBuildInfo.dstAS = nullptr;
                gpuBuildInfo.scratchAddr = {};
                
                auto cpu_geoms = cpuBuildInfo->getGeometries().begin();
                auto geomsCount = cpuBuildInfo->getGeometries().size();
                if(geomsCount == 0)
                {
                    assert(false);
                    continue;
                }

                size_t startGeom = gpuGeoms.size();
                size_t endGeom = gpuGeoms.size() + geomsCount;

                for(uint32_t g = 0; g < geomsCount; ++g) 
                {
                    const auto& cpu_geom = cpu_geoms[g];

                    GPUGeometry gpu_geom = {};
                    gpu_geom.type = cpu_geom.type;
                    gpu_geom.flags = cpu_geom.flags;

                    if(cpu_geom.type == asset::IAccelerationStructure::EGT_TRIANGLES)
                    {
                        gpu_geom.data.triangles.vertexFormat = cpu_geom.data.triangles.vertexFormat;
                        gpu_geom.data.triangles.vertexStride = cpu_geom.data.triangles.vertexStride;
                        gpu_geom.data.triangles.maxVertex = cpu_geom.data.triangles.maxVertex;
                        gpu_geom.data.triangles.indexType = cpu_geom.data.triangles.indexType;

                        {
                            IGPUOffsetBufferPair* gpubuf = (*gpuBufs)[redirs[bufIter++]].get();
                            gpu_geom.data.triangles.indexData.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
                            gpu_geom.data.triangles.indexData.offset = gpubuf->getOffset() + cpu_geom.data.triangles.indexData.offset;
                        }
                        {
                            IGPUOffsetBufferPair* gpubuf = (*gpuBufs)[redirs[bufIter++]].get();
                            gpu_geom.data.triangles.vertexData.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
                            gpu_geom.data.triangles.vertexData.offset = gpubuf->getOffset() + cpu_geom.data.triangles.vertexData.offset;
                        }
                        {
                            IGPUOffsetBufferPair* gpubuf = (*gpuBufs)[redirs[bufIter++]].get();
                            gpu_geom.data.triangles.transformData.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
                            gpu_geom.data.triangles.transformData.offset = gpubuf->getOffset() + cpu_geom.data.triangles.transformData.offset;
                        }
                    }
                    else if(cpu_geom.type == asset::IAccelerationStructure::EGT_AABBS)
                    {
                        gpu_geom.data.aabbs.stride = cpu_geom.data.aabbs.stride;
                        {
                            IGPUOffsetBufferPair* gpubuf = (*gpuBufs)[redirs[bufIter++]].get();
                            gpu_geom.data.aabbs.data.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
                            gpu_geom.data.aabbs.data.offset = gpubuf->getOffset() + cpu_geom.data.aabbs.data.offset;
                        }
                    }
                    else if(cpu_geom.type == asset::IAccelerationStructure::EGT_INSTANCES)
                    {
                        {
                            IGPUOffsetBufferPair* gpubuf = (*gpuBufs)[redirs[bufIter++]].get();
                            gpu_geom.data.instances.data.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
                            gpu_geom.data.instances.data.offset = gpubuf->getOffset() + cpu_geom.data.instances.data.offset;
                        }
                    }

                    gpuGeoms.push_back(gpu_geom);
                }

                gpuBuildInfo.geometries = core::SRange<GPUGeometry>(gpuGeoms.data() + startGeom, gpuGeoms.data() + endGeom);
            }
            
            // Get SizeInfo for each CPUAS -> Create the AS -> Get Total Scratch Buffer Size 
            std::vector<IGPUAccelerationStructure::BuildSizes> buildSizes(toCreateAndBuild.size());
            uint64_t totalScratchBufferSize = 0ull;
            uint64_t maxScratchBufferSize = 0ull;
            for (ptrdiff_t i = 0u, toBuildIndex = 0u; i < assetCount; ++i)
            {
                const asset::ICPUAccelerationStructure* cpuas = _begin[i];
                if(cpuas->hasBuildInfo() == false)
                {
                    // Only those with buildInfo (index in toCreateAndBuild vector) will get passed
                    continue;
                }

                assert(cpuas == toCreateAndBuild[toBuildIndex]);
                assert(toBuildIndex < toCreateAndBuild.size());

                auto buildRanges = cpuas->getBuildRanges().begin();
                auto buildRangesCount = cpuas->getBuildRanges().size();

                auto & gpuBuildInfo = buildGeomInfos[toBuildIndex];
                
                std::vector<uint32_t> maxPrimCount(buildRangesCount);
                for(auto b = 0; b < buildRangesCount; b++)
                  maxPrimCount[b] = buildRanges[b].primitiveCount;

                auto buildSize = _params.device->getAccelerationStructureBuildSizes(gpuBuildInfo, maxPrimCount.data());
                buildSizes[i] = buildSize;

                auto gpuAS = allocateBufferAndCreateAccelerationStructure(buildSize.accelerationStructureSize, cpuas);
                res->operator[](i) = gpuAS;

                // complete the buildGeomInfos (now only thing left is to allocate and set scratchAddr.buffer)
                buildGeomInfos[toBuildIndex].dstAS = gpuAS.get();
                buildGeomInfos[toBuildIndex].scratchAddr.offset = totalScratchBufferSize;

                totalScratchBufferSize += buildSize.buildScratchSize;
                hlsl::max(maxScratchBufferSize, buildSize.buildScratchSize); // maxScratchBufferSize has no use now (unless we changed this function to build 1 by 1 instead of batch builds or have some kind of memory limit?)
                ++toBuildIndex;
            }

            // Allocate Scratch Buffer
            IGPUBuffer::SCreationParams gpuScratchBufParams = {};
            gpuScratchBufParams.size = totalScratchBufferSize;
            gpuScratchBufParams.usage = core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | asset::IBuffer::EUF_STORAGE_BUFFER_BIT; 
            auto gpuScratchBuf = _params.device->createBuffer(std::move(gpuScratchBufParams));
            auto mreqs = gpuScratchBuf->getMemoryReqs();
            mreqs.memoryTypeBits &= _params.device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
            auto gpuScratchBufMem = _params.device->allocate(mreqs, gpuScratchBuf.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);


            for (ptrdiff_t i = 0u; i < toCreateAndBuild.size(); ++i)
            {
                auto & gpuBuildInfo = buildGeomInfos[i];
                gpuBuildInfo.scratchAddr.buffer = gpuScratchBuf;
            }

            // Record CommandBuffer for Building (We have Completed buildInfos + buildRanges for each CPUAS)
            auto & fence = _params.fences[EQU_COMPUTE];
            fence = _params.device->createFence(static_cast<IGPUFence::E_CREATE_FLAGS>(0));
            core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf = _params.perQueue[EQU_COMPUTE].cmdbuf;

            IQueue::SSubmitInfo submit;
            {
                submit.commandBufferCount = 1u;
                submit.commandBuffers = &cmdbuf.get();
                submit.waitSemaphoreCount = 0u;
                submit.pWaitDstStageMask = nullptr;
                submit.pWaitSemaphores = nullptr;
                uint32_t waitSemaphoreCount = 0u;
            }
            
            assert(cmdbuf->getState() == IGPUCommandBuffer::STATE::RECORDING);
            cmdbuf->buildAccelerationStructures({buildGeomInfos.data(),buildGeomInfos.data()+buildGeomInfos.size()},buildRangeInfos.data());
            cmdbuf->end();

            // TODO for future to make this function more sophisticated: Compaction, MemoryLimit for Build

            core::smart_refctd_ptr<IGPUSemaphore> sem;
            
            if (_params.perQueue[EQU_COMPUTE].semaphore)
                sem = _params.device->createSemaphore();

            auto* sem_ptr = sem.get();
            auto* fence_ptr = fence.get();

            submit.signalSemaphoreCount = sem_ptr?1u:0u;
            submit.pSignalSemaphores = sem_ptr?&sem_ptr:nullptr;

            _params.perQueue[EQU_COMPUTE].queue->submit(1u, &submit, fence_ptr);
            if (_params.perQueue[EQU_COMPUTE].semaphore)
                _params.perQueue[EQU_COMPUTE].semaphore[0] = std::move(sem);
        }
    }

    return res;
}
#endif

#endif
