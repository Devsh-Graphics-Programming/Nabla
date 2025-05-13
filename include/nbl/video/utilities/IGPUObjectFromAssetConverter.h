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
                core::max(maxScratchBufferSize, buildSize.buildScratchSize); // maxScratchBufferSize has no use now (unless we changed this function to build 1 by 1 instead of batch builds or have some kind of memory limit?)
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
#endif

#endif
