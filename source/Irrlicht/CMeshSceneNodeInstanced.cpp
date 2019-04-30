// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CMeshSceneNodeInstanced.h"
#include "IVideoDriver.h"
#include "COpenGLDriver.h"
#include "ISceneManager.h"
#include "ICameraSceneNode.h"
#include "IMaterialRenderer.h"
#include "os.h"
#include "irr/video/SGPUMesh.h"

#include "irr/static_if.h"

namespace irr
{
namespace scene
{


uint32_t CMeshSceneNodeInstanced::recullOrder;

//!constructor
CMeshSceneNodeInstanced::CMeshSceneNodeInstanced(IDummyTransformationSceneNode* parent, ISceneManager* mgr, int32_t id,
        const core::vector3df& position, const core::vector3df& rotation, const core::vector3df& scale)
    : IMeshSceneNodeInstanced(parent, mgr, id, position, rotation, scale),
    instanceBBoxes(nullptr), instanceBBoxesCount(0), flagQueryForRetrieval(false),
    gpuCulledLodInstanceDataBuffer(nullptr), dataPerInstanceOutputSize(0),
    extraDataInstanceSize(0), dataPerInstanceInputSize(0), cachedMaterialCount(0)
{
    #ifdef _IRR_DEBUG
    setDebugName("CMeshSceneNodeInstanced");
    #endif


    renderPriority = 0x80000000u;

    lodCullingPointMesh = new video::IGPUMeshBuffer();
    lodCullingPointMesh->setPrimitiveType(asset::EPT_POINTS);
}

//! destructor
CMeshSceneNodeInstanced::~CMeshSceneNodeInstanced()
{
    for (size_t i=0; i<LoD.size(); i++)
    {
        LoD[i].mesh->drop();
        LoD[i].query->drop();
    }
    for (size_t i=0; i<xfb.size(); i++)
        xfb[i]->drop();

    lodCullingPointMesh->drop();

    if (instanceBBoxes)
        _IRR_ALIGNED_FREE(instanceBBoxes);
    if (gpuCulledLodInstanceDataBuffer)
        gpuCulledLodInstanceDataBuffer->drop();
}


//! Sets a new meshbuffer
bool CMeshSceneNodeInstanced::setLoDMeshes(const core::vector<MeshLoD>& levelsOfDetail, const size_t& dataSizePerInstanceOutput, const video::SGPUMaterial& lodSelectionShader, VaoSetupOverrideFunc vaoSetupOverride, const size_t shaderLoDsPerPass, void* overrideUserData, const size_t& extraDataSizePerInstanceInput)
{
    for (size_t i=0; i<LoD.size(); i++)
    {
        LoD[i].mesh->drop();
        LoD[i].query->drop();
    }
    LoD.clear();
    for (size_t i=0; i<xfb.size(); i++)
        xfb[i]->drop();
    xfb.clear();

    if (instanceDataAllocator)
        instanceDataAllocator->drop();
    if (instanceBBoxes)
        _IRR_ALIGNED_FREE(instanceBBoxes);
    if (gpuCulledLodInstanceDataBuffer)
        gpuCulledLodInstanceDataBuffer->drop();
    instanceDataAllocator = nullptr;
    instanceBBoxes = nullptr;
    gpuCulledLodInstanceDataBuffer = nullptr;
    extraDataInstanceSize = 0;

    lodCullingPointMesh->setMeshDataAndFormat(nullptr);
    lodCullingPointMesh->setIndexCount(0);

    if (levelsOfDetail.size()==0||!vaoSetupOverride)
        return false;

    if (!levelsOfDetail[0].mesh||levelsOfDetail[0].lodDistance<=0.f)
        return false;

    for (size_t j=1; j<levelsOfDetail.size(); j++)
    {
        if (!levelsOfDetail[j].mesh||levelsOfDetail[j].lodDistance<=levelsOfDetail[j-1].lodDistance)
            return false;
    }

#ifdef _IRR_COMPILE_WITH_OPENGL_
    if (shaderLoDsPerPass>video::COpenGLExtensionHandler::MaxVertexStreams)
        return false;
#endif // _IRR_COMPILE_WITH_OPENGL_
    gpuLoDsPerPass = shaderLoDsPerPass;

	extraDataInstanceSize = extraDataSizePerInstanceInput;
    auto visibilityPadding = 4u-(extraDataInstanceSize&0x3u);

    video::IVideoDriver* driver = SceneManager->getVideoDriver();

    dataPerInstanceInputSize = extraDataInstanceSize+visibilityPadding+48+36;
    auto buffSize = dataPerInstanceInputSize*512u;
    instanceDataAllocator = new std::remove_pointer<decltype(instanceDataAllocator)>::type(driver,core::allocator<uint8_t>(),0u,0u,core::roundDownToPoT(dataPerInstanceInputSize),buffSize,dataPerInstanceInputSize,nullptr);
	instanceBBoxesCount = getCurrentInstanceCapacity();
	instanceBBoxes = (core::aabbox3df*)_IRR_ALIGNED_MALLOC(instanceBBoxesCount*sizeof(core::aabbox3df),_IRR_SIMD_ALIGNMENT);
	for (size_t i=0; i<instanceBBoxesCount; i++)
    {
        instanceBBoxes[i].MinEdge.set( FLT_MAX, FLT_MAX, FLT_MAX);
        instanceBBoxes[i].MaxEdge.set(-FLT_MAX,-FLT_MAX,-FLT_MAX);
    }

    xfb.resize((levelsOfDetail.size()+gpuLoDsPerPass-1)/gpuLoDsPerPass);

	gpuCulledLodInstanceDataBuffer = SceneManager->getVideoDriver()->createDeviceLocalGPUBufferOnDedMem(dataSizePerInstanceOutput*instanceBBoxesCount*gpuLoDsPerPass*xfb.size());


	dataPerInstanceOutputSize = dataSizePerInstanceOutput;
    {
        video::IGPUBuffer* buff = instanceDataAllocator->getFrontBuffer();

        video::IGPUMeshDataFormatDesc* vao = SceneManager->getVideoDriver()->createGPUMeshDataFormatDesc();
        lodCullingPointMesh->setMeshDataAndFormat(vao);
        vao->drop();

        uint32_t floatComponents = extraDataInstanceSize+1;
        floatComponents /= 4;
        floatComponents += 12+9;
        if (floatComponents>asset::EVAI_COUNT*4)
        {
            for (uint32_t i=0; i<asset::EVAI_COUNT; i++)
                vao->setVertexAttrBuffer(buff,(asset::E_VERTEX_ATTRIBUTE_ID)i,asset::EF_R32G32B32A32_SFLOAT,dataPerInstanceInputSize,i*16);
        }
        else
        {
            size_t memoryUsed = 0;
            uint32_t attr = 0;
            for (; attr*4+3<floatComponents; attr++)
            {
                vao->setVertexAttrBuffer(buff,(asset::E_VERTEX_ATTRIBUTE_ID)attr,asset::EF_R32G32B32A32_SFLOAT,dataPerInstanceInputSize,attr*16);
                memoryUsed+=16;
            }
            memoryUsed -= (12+9)*4;

            size_t leftOverMemory = extraDataInstanceSize+1-memoryUsed;

            auto convertFunc = [](size_t x) { // rename this? What's this for actually?
                switch (x)
                {
                case 1ull: return asset::EF_R32_UINT;
                case 2ull: return asset::EF_R32G32_UINT;
                case 3ull: return asset::EF_R32G32B32_UINT;
                default: return asset::EF_R32G32B32A32_UINT;
                }
            };

            //assume a padding of 4 at the end
            vao->setVertexAttrBuffer(buff,(asset::E_VERTEX_ATTRIBUTE_ID)attr,convertFunc(((leftOverMemory+3)/4)),dataPerInstanceInputSize,attr*16);
        }
    }


    for (size_t i=0; i<levelsOfDetail.size(); i++)
    {
        cachedMaterialCount += levelsOfDetail[i].mesh->getMeshBufferCount();

        LoDData tmp;
        tmp.distanceSQ = levelsOfDetail[i].lodDistance;
        tmp.distanceSQ *= tmp.distanceSQ;

        tmp.mesh = new video::SGPUMesh();
        for (size_t j=0; j<levelsOfDetail[i].mesh->getMeshBufferCount(); j++)
        {
            video::IGPUMeshBuffer* origBuff = levelsOfDetail[i].mesh->getMeshBuffer(j);

            video::IGPUMeshBuffer* meshBuff = new video::IGPUMeshBuffer();
            meshBuff->setBaseVertex(origBuff->getBaseVertex());
            if (origBuff->isIndexCountGivenByXFormFeedback())
                meshBuff->setIndexCountFromXFormFeedback(origBuff->getXFormFeedback(),origBuff->getXFormFeedbackStream());
            else
                meshBuff->setIndexCount(origBuff->getIndexCount());
            meshBuff->setIndexBufferOffset(origBuff->getIndexBufferOffset());
            meshBuff->setIndexType(origBuff->getIndexType());
            meshBuff->setPrimitiveType(origBuff->getPrimitiveType());

            asset::IMeshDataFormatDesc<video::IGPUBuffer>* vao = vaoSetupOverride(SceneManager,gpuCulledLodInstanceDataBuffer,dataSizePerInstanceOutput,origBuff->getMeshDataAndFormat(),overrideUserData);
            meshBuff->setMeshDataAndFormat(vao);
            vao->drop();

            meshBuff->getMaterial() = origBuff->getMaterial();
            meshBuff->setBoundingBox(origBuff->getBoundingBox());
            tmp.mesh->addMeshBuffer(meshBuff);
            meshBuff->drop();
        }
        tmp.mesh->setBoundingBox(levelsOfDetail[i].mesh->getBoundingBox());
        if (i)
            LoDInvariantBox.addInternalBox(levelsOfDetail[i].mesh->getBoundingBox());
        else
            LoDInvariantBox = levelsOfDetail[i].mesh->getBoundingBox();

        tmp.query = SceneManager->getVideoDriver()->createXFormFeedbackPrimitiveQuery();
        LoD.push_back(tmp);
    }

    for (size_t i=0; i<xfb.size(); i++)
    {
        xfb[i] = SceneManager->getVideoDriver()->createTransformFeedback();

        for (size_t j=0; j<gpuLoDsPerPass; j++)
            xfb[i]->bindOutputBuffer(j,gpuCulledLodInstanceDataBuffer,(i*gpuLoDsPerPass+j)*dataSizePerInstanceOutput*instanceBBoxesCount,dataSizePerInstanceOutput*instanceBBoxesCount);
    }

    lodCullingPointMesh->getMaterial() = lodSelectionShader;

    return true;
}

uint32_t CMeshSceneNodeInstanced::addInstance(const core::matrix4x3& relativeTransform, const void* extraData)
{
    uint32_t ix;
    if (!addInstances(&ix,1,&relativeTransform,extraData))
        return kInvalidInstanceID;

    return ix;
}

bool CMeshSceneNodeInstanced::addInstances(uint32_t* instanceIDs, const size_t& instanceCount, const core::matrix4x3* relativeTransforms, const void* extraData)
{
    {//dummyBytes, aligns scope
    core::vector<uint32_t> dummyBytes_(instanceCount);
    core::vector<uint32_t> aligns_(instanceCount);
    uint32_t* const dummyBytes = dummyBytes_.data();
    uint32_t* const aligns = aligns_.data();
    for (size_t i=0; i<instanceCount; i++)
    {
        instanceIDs[i] = kInvalidInstanceID;
        dummyBytes[i] = dataPerInstanceInputSize;
        aligns[i] = 4u; // 4-byte alignment
    }

    instanceDataAllocator->multi_alloc_addr(instanceCount,instanceIDs,static_cast<const uint32_t*>(dummyBytes),static_cast<const uint32_t*>(aligns));
    bool success = true;
    for (size_t i=0; i<instanceCount&&success; i++)
        success = instanceIDs[i]!=kInvalidInstanceID;
    if (!success)
    {
        for (size_t i=0; i<instanceCount; i++)
        {
            if (instanceIDs[i]==kInvalidInstanceID)
                continue;
            instanceDataAllocator->multi_free_addr(1u,instanceIDs+i,dummyBytes+i);
            instanceIDs[i] = kInvalidInstanceID;
        }
        return false;
    }
    }// end of arbitrary scope

    if (getCurrentInstanceCapacity()!=instanceBBoxesCount)
    {
        size_t newCount = getCurrentInstanceCapacity();
        // kind-of realloc
        {
            size_t newSize = newCount*sizeof(core::aabbox3df);
            void* newPtr = _IRR_ALIGNED_MALLOC(newSize,_IRR_SIMD_ALIGNMENT);
            memcpy(newPtr,instanceBBoxes,std::min(instanceBBoxesCount*sizeof(core::aabbox3df),newSize));
            _IRR_ALIGNED_FREE(instanceBBoxes);
            instanceBBoxes = (core::aabbox3df*)newPtr;
        }
        for (size_t i=instanceBBoxesCount; i<newCount; i++)
        {
            instanceBBoxes[i].MinEdge.set( FLT_MAX, FLT_MAX, FLT_MAX);
            instanceBBoxes[i].MaxEdge.set(-FLT_MAX,-FLT_MAX,-FLT_MAX);
        }
        instanceBBoxesCount = newCount;
    }
    needsBBoxRecompute = true;

    uint8_t* base_pointer = reinterpret_cast<uint8_t*>(instanceDataAllocator->getBackBufferPointer());
    for (size_t i=0; i<instanceCount; i++)
    {
        {
            uint32_t blockID = getBlockIDFromAddr(instanceIDs[i]);
            instanceBBoxes[blockID] = LoDInvariantBox;
            relativeTransforms[i].transformBoxEx(instanceBBoxes[blockID]);
        }
        size_t redirect = instanceDataAllocator->getAddressAllocator().get_real_addr(instanceIDs[i]);
        instanceDataAllocator->markRangeForPush(redirect,redirect+dataPerInstanceInputSize);
        uint8_t* ptr = base_pointer+redirect;
        memcpy(ptr,relativeTransforms+i,48);

        core::matrix4x3 instanceInverse;
        relativeTransforms[i].getInverse(instanceInverse);
        float* instance3x3TranposeInverse = reinterpret_cast<float*>(ptr+48);
        instance3x3TranposeInverse[0] = instanceInverse(0,0);
        instance3x3TranposeInverse[1] = instanceInverse(0,1);
        instance3x3TranposeInverse[2] = instanceInverse(0,2);
        instance3x3TranposeInverse[3] = instanceInverse(1,0);
        instance3x3TranposeInverse[4] = instanceInverse(1,1);
        instance3x3TranposeInverse[5] = instanceInverse(1,2);
        instance3x3TranposeInverse[6] = instanceInverse(2,0);
        instance3x3TranposeInverse[7] = instanceInverse(2,1);
        instance3x3TranposeInverse[8] = instanceInverse(2,2);
        if (extraData&&extraDataInstanceSize)
            memcpy(ptr+48+36,reinterpret_cast<const uint8_t*>(extraData)+extraDataInstanceSize*i,extraDataInstanceSize);
        ptr[48+36+extraDataInstanceSize] = 0xffu;
    }

    lodCullingPointMesh->setIndexCount(lodCullingPointMesh->getIndexCount()+instanceCount);

    return true;
}

void CMeshSceneNodeInstanced::setInstanceTransform(const uint32_t& instanceID, const core::matrix4x3& relativeTransform)
{
    {
        uint32_t blockID = getBlockIDFromAddr(instanceID);
        instanceBBoxes[blockID] = LoDInvariantBox;
        relativeTransform.transformBoxEx(instanceBBoxes[blockID]);
    }

    size_t redirect = instanceDataAllocator->getAddressAllocator().get_real_addr(instanceID);
    instanceDataAllocator->markRangeForPush(redirect,redirect+48+36);
    uint8_t* ptr = reinterpret_cast<uint8_t*>(instanceDataAllocator->getBackBufferPointer())+redirect;
    memcpy(ptr,relativeTransform.pointer(),48);

    core::matrix4x3 instanceInverse;
    relativeTransform.getInverse(instanceInverse);
    float* instance3x3TranposeInverse = reinterpret_cast<float*>(ptr+48);
    instance3x3TranposeInverse[0] = instanceInverse(0,0);
    instance3x3TranposeInverse[1] = instanceInverse(0,1);
    instance3x3TranposeInverse[2] = instanceInverse(0,2);
    instance3x3TranposeInverse[3] = instanceInverse(1,0);
    instance3x3TranposeInverse[4] = instanceInverse(1,1);
    instance3x3TranposeInverse[5] = instanceInverse(1,2);
    instance3x3TranposeInverse[6] = instanceInverse(2,0);
    instance3x3TranposeInverse[7] = instanceInverse(2,1);
    instance3x3TranposeInverse[8] = instanceInverse(2,2);

    needsBBoxRecompute = true;
}

core::matrix4x3 CMeshSceneNodeInstanced::getInstanceTransform(const uint32_t& instanceID)
{
    core::matrix4x3 retval;
    size_t redir = instanceDataAllocator->getAddressAllocator().get_real_addr(instanceID);
    if (redir==kInvalidInstanceID)
    {
        _IRR_BREAK_IF(true);
        memset(retval.pointer(),0,48);
    }
    else
        memcpy(retval.pointer(),reinterpret_cast<uint8_t*>(instanceDataAllocator->getBackBufferPointer())+redir,sizeof(core::matrix4x3));

    return retval;
}

void CMeshSceneNodeInstanced::setInstanceVisible(const uint32_t& instanceID, const bool& visible)
{
    size_t redirect = instanceDataAllocator->getAddressAllocator().get_real_addr(instanceID)+36+48+extraDataInstanceSize;
    instanceDataAllocator->markRangeForPush(redirect,redirect+1u);
    reinterpret_cast<uint8_t*>(instanceDataAllocator->getBackBufferPointer())[redirect] = visible;
    /// update BBox?
}

void CMeshSceneNodeInstanced::setInstanceData(const uint32_t& instanceID, const void* data)
{
    if (extraDataInstanceSize==0)
        return;

    size_t redirect = instanceDataAllocator->getAddressAllocator().get_real_addr(instanceID)+36+48;
    instanceDataAllocator->markRangeForPush(redirect,redirect+extraDataInstanceSize);
    uint8_t* ptr = reinterpret_cast<uint8_t*>(instanceDataAllocator->getBackBufferPointer())+redirect;
    memcpy(ptr,data,extraDataInstanceSize);
}

void CMeshSceneNodeInstanced::removeInstance(const uint32_t& instanceID)
{
    removeInstances(1,&instanceID);
}

void CMeshSceneNodeInstanced::removeInstances(const size_t& instanceCount, const uint32_t* instanceIDs)
{
    constexpr bool usesContiguousAddrAllocator = std::is_same<core::ContiguousPoolAddressAllocatorST<uint32_t>,InstanceDataAddressAllocator>::value;

    uint32_t minRedirect  = kInvalidInstanceID;
    for (size_t i=0; i<instanceCount; i++)
    {
		IRR_PSEUDO_IF_CONSTEXPR_BEGIN(usesContiguousAddrAllocator)
		{
            uint32_t redirect =  instanceDataAllocator->getAddressAllocator().get_real_addr(instanceIDs[i]);
            if (redirect<minRedirect)
                minRedirect = redirect;
        }
		IRR_PSEUDO_IF_CONSTEXPR_END

        uint32_t blockID = getBlockIDFromAddr(instanceIDs[i]);
        instanceBBoxes[blockID].MinEdge.set( FLT_MAX, FLT_MAX, FLT_MAX);
        instanceBBoxes[blockID].MaxEdge.set(-FLT_MAX,-FLT_MAX,-FLT_MAX);
    }

    {// dummyBytes scope
    core::vector<uint32_t> dummyBytes_(instanceCount,dataPerInstanceInputSize);
    uint32_t* const dummyBytes = dummyBytes_.data();

    instanceDataAllocator->multi_free_addr(instanceCount,instanceIDs,static_cast<const uint32_t*>(dummyBytes));
    }

	IRR_PSEUDO_IF_CONSTEXPR_BEGIN(usesContiguousAddrAllocator)
	{
        // everything got shifted down by 1 so mark dirty
        instanceDataAllocator->markRangeForPush(minRedirect,core::address_allocator_traits<InstanceDataAddressAllocator>::get_allocated_size(instanceDataAllocator->getAddressAllocator()));
    }
	IRR_PSEUDO_IF_CONSTEXPR_END

    if (getCurrentInstanceCapacity()!=instanceBBoxesCount)
    {
        size_t newCount = getCurrentInstanceCapacity();
        { // kind-of realloc
            size_t newSize = newCount*sizeof(core::aabbox3df);
            void* newPtr = _IRR_ALIGNED_MALLOC(newSize,_IRR_SIMD_ALIGNMENT);
            memcpy(newPtr,instanceBBoxes,newSize);
            _IRR_ALIGNED_FREE(instanceBBoxes);
            instanceBBoxes = (core::aabbox3df*)newPtr;
        }
        for (size_t i=instanceBBoxesCount; i<newCount; i++)
        {
            instanceBBoxes[i].MinEdge.set( FLT_MAX, FLT_MAX, FLT_MAX);
            instanceBBoxes[i].MaxEdge.set(-FLT_MAX,-FLT_MAX,-FLT_MAX);
        }
        instanceBBoxesCount = newCount;
    }
    needsBBoxRecompute = true;

    lodCullingPointMesh->setIndexCount(lodCullingPointMesh->getIndexCount()-instanceCount);
}

void CMeshSceneNodeInstanced::RecullInstances()
{
    if (LoD.size()==0||!instanceDataAllocator||getInstanceCount()==0||!SceneManager)
    {
        for (size_t i=0; i<LoD.size(); i++)
        for (size_t j=0; j<LoD[i].mesh->getMeshBufferCount(); j++)
        {
            LoD[i].mesh->getMeshBuffer(j)->setInstanceCount(0);
            LoD[i].mesh->getMeshBuffer(j)->setBaseInstance(0);
        }
        Box.MinEdge = Box.MaxEdge = core::vector3df(0.f);
        return;
    }

    video::IVideoDriver* driver = SceneManager->getVideoDriver();

    {
        //can swap before or after, but defubuteky before tform feedback shadeur
        instanceDataAllocator->pushBuffer(driver->getDefaultUpStreamingBuffer());

        size_t outputSizePerLoD = dataPerInstanceOutputSize*getCurrentInstanceCapacity();
        if (gpuCulledLodInstanceDataBuffer->getSize()!=xfb.size()*gpuLoDsPerPass*outputSizePerLoD)
        {
            video::IDriverMemoryBacked::SDriverMemoryRequirements reqs = gpuCulledLodInstanceDataBuffer->getMemoryReqs();
            reqs.vulkanReqs.size = xfb.size()*gpuLoDsPerPass*outputSizePerLoD;
            {auto rep = SceneManager->getVideoDriver()->createGPUBufferOnDedMem(reqs,gpuCulledLodInstanceDataBuffer->canUpdateSubRange()); gpuCulledLodInstanceDataBuffer->pseudoMoveAssign(rep); rep->drop();}
            for (size_t i=0; i<xfb.size(); i++)
            {
                for (size_t j=0; j<gpuLoDsPerPass; j++)
                    xfb[i]->bindOutputBuffer(j,gpuCulledLodInstanceDataBuffer,(i*gpuLoDsPerPass+j)*outputSizePerLoD,outputSizePerLoD);
            }
        }

        driver->setTransform(video::E4X3TS_WORLD,AbsoluteTransformation);
        for (size_t i=0; i<xfb.size(); i++)
        {
            reinterpret_cast<uint32_t&>(lodCullingPointMesh->getMaterial().MaterialTypeParam) = i*gpuLoDsPerPass;
            reinterpret_cast<uint32_t&>(lodCullingPointMesh->getMaterial().MaterialTypeParam2) = i*gpuLoDsPerPass+gpuLoDsPerPass-1;
            driver->setMaterial(lodCullingPointMesh->getMaterial());
            driver->beginTransformFeedback(xfb[i],lodCullingPointMesh->getMaterial().MaterialType,asset::EPT_POINTS);
            for (size_t j=0; j<gpuLoDsPerPass&&(i*gpuLoDsPerPass+j)<LoD.size(); j++)
                driver->beginQuery(LoD[i*gpuLoDsPerPass+j].query,j);
            driver->drawMeshBuffer(lodCullingPointMesh);
            for (size_t j=0; j<gpuLoDsPerPass&&(i*gpuLoDsPerPass+j)<LoD.size(); j++)
                driver->endQuery(LoD[i*gpuLoDsPerPass+j].query,j);
            driver->endTransformFeedback();
        }

        renderPriority = 0x80000000u-(++recullOrder);
        flagQueryForRetrieval = true;
    }
}

//! frame
void CMeshSceneNodeInstanced::OnRegisterSceneNode()
{
    ISceneNode::OnRegisterSceneNode();

	if (IsVisible&&LoD.size()&&instanceDataAllocator&&getInstanceCount()&&canProceedPastFence())
	{
		// because this node supports rendering of mixed mode meshes consisting of
		// transparent and solid material at the same time, we need to go through all
		// materials, check of what type they are and register this node for the right
		// render pass according to that.

		video::IVideoDriver* driver = SceneManager->getVideoDriver();

		PassCount = 0;
		int transparentCount = 0;
		int solidCount = 0;

        for (size_t i=0; i<LoD.size(); ++i)
        for (size_t j=0; j<LoD[i].mesh->getMeshBufferCount(); j++)
        {
            video::IGPUMeshBuffer* mb = LoD[i].mesh->getMeshBuffer(j);
            if (!mb||(mb->getIndexCount()<1 && !mb->isIndexCountGivenByXFormFeedback()))
                continue;

            video::IMaterialRenderer* rnd = driver->getMaterialRenderer(mb->getMaterial().MaterialType);

            if (rnd && rnd->isTransparent())
                ++transparentCount;
            else
                ++solidCount;

            if (solidCount && transparentCount)
                break;
        }

		// register according to material types counted
		uint32_t taken = 0;
		if (solidCount)
			taken |= SceneManager->registerNodeForRendering(this, scene::ESNRP_SOLID);

		if (transparentCount)
			taken |= SceneManager->registerNodeForRendering(this, scene::ESNRP_TRANSPARENT);

        if (taken)
            RecullInstances();
	}
}


//! renders the node.
void CMeshSceneNodeInstanced::render()
{
	video::IVideoDriver* driver = SceneManager->getVideoDriver();

	if (LoD.size()==0 || !driver)
		return;

	bool isTransparentPass =
		SceneManager->getSceneNodeRenderPass() == scene::ESNRP_TRANSPARENT;

	++PassCount;

	driver->setTransform(video::E4X3TS_WORLD, AbsoluteTransformation);


	if (flagQueryForRetrieval)
    {
/*#ifdef _IRR_DEBUG
        if (!LoD[LoD.size()-1].query->isQueryReady())
            os::Printer::log("GPU Culling Feedback Transform Instance Count Query Not Ready yet, STALLING CPU!\n",ELL_WARNING);
#endif // _IRR_DEBUG*/
        for (size_t j=0; j<LoD.size(); j++)
        {
            uint32_t tmp;
            LoD[j].query->getQueryResult(&tmp);
            for (size_t i=0; i<LoD[j].mesh->getMeshBufferCount(); i++)
            {
                LoD[j].mesh->getMeshBuffer(i)->setInstanceCount(tmp);
                LoD[j].mesh->getMeshBuffer(i)->setBaseInstance(xfb[j/gpuLoDsPerPass]->getOutputBufferOffset(j%gpuLoDsPerPass)/dataPerInstanceOutputSize);
            }
        }
        flagQueryForRetrieval = false;
    }

    for (uint32_t i=0; i<LoD.size(); ++i)
    {
        for (size_t j=0; j<LoD[i].mesh->getMeshBufferCount(); j++)
        {
            const video::SGPUMaterial& material = LoD[i].mesh->getMeshBuffer(j)->getMaterial();

            video::IMaterialRenderer* rnd = driver->getMaterialRenderer(material.MaterialType);
            bool transparent = (rnd && rnd->isTransparent());

            // only render transparent buffer if this is the transparent render pass
            // and solid only in solid pass
            if (transparent == isTransparentPass)
            {
                driver->setMaterial(material);
                driver->drawMeshBuffer(LoD[i].mesh->getMeshBuffer(j));
            }
        }
    }
}



//! Creates a clone of this scene node and its children.
ISceneNode* CMeshSceneNodeInstanced::clone(IDummyTransformationSceneNode* newParent, ISceneManager* newManager)
{
	if (!newParent)
		newParent = Parent;
	if (!newManager)
		newManager = SceneManager;

    return NULL;
/**
	CMeshSceneNodeInstanced* nb = new CMeshSceneNodeInstanced(Mesh, newParent,
		newManager, ID, RelativeTranslation, RelativeRotation, RelativeScale);

	nb->cloneMembers(this, newManager);
	nb->ReferencingMeshMaterials = ReferencingMeshMaterials;
	nb->Materials = Materials;

	if (newParent)
		nb->drop();
	return nb;**/
}


} // end namespace scene
} // end namespace irr


