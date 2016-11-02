// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CMeshSceneNodeInstanced.h"
#include "IVideoDriver.h"
#include "ISceneManager.h"
#include "ICameraSceneNode.h"
#include "IMaterialRenderer.h"
#include "os.h"

namespace irr
{
namespace scene
{


uint32_t CMeshSceneNodeInstanced::recullOrder;


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

    if (instanceDataBuffer)
        instanceDataBuffer->drop();
    if (instanceBBoxes)
        free(instanceBBoxes);
    if (culledLodInstanceDataBuffer)
        culledLodInstanceDataBuffer->drop();
}

void inputBuffersOnTransformFeedback(video::IGPUBuffer* buff, void* node)
{
    CMeshSceneNodeInstanced* actualObject = reinterpret_cast<CMeshSceneNodeInstanced*>(node);
    IGPUMeshDataFormatDesc* vao = actualObject->lodCullingPointVAO;

    uint32_t floatComponents = actualObject->extraDataInstanceSize+1;
    floatComponents /= 4;
    floatComponents += 12+9;
    if (floatComponents>EVAI_COUNT*4)
    {
        for (uint32_t i=0; i<EVAI_COUNT; i++)
            vao->mapVertexAttrBuffer(buff,(E_VERTEX_ATTRIBUTE_ID)i,ECPA_FOUR,ECT_FLOAT,actualObject->extraDataInstanceSize+12*4+36+actualObject->visibilityPadding,i*16);
        return;
    }

    size_t memoryUsed = 0;
    uint32_t attr = 0;
    for (; attr*4+3<floatComponents; attr++)
    {
        vao->mapVertexAttrBuffer(buff,(E_VERTEX_ATTRIBUTE_ID)attr,ECPA_FOUR,ECT_FLOAT,actualObject->extraDataInstanceSize+12*4+36+actualObject->visibilityPadding,attr*16);
        memoryUsed+=16;
    }
    memoryUsed -= (12+9)*4;

    size_t leftOverMemory = actualObject->extraDataInstanceSize+1-memoryUsed;
    //assume a padding of 4 at the end
    vao->mapVertexAttrBuffer(buff,(E_VERTEX_ATTRIBUTE_ID)attr,(E_COMPONENTS_PER_ATTRIBUTE)((leftOverMemory+3)/4),ECT_INTEGER_UNSIGNED_INT,actualObject->extraDataInstanceSize+12*4+36+actualObject->visibilityPadding,attr*16);
}


//! Sets a new meshbuffer
bool CMeshSceneNodeInstanced::setLoDMeshes(std::vector<MeshLoD> levelsOfDetail, const size_t& dataSizePerInstanceOutput, const video::SMaterial& lodSelectionShader, IGPUMeshDataFormatDesc* (*vaoSetupOverride)(ISceneManager*,video::IGPUBuffer*,const std::vector<MeshLoD>&,const size_t&,const size_t&,const size_t&), const size_t& extraDataSizePerInstanceInput)
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

    if (instanceDataBuffer)
        instanceDataBuffer->drop();
    if (instanceBBoxes)
        free(instanceBBoxes);
    if (culledLodInstanceDataBuffer)
        culledLodInstanceDataBuffer->drop();
    instanceDataBuffer = NULL;
    instanceBBoxes = NULL;
    culledLodInstanceDataBuffer = NULL;
    extraDataInstanceSize = 0;

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

    visibilityPadding = 4-(extraDataInstanceSize&0x3u);

	instanceDataBuffer = new video::IMetaGranularGPUMappedBuffer(SceneManager->getVideoDriver(),extraDataSizePerInstanceInput+visibilityPadding+48+36,4096,false,16*1024,16*1024);
	instanceBBoxesCount = instanceDataBuffer->getCapacity();
	instanceBBoxes = (core::aabbox3df*)malloc(instanceBBoxesCount*sizeof(core::aabbox3df));
	for (size_t i=0; i<instanceBBoxesCount; i++)
    {
        instanceBBoxes[i].MinEdge.set( FLT_MAX, FLT_MAX, FLT_MAX);
        instanceBBoxes[i].MaxEdge.set(-FLT_MAX,-FLT_MAX,-FLT_MAX);
    }
	culledLodInstanceDataBuffer = SceneManager->getVideoDriver()->createGPUBuffer(dataSizePerInstanceOutput*instanceBBoxesCount*levelsOfDetail.size(),NULL);
	instanceDataBufferChanged = false;

	extraDataInstanceSize = extraDataSizePerInstanceInput;
	dataPerInstanceOutputSize = dataSizePerInstanceOutput;
	inputBuffersOnTransformFeedback(instanceDataBuffer->getFrontBuffer(),this);


    for (size_t i=0; i<levelsOfDetail.size(); i++)
    {
        cachedMaterialCount += levelsOfDetail[i].mesh->getMeshBufferCount();

        LoDData tmp;
        tmp.distanceSQ = levelsOfDetail[i].lodDistance;
        tmp.distanceSQ *= tmp.distanceSQ;

        tmp.mesh = new SGPUMesh();
        for (size_t j=0; j<levelsOfDetail[i].mesh->getMeshBufferCount(); j++)
        {
            IGPUMeshBuffer* meshBuff = new IGPUMeshBuffer();
            meshBuff->setBaseVertex(levelsOfDetail[i].mesh->getMeshBuffer(j)->getBaseVertex());
            if (levelsOfDetail[i].mesh->getMeshBuffer(j)->isIndexCountGivenByXFormFeedback())
                meshBuff->setIndexCountFromXFormFeedback(levelsOfDetail[i].mesh->getMeshBuffer(j)->getXFormFeedback());
            else
                meshBuff->setIndexCount(levelsOfDetail[i].mesh->getMeshBuffer(j)->getIndexCount());
            meshBuff->setIndexBufferOffset(levelsOfDetail[i].mesh->getMeshBuffer(j)->getIndexBufferOffset());
            meshBuff->setIndexType(levelsOfDetail[i].mesh->getMeshBuffer(j)->getIndexType());
            meshBuff->setPrimitiveType(levelsOfDetail[i].mesh->getMeshBuffer(j)->getPrimitiveType());

            IGPUMeshDataFormatDesc* vao = vaoSetupOverride(SceneManager,culledLodInstanceDataBuffer,levelsOfDetail,dataSizePerInstanceOutput,i,j);
            meshBuff->setMeshDataAndFormat(vao);
            vao->drop();

            meshBuff->getMaterial() = levelsOfDetail[i].mesh->getMeshBuffer(j)->getMaterial();
            meshBuff->setBoundingBox(levelsOfDetail[i].mesh->getMeshBuffer(j)->getBoundingBox());
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

        video::ITransformFeedback* xformFeedback = SceneManager->getVideoDriver()->createTransformFeedback();
        xformFeedback->bindOutputBuffer(0,culledLodInstanceDataBuffer,i*culledLodInstanceDataBuffer->getSize()/levelsOfDetail.size(),culledLodInstanceDataBuffer->getSize()/levelsOfDetail.size());
        xfb.push_back(xformFeedback);
    }

    lodCullingPointMesh->getMaterial() = lodSelectionShader;

    return true;
}

uint32_t CMeshSceneNodeInstanced::addInstance(const core::matrix4x3& relativeTransform, void* extraData)
{
    uint32_t ix;
    if (!addInstances(&ix,1,&relativeTransform,extraData))
        return 0xdeadbeefu;

    return ix;
}

bool CMeshSceneNodeInstanced::addInstances(uint32_t* instanceIDs, const size_t& instanceCount, const core::matrix4x3* relativeTransforms, void* extraData)
{
    if (!instanceDataBuffer->Alloc(instanceIDs,instanceCount))
    {
        for (size_t i=0; i<instanceCount; i++)
                instanceIDs[i] = 0xdeadbeefu;
        return false;
    }
    if (instanceDataBuffer->getCapacity()!=instanceBBoxesCount)
    {
        size_t newCount = instanceDataBuffer->getCapacity();
        instanceBBoxes = (core::aabbox3df*)realloc(instanceBBoxes,newCount*sizeof(core::aabbox3df));
        for (size_t i=instanceBBoxesCount; i<newCount; i++)
        {
            instanceBBoxes[i].MinEdge.set( FLT_MAX, FLT_MAX, FLT_MAX);
            instanceBBoxes[i].MaxEdge.set(-FLT_MAX,-FLT_MAX,-FLT_MAX);
        }
        instanceBBoxesCount = newCount;
    }
    needsBBoxRecompute = true;
    instanceDataBufferChanged = true;

    uint8_t* base_pointer = reinterpret_cast<uint8_t*>(instanceDataBuffer->getBackBufferPointer());
    for (size_t i=0; i<instanceCount; i++)
    {
        size_t redirect = instanceDataBuffer->getRedirectFromID(instanceIDs[i]);
        {
            instanceBBoxes[instanceIDs[i]] = LoDInvariantBox;
            relativeTransforms[i].transformBoxEx(instanceBBoxes[instanceIDs[i]]);
        }
        uint8_t* ptr = base_pointer+redirect*(extraDataInstanceSize+12*4+36+visibilityPadding);
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
            memcpy(ptr+48+36,reinterpret_cast<uint8_t*>(extraData)+extraDataInstanceSize*i,extraDataInstanceSize);
        ptr[48+36+extraDataInstanceSize] = 0xffu;
    }

    lodCullingPointMesh->setIndexCount(lodCullingPointMesh->getIndexCount()+instanceCount);

    return true;
}

void CMeshSceneNodeInstanced::setInstanceTransform(const uint32_t& instanceID, const core::matrix4x3& relativeTransform)
{
    size_t redirect = instanceDataBuffer->getRedirectFromID(instanceID);
    {
        instanceBBoxes[instanceID] = LoDInvariantBox;
        relativeTransform.transformBoxEx(instanceBBoxes[instanceID]);
    }

    uint8_t* ptr = reinterpret_cast<uint8_t*>(instanceDataBuffer->getBackBufferPointer())+redirect*(extraDataInstanceSize+12*4+36+visibilityPadding);
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

    instanceDataBufferChanged = true;
    needsBBoxRecompute = true;
}

core::matrix4x3 CMeshSceneNodeInstanced::getInstanceTransform(const uint32_t& instanceID)
{
    size_t redir = instanceDataBuffer->getRedirectFromID(instanceID);
    if (redir==0xdeadbeefu)
    {
        core::matrix4x3 retval(core::matrix4x3::EM4CONST_NOTHING);
        memset(retval.pointer(),0,48);
        return retval;
    }
    else
    {
        return reinterpret_cast<core::matrix4x3*>(reinterpret_cast<uint8_t*>(instanceDataBuffer->getBackBufferPointer())+redir*(extraDataInstanceSize+12*4+36+visibilityPadding))[0];
    }
}

void CMeshSceneNodeInstanced::setInstanceVisible(const uint32_t& instanceID, const bool& visible)
{
    reinterpret_cast<uint8_t*>(instanceDataBuffer->getBackBufferPointer())[instanceDataBuffer->getRedirectFromID(instanceID)*(extraDataInstanceSize+12*4+36+visibilityPadding)+36+48+extraDataInstanceSize] = visible;
    instanceDataBufferChanged = true;
    /// update BBox?
}

void CMeshSceneNodeInstanced::setInstanceData(const uint32_t& instanceID, void* data)
{
    if (extraDataInstanceSize==0)
        return;

    uint8_t* ptr = reinterpret_cast<uint8_t*>(instanceDataBuffer->getBackBufferPointer())+instanceDataBuffer->getRedirectFromID(instanceID)*(extraDataInstanceSize+12*4+36+visibilityPadding)+48+36;
    memcpy(ptr,data,extraDataInstanceSize);
    instanceDataBufferChanged = true;
}

void CMeshSceneNodeInstanced::removeInstance(const uint32_t& instanceID)
{
    removeInstances(1,&instanceID);
}

void CMeshSceneNodeInstanced::removeInstances(const size_t& instanceCount, const uint32_t* instanceIDs)
{
    for (size_t i=0; i<instanceCount; i++)
    {
        size_t redirect = instanceDataBuffer->getRedirectFromID(instanceIDs[i]);
        instanceBBoxes[instanceIDs[i]].MinEdge.set( FLT_MAX, FLT_MAX, FLT_MAX);
        instanceBBoxes[instanceIDs[i]].MaxEdge.set(-FLT_MAX,-FLT_MAX,-FLT_MAX);
    }

    instanceDataBuffer->Free(instanceIDs,instanceCount);
    if (instanceDataBuffer->getCapacity()!=instanceBBoxesCount)
    {
        size_t newCount = instanceDataBuffer->getCapacity();
        instanceBBoxes = (core::aabbox3df*)realloc(instanceBBoxes,newCount*sizeof(core::aabbox3df));
        for (size_t i=instanceBBoxesCount; i<newCount; i++)
        {
            instanceBBoxes[i].MinEdge.set( FLT_MAX, FLT_MAX, FLT_MAX);
            instanceBBoxes[i].MaxEdge.set(-FLT_MAX,-FLT_MAX,-FLT_MAX);
        }
        instanceBBoxesCount = newCount;
    }
    needsBBoxRecompute = true;
    instanceDataBufferChanged = true;

    lodCullingPointMesh->setIndexCount(lodCullingPointMesh->getIndexCount()-instanceCount);
}

void CMeshSceneNodeInstanced::RecullInstances()
{
    if (LoD.size()==0||!instanceDataBuffer||instanceDataBuffer->getAllocatedCount()==0||!SceneManager)
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

    //can swap before or after, but defubuteky before tform feedback shadeur
    if (instanceDataBufferChanged)
        instanceDataBuffer->SwapBuffers();

    size_t outputSizePerLoD = dataPerInstanceOutputSize*instanceDataBuffer->getCapacity();
    if (culledLodInstanceDataBuffer->getSize()!=LoD.size()*outputSizePerLoD)
    {
        culledLodInstanceDataBuffer->reallocate(LoD.size()*outputSizePerLoD,false,true);
        for (size_t i=0; i<LoD.size(); i++)
            xfb[i]->bindOutputBuffer(0,culledLodInstanceDataBuffer,i*outputSizePerLoD,outputSizePerLoD);
    }

    driver->setTransform(video::E4X3TS_WORLD,AbsoluteTransformation);
    for (uint32_t i=0; i<LoD.size(); i++)
    {
        lodCullingPointMesh->getMaterial().MaterialTypeParam = reinterpret_cast<float&>(i);
        lodCullingPointMesh->getMaterial().MaterialTypeParam2 = reinterpret_cast<float&>(i);
        driver->setMaterial(lodCullingPointMesh->getMaterial());
        driver->beginTransformFeedback(xfb[i],lodCullingPointMesh->getMaterial().MaterialType,scene::EPT_POINTS);
        driver->beginQuery(LoD[i].query);
        driver->drawMeshBuffer(lodCullingPointMesh);
        driver->endQuery(LoD[i].query);
        driver->endTransformFeedback();
    }

	renderPriority = 0x80000000u-(++recullOrder);
    flagQueryForRetrieval = true;
}

//! frame
void CMeshSceneNodeInstanced::OnRegisterSceneNode()
{
    ISceneNode::OnRegisterSceneNode();

	if (IsVisible&&LoD.size()&&instanceDataBuffer&&instanceDataBuffer->getAllocatedCount())
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
            scene::IGPUMeshBuffer* mb = LoD[i].mesh->getMeshBuffer(j);
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
/*#ifdef _DEBUG
        if (!LoD[LoD.size()-1].query->isQueryReady())
            os::Printer::log("GPU Culling Feedback Transform Instance Count Query Not Ready yet, STALLING CPU!\n",ELL_WARNING);
#endif // _DEBUG*/
        for (size_t j=0; j<LoD.size(); j++)
        {
            uint32_t tmp;
            LoD[j].query->getQueryResult(&tmp);
            for (size_t i=0; i<LoD[j].mesh->getMeshBufferCount(); i++)
            {
                LoD[j].mesh->getMeshBuffer(i)->setInstanceCount(tmp);
                LoD[j].mesh->getMeshBuffer(i)->setBaseInstance(xfb[j]->getOutputBufferOffset(0)/dataPerInstanceOutputSize);
            }
        }
        flagQueryForRetrieval = false;
    }

    for (u32 i=0; i<LoD.size(); ++i)
    {
        for (size_t j=0; j<LoD[i].mesh->getMeshBufferCount(); j++)
        {
            const video::SMaterial& material = LoD[i].mesh->getMeshBuffer(j)->getMaterial();

            video::IMaterialRenderer* rnd = driver->getMaterialRenderer(material.MaterialType);
            bool transparent = (rnd && rnd->isTransparent());

            // only render transparent buffer if this is the transparent render pass
            // and solid only in solid pass
            if (transparent == isTransparentPass)
            {
                driver->setMaterial(material);
                driver->drawMeshBuffer(LoD[i].mesh->getMeshBuffer(j), (AutomaticCullingState & scene::EAC_COND_RENDER) ? query:NULL);
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


