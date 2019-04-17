// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CSkyBoxSceneNode.h"
#include "IVideoDriver.h"
#include "ISceneManager.h"
#include "ICameraSceneNode.h"

#include "os.h"

namespace irr
{
namespace scene
{


//! constructor
CSkyBoxSceneNode::CSkyBoxSceneNode(video::ITexture* top, video::ITexture* bottom, video::ITexture* left,
			video::ITexture* right, video::ITexture* front, video::ITexture* back,
			video::IGPUBuffer* vertPositions, size_t positionsOffsetInBuf,
			IDummyTransformationSceneNode* parent, ISceneManager* mgr, int32_t id)
: ISceneNode(parent, mgr, id)
{
	#ifdef _IRR_DEBUG
	setDebugName("CSkyBoxSceneNode");
	#endif

	setAutomaticCulling(scene::EAC_OFF);
	Box.MaxEdge.set(0,0,0);
	Box.MinEdge.set(0,0,0);


	// create material

	video::SGPUMaterial mat;
	mat.ZBuffer = video::ECFN_NEVER;
	mat.ZWriteEnable = false;
	mat.TextureLayer[0].SamplingParams.TextureWrapU = video::ETC_CLAMP_TO_EDGE;
	mat.TextureLayer[0].SamplingParams.TextureWrapV = video::ETC_CLAMP_TO_EDGE;

	/* Hey, I am no artist, but look at that
	   cool ASCII art I made! ;)

       -111         111
          /6--------/5        y
         /  |      / |        ^  z
        /   |   11-1 |        | /
  -11-1 3---------2  |        |/
        |   7- - -| -4 1-11    *---->x
        | -1-11   |  /       3-------|2
        |/        | /         |    //|
        0---------1/          |  //  |
     -1-1-1     1-1-1         |//    |
	                     0--------1
	*/

	video::ITexture* tex = front;
	if (!tex) tex = left;
	if (!tex) tex = back;
	if (!tex) tex = right;
	if (!tex) tex = top;
	if (!tex) tex = bottom;

	const float onepixel = tex?(1.0f / (tex->getSize()[0] * 1.5f)) : 0.0f;
	const float t = 1.0f - onepixel;
	const float o = 0.0f + onepixel;

	video::IVideoDriver* driver = SceneManager->getVideoDriver();
	// create front side
	Material[0] = mat;
	Material[0].setTexture(0, front);
	float texcoords[2*4*2];
	texcoords[0] = t;
	texcoords[1] = t;
	texcoords[2] = o;
	texcoords[3] = t;
	texcoords[4] = o;
	texcoords[5] = o;
	texcoords[6] = t;
	texcoords[7] = o;
	// one odd side
	texcoords[8] = o;
	texcoords[9] = o;
	texcoords[10] = t;
	texcoords[11] = o;
	texcoords[12] = t;
	texcoords[13] = t;
	texcoords[14] = o;
	texcoords[15] = t;
	video::IDriverMemoryBacked::SDriverMemoryRequirements reqs;
	reqs.vulkanReqs.size = sizeof(texcoords);
	reqs.vulkanReqs.alignment = 4;
	reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
	reqs.memoryHeapLocation = video::IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
	reqs.mappingCapability = video::IDriverMemoryAllocation::EMCAF_NO_MAPPING_ACCESS;
	reqs.prefersDedicatedAllocation = true;
	reqs.requiresDedicatedAllocation = true;
    video::IGPUBuffer* texcoordBuf = SceneManager->getVideoDriver()->createGPUBufferOnDedMem(reqs,true);
    texcoordBuf->updateSubRange(video::IDriverMemoryAllocation::MemoryRange(0,reqs.vulkanReqs.size),texcoords);

	// create left side
	Material[1] = mat;
	Material[1].setTexture(0, left);

	// create back side
	Material[2] = mat;
	Material[2].setTexture(0, back);

	// create right side
	Material[3] = mat;
	Material[3].setTexture(0, right);

	// create top side
	Material[4] = mat;
	Material[4].setTexture(0, top);

	// create bottom side
	Material[5] = mat;
	Material[5].setTexture(0, bottom);


	for (size_t i=0; i<6; i++)
    {
        sides[i] = new video::IGPUMeshBuffer();
        sides[i]->setPrimitiveType(asset::EPT_TRIANGLE_FAN);
        sides[i]->setIndexCount(4);
        video::IGPUMeshDataFormatDesc* desc = driver->createGPUMeshDataFormatDesc();
        sides[i]->setMeshDataAndFormat(desc);
        desc->drop();
    }
    sides[0]->getMeshDataAndFormat()->setVertexAttrBuffer(vertPositions,asset::EVAI_ATTR0,asset::EF_R8G8B8_SSCALED,0,positionsOffsetInBuf);
    sides[0]->getMeshDataAndFormat()->setVertexAttrBuffer(texcoordBuf,asset::EVAI_ATTR2,asset::EF_R32G32_SFLOAT);
    sides[1]->getMeshDataAndFormat()->setVertexAttrBuffer(vertPositions,asset::EVAI_ATTR0,asset::EF_R8G8B8_SSCALED,0,positionsOffsetInBuf+3*4*1);
    sides[1]->getMeshDataAndFormat()->setVertexAttrBuffer(texcoordBuf,asset::EVAI_ATTR2,asset::EF_R32G32_SFLOAT);
    sides[2]->getMeshDataAndFormat()->setVertexAttrBuffer(vertPositions,asset::EVAI_ATTR0,asset::EF_R8G8B8_SSCALED,0,positionsOffsetInBuf+3*4*2);
    sides[2]->getMeshDataAndFormat()->setVertexAttrBuffer(texcoordBuf,asset::EVAI_ATTR2,asset::EF_R32G32_SFLOAT);
    sides[3]->getMeshDataAndFormat()->setVertexAttrBuffer(vertPositions,asset::EVAI_ATTR0,asset::EF_R8G8B8_SSCALED,0,positionsOffsetInBuf+3*4*3);
    sides[3]->getMeshDataAndFormat()->setVertexAttrBuffer(texcoordBuf,asset::EVAI_ATTR2,asset::EF_R32G32_SFLOAT);
    sides[4]->getMeshDataAndFormat()->setVertexAttrBuffer(vertPositions,asset::EVAI_ATTR0,asset::EF_R8G8B8_SSCALED,0,positionsOffsetInBuf+3*4*4);
    sides[4]->getMeshDataAndFormat()->setVertexAttrBuffer(texcoordBuf,asset::EVAI_ATTR2,asset::EF_R32G32_SFLOAT);
    sides[5]->getMeshDataAndFormat()->setVertexAttrBuffer(vertPositions,asset::EVAI_ATTR0,asset::EF_R8G8B8_SSCALED,0,positionsOffsetInBuf+3*4*5);
    sides[5]->getMeshDataAndFormat()->setVertexAttrBuffer(texcoordBuf,asset::EVAI_ATTR2,asset::EF_R32G32_SFLOAT,0,2*4*sizeof(float));
    texcoordBuf->drop();
}

CSkyBoxSceneNode::CSkyBoxSceneNode(CSkyBoxSceneNode* other,
			IDummyTransformationSceneNode* parent, ISceneManager* mgr, int32_t id)
: ISceneNode(parent, mgr, id)
{
	#ifdef _IRR_DEBUG
	setDebugName("CSkyBoxSceneNode");
	#endif

	setAutomaticCulling(scene::EAC_OFF);
	Box.MaxEdge.set(0,0,0);
	Box.MinEdge.set(0,0,0);


	for (size_t i=0; i<6; i++)
    {
        other->sides[i]->grab();
        sides[i] = other->sides[i];
		Material[i] = other->Material[i];
    }
}


//! renders the node.
void CSkyBoxSceneNode::render()
{
	video::IVideoDriver* driver = SceneManager->getVideoDriver();
	scene::ICameraSceneNode* camera = SceneManager->getActiveCamera();

	if (!camera || !driver || !canProceedPastFence())
		return;

	if ( !camera->getProjectionMatrix().isOrthogonal() ) // check this actually works!
	{
		// draw perspective skybox

		core::matrix4x3 translate(AbsoluteTransformation);
		translate.setTranslation(camera->getAbsolutePosition());

		// Draw the sky box between the near and far clip plane
		const float viewDistance = (camera->getNearValue() + camera->getFarValue()) * 0.5f;
		core::matrix4x3 scale;
		scale.setScale(core::vector3df(viewDistance, viewDistance, viewDistance));

		driver->setTransform(video::E4X3TS_WORLD, concatenateBFollowedByA(translate,scale));

		for (int32_t i=0; i<6; ++i)
		{
			driver->setMaterial(Material[i]);
			driver->drawMeshBuffer(sides[i]);
		}
	}
	else
	{
		// draw orthogonal skybox,
		// simply choose one texture and draw it as 2d picture.
		// there could be better ways to do this, but currently I think this is ok.

		core::vector3df lookVect = camera->getTarget() - camera->getAbsolutePosition();
		lookVect.normalize();
		core::vector3df absVect( core::abs_(lookVect.X),
					 core::abs_(lookVect.Y),
					 core::abs_(lookVect.Z));

		int idx = 0;

		if ( absVect.X >= absVect.Y && absVect.X >= absVect.Z )
		{
			// x direction
			idx = lookVect.X > 0 ? 0 : 2;
		}
		else
		if ( absVect.Y >= absVect.X && absVect.Y >= absVect.Z )
		{
			// y direction
			idx = lookVect.Y > 0 ? 4 : 5;
		}
		else
		if ( absVect.Z >= absVect.X && absVect.Z >= absVect.Y )
		{
			// z direction
			idx = lookVect.Z > 0 ? 1 : 3;
		}

		video::IVirtualTexture* vtex = Material[idx].getTexture(0);

		if ( vtex && vtex->getVirtualTextureType()==video::IVirtualTexture::EVTT_OPAQUE_FILTERABLE )
		{
		    video::ITexture* texture = static_cast<video::ITexture*>(vtex);

			core::rect<int32_t> rctDest(core::position2d<int32_t>(-1,0),
									core::dimension2di(driver->getCurrentRenderTargetSize()));
			core::rect<int32_t> rctSrc(core::position2d<int32_t>(0,0),
									core::dimension2di(*reinterpret_cast<const core::dimension2du*>(texture->getSize())));

			driver->draw2DImage(texture, rctDest, rctSrc);
		}
	}
}



//! returns the axis aligned bounding box of this node
const core::aabbox3d<float>& CSkyBoxSceneNode::getBoundingBox()
{
	return Box;
}


void CSkyBoxSceneNode::OnRegisterSceneNode()
{
	if (IsVisible)
		SceneManager->registerNodeForRendering(this, ESNRP_SKY_BOX);

	ISceneNode::OnRegisterSceneNode();
}


//! returns the material based on the zero based index i. To get the amount
//! of materials used by this scene node, use getMaterialCount().
//! This function is needed for inserting the node into the scene hirachy on a
//! optimal position for minimizing renderstate changes, but can also be used
//! to directly modify the material of a scene node.
video::SGPUMaterial& CSkyBoxSceneNode::getMaterial(uint32_t i)
{
	return Material[i];
}


//! returns amount of materials used by this scene node.
uint32_t CSkyBoxSceneNode::getMaterialCount() const
{
	return 6;
}


//! Creates a clone of this scene node and its children.
ISceneNode* CSkyBoxSceneNode::clone(IDummyTransformationSceneNode* newParent, ISceneManager* newManager)
{
	if (!newParent) newParent = Parent;
	if (!newManager) newManager = SceneManager;

	CSkyBoxSceneNode* nb = new CSkyBoxSceneNode(this, newParent,
		newManager, ID);

	nb->cloneMembers(this, newManager);

	if ( newParent )
		nb->drop();
	return nb;
}

} // end namespace scene
} // end namespace irr

