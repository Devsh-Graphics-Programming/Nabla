// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CBillboardSceneNode.h"
#include "IVideoDriver.h"
#include "ISceneManager.h"
#include "ICameraSceneNode.h"
#include "os.h"

namespace irr
{
namespace scene
{

//! constructor
CBillboardSceneNode::CBillboardSceneNode(ISceneNode* parent, ISceneManager* mgr, s32 id,
			const core::vector3df& position, const core::dimension2d<f32>& size,
			video::SColor colorTop, video::SColor colorBottom)
	: IBillboardSceneNode(parent, mgr, id, position)
{
	#ifdef _DEBUG
	setDebugName("CBillboardSceneNode");
	#endif

	setSize(size);

    meshbuffer = new IGPUMeshBuffer();
    meshbuffer->setIndexCount(4);
    meshbuffer->setPrimitiveType(EPT_TRIANGLE_STRIP);

    desc = SceneManager->getVideoDriver()->createGPUMeshDataFormatDesc();
    meshbuffer->setMeshDataAndFormat(desc);
    desc->drop();

    vertexBuffer = SceneManager->getVideoDriver()->createGPUBuffer(4*sizeof(float)*3,NULL,true,true);
    desc->mapVertexAttrBuffer(vertexBuffer,EVAI_ATTR0,ECPA_THREE,ECT_FLOAT);
    vertexBuffer->drop();

    setColor(colorTop,colorBottom);
}


//! pre render event
void CBillboardSceneNode::OnRegisterSceneNode()
{
	if (IsVisible)
		SceneManager->registerNodeForRendering(this);

	ISceneNode::OnRegisterSceneNode();
}


//! render
void CBillboardSceneNode::render()
{
	video::IVideoDriver* driver = SceneManager->getVideoDriver();
	ICameraSceneNode* camera = SceneManager->getActiveCamera();

	if (!camera || !driver)
		return;

	// make billboard look to camera

	core::vector3df pos = getAbsolutePosition();

	core::vector3df campos = camera->getAbsolutePosition();
	core::vector3df target = camera->getTarget();
	core::vector3df up = camera->getUpVector();
	core::vector3df view = target - campos;
	view.normalize();

	core::vector3df horizontal = up.crossProduct(view);
	if ( horizontal.getLength() == 0 )
	{
		horizontal.set(up.Y,up.X,up.Z);
	}
	horizontal.normalize();
	core::vector3df topHorizontal = horizontal * 0.5f * TopEdgeWidth;
	horizontal *= 0.5f * Size.Width;

	// pointing down!
	core::vector3df vertical = horizontal.crossProduct(view);
	vertical.normalize();
	vertical *= 0.5f * Size.Height;


	core::vector3df vertices[4];
	vertices[0] = pos + horizontal + vertical;
	vertices[1] = pos - horizontal + vertical;
	vertices[2] = pos + topHorizontal - vertical;
	vertices[3] = pos - topHorizontal - vertical;

	vertexBuffer->updateSubRange(0,sizeof(vertices),vertices);

	// draw

	if (DebugDataVisible & scene::EDS_BBOX)
	{
		driver->setTransform(video::ETS_WORLD, AbsoluteTransformation);
		video::SMaterial m;
		driver->setMaterial(m);
		driver->draw3DBox(BBox, video::SColor(0,208,195,152));
	}

	driver->setTransform(video::ETS_WORLD, core::IdentityMatrix);

	driver->setMaterial(Material);

	driver->drawMeshBuffer(meshbuffer, (AutomaticCullingState & scene::EAC_COND_RENDER) ? query:NULL);
}


//! sets the size of the billboard
void CBillboardSceneNode::setSize(const core::dimension2d<f32>& size)
{
	Size = size;

	if (core::equals(Size.Width, 0.0f))
		Size.Width = 1.0f;
	TopEdgeWidth = Size.Width;

	if (core::equals(Size.Height, 0.0f))
		Size.Height = 1.0f;

	const f32 avg = (Size.Width + Size.Height)/6;
	BBox.MinEdge.set(-avg,-avg,-avg);
	BBox.MaxEdge.set(avg,avg,avg);
}


void CBillboardSceneNode::setSize(f32 height, f32 bottomEdgeWidth, f32 topEdgeWidth)
{
	Size.set(bottomEdgeWidth, height);
	TopEdgeWidth = topEdgeWidth;

	if (core::equals(Size.Height, 0.0f))
		Size.Height = 1.0f;

	if (core::equals(Size.Width, 0.f) && core::equals(TopEdgeWidth, 0.f))
	{
		Size.Width = 1.0f;
		TopEdgeWidth = 1.0f;
	}

	const f32 avg = (core::max_(Size.Width,TopEdgeWidth) + Size.Height)/6;
	BBox.MinEdge.set(-avg,-avg,-avg);
	BBox.MaxEdge.set(avg,avg,avg);
}


//! Gets the widths of the top and bottom edges of the billboard.
void CBillboardSceneNode::getSize(f32& height, f32& bottomEdgeWidth,
		f32& topEdgeWidth) const
{
	height = Size.Height;
	bottomEdgeWidth = Size.Width;
	topEdgeWidth = TopEdgeWidth;
}

//! Set the color of all vertices of the billboard
//! \param overallColor: the color to set
void CBillboardSceneNode::setColor(const video::SColor& overallColor)
{
    setColor(overallColor,overallColor);
}


//! Set the color of the top and bottom vertices of the billboard
//! \param topColor: the color to set the top vertices
//! \param bottomColor: the color to set the bottom vertices
void CBillboardSceneNode::setColor(const video::SColor& topColor,
		const video::SColor& bottomColor)
{
    uint8_t staticVxData[4*(4+2)];
    //colors
    ((video::SColor*)staticVxData)[0] = bottomColor;
    ((video::SColor*)staticVxData)[1] = bottomColor;
    ((video::SColor*)staticVxData)[2] = topColor;
    ((video::SColor*)staticVxData)[3] = topColor;
    //texcoords
    staticVxData[16+0] = 1;
    staticVxData[16+1] = 1;
    staticVxData[16+2] = 0;
    staticVxData[16+3] = 1;
    staticVxData[16+4] = 1;
    staticVxData[16+5] = 0;
    staticVxData[16+6] = 0;
    staticVxData[16+7] = 0;


    video::IGPUBuffer* buf = SceneManager->getVideoDriver()->createGPUBuffer(sizeof(staticVxData),staticVxData);
    desc->mapVertexAttrBuffer(buf,EVAI_ATTR1,ECPA_REVERSED_OR_BGRA,ECT_NORMALIZED_UNSIGNED_BYTE);
    desc->mapVertexAttrBuffer(buf,EVAI_ATTR2,ECPA_TWO,ECT_UNSIGNED_BYTE,0,16);
    buf->drop();
}



//! Creates a clone of this scene node and its children.
ISceneNode* CBillboardSceneNode::clone(ISceneNode* newParent, ISceneManager* newManager)
{
	if (!newParent)
		newParent = Parent;
	if (!newManager)
		newManager = SceneManager;

	CBillboardSceneNode* nb = new CBillboardSceneNode(newParent,
		newManager, ID, RelativeTranslation, Size);

	nb->cloneMembers(this, newManager);
	nb->Material = Material;
	nb->TopEdgeWidth = this->TopEdgeWidth;

	if ( newParent )
		nb->drop();
	return nb;
}


} // end namespace scene
} // end namespace irr

