// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_BILLBOARD_SCENE_NODE_H_INCLUDED__
#define __C_BILLBOARD_SCENE_NODE_H_INCLUDED__

#include "IBillboardSceneNode.h"
#include "IMeshBuffer.h"

namespace irr
{
namespace scene
{

//! Scene node which is a billboard. A billboard is like a 3d sprite: A 2d element,
//! which always looks to the camera.
class CBillboardSceneNode : virtual public IBillboardSceneNode
{
public:

	//! constructor
	CBillboardSceneNode(ISceneNode* parent, ISceneManager* mgr, s32 id,
		const core::vector3df& position, const core::dimension2d<f32>& size,
		video::SColor colorTop=video::SColor(0xFFFFFFFF),
		video::SColor colorBottom=video::SColor(0xFFFFFFFF));

	//! pre render event
	virtual void OnRegisterSceneNode();

	//! render
	virtual void render();

	//! returns the axis aligned bounding box of this node
	virtual const core::aabbox3d<f32>& getBoundingBox() const {return BBox;}

	//! sets the size of the billboard
	virtual void setSize(const core::dimension2d<f32>& size);

	//! Sets the widths of the top and bottom edges of the billboard independently.
	virtual void setSize(f32 height, f32 bottomEdgeWidth, f32 topEdgeWidth);

 	//! gets the size of the billboard
 	virtual const core::dimension2d<f32>& getSize() const {return Size;}

	//! Gets the widths of the top and bottom edges of the billboard.
	virtual void getSize(f32& height, f32& bottomEdgeWidth, f32& topEdgeWidth) const;

	virtual video::SMaterial& getMaterial(u32 i) {return Material;}

	//! returns amount of materials used by this scene node.
	virtual u32 getMaterialCount() const {return 1;}

	//! Set the color of all vertices of the billboard
	//! \param overallColor: the color to set
	virtual void setColor(const video::SColor& overallColor);

	//! Set the color of the top and bottom vertices of the billboard
	//! \param topColor: the color to set the top vertices
	//! \param bottomColor: the color to set the bottom vertices
	virtual void setColor(const video::SColor& topColor,
			const video::SColor& bottomColor);

	//! Returns type of the scene node
	virtual ESCENE_NODE_TYPE getType() const { return ESNT_BILLBOARD; }

	//! Creates a clone of this scene node and its children.
	virtual ISceneNode* clone(ISceneNode* newParent=0, ISceneManager* newManager=0);

private:

	//! Size.Width is the bottom edge width
	core::dimension2d<f32> Size;
	f32 TopEdgeWidth;
	core::aabbox3d<f32> BBox;
	video::SMaterial Material;

    video::IGPUBuffer* vertexBuffer;
    IGPUMeshDataFormatDesc* desc;
	IGPUMeshBuffer* meshbuffer;
};


} // end namespace scene
} // end namespace irr

#endif

