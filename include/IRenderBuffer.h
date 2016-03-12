// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_RENDER_BUFFER_H_INCLUDED__
#define __I_RENDER_BUFFER_H_INCLUDED__

#include "IFrameBuffer.h"
#include "dimension2d.h"

namespace irr
{
namespace video
{

class IRenderBuffer : public IRenderable
{
public:

	E_RENDERABLE_TYPE getRenderableType() const {return ERT_RENDERBUFFER;}

	virtual void resize(const core::dimension2du &newSize) = 0;

protected:
};


} // end namespace video
} // end namespace irr

#endif


