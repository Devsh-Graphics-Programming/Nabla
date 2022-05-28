// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_BULLET_C_DEBUG_RENDER_INCLUDED_
#define _NBL_EXT_BULLET_C_DEBUG_RENDER_INCLUDED_

#include "nabla.h"
#include "nbl/core/IReferenceCounted.h"
#include "btBulletDynamicsCommon.h"

#include "nbl/ext/DebugDraw/CDraw3DLine.h"

namespace nbl
{
namespace ext
{
namespace DebugDraw {
    class CDraw3DLine;
}

namespace Bullet3
{



class NBL_API CDebugRender : public btIDebugDraw
{
public:
    CDebugRender(nbl::video::IVideoDriver *driver);


    virtual void draw();
   
    virtual void clearLines();
    virtual void drawLine(const btVector3 &from, const btVector3 &to, const btVector3 &color) override;
   
    virtual void drawContactPoint(const btVector3& PointOnB, const btVector3& normalOnB, btScalar distance, int lifeTime, const btVector3& color);
    virtual void reportErrorWarning(const char* warningString);
    virtual void draw3dText(const btVector3& location, const char* textString);
    virtual void setDebugMode(int debugMode);
    virtual int	getDebugMode() const;

    
protected:
    
    nbl::video::IVideoDriver *m_driver;
    core::smart_refctd_ptr<nbl::ext::DebugDraw::CDraw3DLine> m_lineRender;
    core::vector<std::pair<DebugDraw::S3DLineVertex, DebugDraw::S3DLineVertex>> m_scene;

    int m_drawMode;
};


}
}
}

#endif