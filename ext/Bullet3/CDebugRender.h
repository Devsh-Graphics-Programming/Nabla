#ifndef _IRR_EXT_BULLET_C_DEBUG_RENDER_INCLUDED_
#define _IRR_EXT_BULLET_C_DEBUG_RENDER_INCLUDED_

#include "irrlicht.h"
#include "irr/core/IReferenceCounted.h"
#include "btBulletDynamicsCommon.h"

#include "../../ext/DebugDraw/CDraw3DLine.h"

namespace irr
{
namespace ext
{
namespace DebugDraw {
    class CDraw3DLine;
}

namespace Bullet3
{



class CDebugRender : public btIDebugDraw
{
public:
    CDebugRender(irr::video::IVideoDriver *driver);
    ~CDebugRender();


    virtual void draw();
   
    virtual void clearLines();
    virtual void drawLine(const btVector3 &from, const btVector3 &to, const btVector3 &color) override;
   
    virtual void drawContactPoint(const btVector3& PointOnB, const btVector3& normalOnB, btScalar distance, int lifeTime, const btVector3& color);
    virtual void reportErrorWarning(const char* warningString);
    virtual void draw3dText(const btVector3& location, const char* textString);
    virtual void setDebugMode(int debugMode);
    virtual int	getDebugMode() const;

    
protected:
    
    irr::video::IVideoDriver *m_driver;
    irr::ext::DebugDraw::CDraw3DLine *m_lineRender;
    core::vector<std::pair<DebugDraw::S3DLineVertex, DebugDraw::S3DLineVertex>> m_scene;

    int m_drawMode;
};


}
}
}

#endif