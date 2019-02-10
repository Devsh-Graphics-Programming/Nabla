#ifndef _IRR_EXT_BULLET_C_DEBUG_RENDER_INCLUDED_
#define _IRR_EXT_BULLET_C_DEBUG_RENDER_INCLUDED_

#include "irrlicht.h"
#include "irr/core/IReferenceCounted.h"
#include "btBulletDynamicsCommon.h"

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

    virtual void clear();
    virtual void draw();
   

    virtual void drawLine(const btVector3 &from, const btVector3 &to, const btVector3 &color) override;
   
    virtual void drawContactPoint(const btVector3& PointOnB, const btVector3& normalOnB, btScalar distance, int lifeTime, const btVector3& color);
    virtual void reportErrorWarning(const char* warningString);
    virtual void draw3dText(const btVector3& location, const char* textString);
    virtual void setDebugMode(int debugMode);
    virtual int	getDebugMode() const;

    
protected:
    struct DebugVertex
    {
        float pos[3];
        uint8_t col[3];
        uint8_t padding;

    } PACK_STRUCT;
#include "irr/irrunpack.h"

    std::vector<DebugVertex> m_scene;

    video::IVideoDriver *m_driver;
    scene::IGPUMeshBuffer *m_mesh;
    scene::IGPUMeshDataFormatDesc *m_desc;

    video::SMaterial m_material;

};


}
}
}

#endif