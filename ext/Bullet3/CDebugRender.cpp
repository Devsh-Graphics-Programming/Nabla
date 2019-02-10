#include "../../ext/Bullet3/CDebugRender.h"
#include "../../ext/Bullet3/BulletUtility.h"
#include "../../ext/DebugDraw/Draw3DLineShaders.h"


#include <iostream>

using namespace irr;
using namespace ext;
using namespace Bullet3;



class DebugRenderCallback : public video::IShaderConstantSetCallBack
{
    int32_t mvpUniformLocation;
    video::E_SHADER_CONSTANT_TYPE mvpUniformType;

public:
    DebugRenderCallback() : mvpUniformLocation(-1), mvpUniformType(video::ESCT_FLOAT_VEC3) {}

    virtual void PostLink(video::IMaterialRendererServices *services, const video::E_MATERIAL_TYPE &matType, const core::vector<video::SConstantLocationNamePair>& constants) {
        mvpUniformLocation = constants[0].location;
        mvpUniformType = constants[0].type;
    }

    virtual void OnSetConstants(video::IMaterialRendererServices *services, int32_t userData) {
        services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(), mvpUniformLocation, mvpUniformType, 1);
    }

    virtual void OnUnsetMaterial() {}
};

CDebugRender::CDebugRender(irr::video::IVideoDriver *driver) 
:   m_driver(driver), 
    m_desc(driver->createGPUMeshDataFormatDesc()),
    m_mesh(new scene::IGPUMeshBuffer) {
    DebugRenderCallback *callback = new DebugRenderCallback;
    
   ///m_material.MaterialType = (video::E_MATERIAL_TYPE)
      //  driver->getGPUProgrammingServices()->addHighLevelShaderMaterial(
        //    DebugDraw::Draw3DLineVertexShader, nullptr, nullptr, nullptr, 
        //    DebugDraw::Draw3DLineFragmentShader, 2, video::EMT_SOLID, callback, 0);
   // callback->drop();

   // m_mesh->setPrimitiveType(scene::EPT_LINES);
   // m_mesh->setIndexType(scene::EIT_UNKNOWN);

   // auto buff = driver->getDefaultUpStreamingBuffer()->getBuffer();
    //m_desc->mapVertexAttrBuffer(buff, scene::EVAI_ATTR0, scene::ECPA_THREE, scene::ECT_FLOAT,
     //   sizeof(DebugVertex), offsetof(DebugVertex, DebugVertex::pos[0]));

   // m_desc->mapVertexAttrBuffer(buff, scene::EVAI_ATTR1, scene::ECPA_THREE, scene::ECT_NORMALIZED_UNSIGNED_BYTE,
     //   sizeof(DebugVertex), offsetof(DebugVertex, DebugVertex::pos[1]));

   // m_mesh->setMeshDataAndFormat(m_desc);
   // m_desc->drop();

    
}

CDebugRender::~CDebugRender() {
    //m_lineRender->drop();
}

void CDebugRender::clear() {
    m_scene.clear();
}

void CDebugRender::draw() {

}

void CDebugRender::drawLine(const btVector3 &from, const btVector3 &to, const btVector3 &color) {
   /* m_lineRender->draw(from.x(), from.y(), from.z(),
        to.x(), to.y(), to.z(),
        color.x(), color.y(), color.z(), 1.0f);*/
}

void CDebugRender::drawContactPoint(const btVector3& PointOnB, const btVector3& normalOnB, btScalar distance, int lifeTime, const btVector3& color) {}
void CDebugRender::reportErrorWarning(const char* warningString) {}
void CDebugRender::draw3dText(const btVector3& location, const char* textString) {}
void CDebugRender::setDebugMode(int debugMode) {}
int	CDebugRender::getDebugMode() const { return btIDebugDraw::DBG_DrawWireframe;  }