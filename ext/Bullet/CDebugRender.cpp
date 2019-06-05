#include "CDebugRender.h"
#include "BulletUtility.h"
#include "../../ext/DebugDraw/Draw3DLineShaders.h"


#include <iostream>

using namespace irr;
using namespace ext;
using namespace Bullet3;



CDebugRender::CDebugRender(irr::video::IVideoDriver *driver) 
:   m_driver(driver),
    m_drawMode(btIDebugDraw::DBG_DrawWireframe)
{
    m_lineRender = ext::DebugDraw::CDraw3DLine::create(driver);
}

CDebugRender::~CDebugRender() {
    m_lineRender->drop();
}

void CDebugRender::clearLines() {
    m_scene.clear();
}

void CDebugRender::draw() {
    m_lineRender->draw(m_scene);
}

void CDebugRender::drawLine(const btVector3 &from, const btVector3 &to, const btVector3 &color) {
    DebugDraw::S3DLineVertex toV{ to.x(), to.y(), to.z(), color.x(), color.y(), color.z(), 1.0f };
    DebugDraw::S3DLineVertex fromV{ from.x(), from.y(), from.z(), color.x(), color.y(), color.z(), 1.0f };
    
    m_scene.push_back({ toV, fromV });
}

void CDebugRender::drawContactPoint(const btVector3& PointOnB, const btVector3& normalOnB, btScalar distance, int lifeTime, const btVector3& color) {}
void CDebugRender::reportErrorWarning(const char* warningString) {}
void CDebugRender::draw3dText(const btVector3& location, const char* textString) {}
void CDebugRender::setDebugMode(int debugMode) { m_drawMode = debugMode; }
int	CDebugRender::getDebugMode() const { return m_drawMode;  }