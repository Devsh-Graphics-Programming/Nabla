#ifndef __IRR_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__
#define __IRR_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__

#include "irr/asset/ICPUSpecializedShader.h"
#include "irr/video/IGPUSpecializedShader.h"
#include "COpenGLExtensionHandler.h"

namespace irr { namespace video
{

class COpenGLSpecializedShader : public IGPUSpecializedShader
{
public:
    COpenGLSpecializedShader(video::IVideoDriver* _driver, const asset::ICPUSpecializedShader* _cpushader);


    GLuint getOpenGLName() const { return m_GLname; }

private:
    GLuint m_GLname;
};

}}


#endif//__IRR_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__
