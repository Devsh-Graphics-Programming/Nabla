#ifndef __IRR_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__
#define __IRR_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__

#include "irr/asset/ICPUSpecializedShader.h"
#include "irr/video/IGPUSpecializedShader.h"
#include "COpenGLExtensionHandler.h"
#include "irr/video/COpenGLShader.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_

namespace irr
{
namespace video
{

class COpenGLSpecializedShader : public IGPUSpecializedShader
{
public:
    COpenGLSpecializedShader(const video::IVideoDriver* _driver, const asset::ICPUBuffer* _spirv, const asset::ISpecializationInfo* _specInfo);


    GLuint getOpenGLName() const { return m_GLname; }
    GLenum getStage() const { return m_stage; }

private:
    GLuint m_GLname;
    GLenum m_stage;
};

}
}
#endif

#endif//__IRR_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__
