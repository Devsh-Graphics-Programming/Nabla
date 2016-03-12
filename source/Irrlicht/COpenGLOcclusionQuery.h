#ifndef COPENGLOCCLUSIONQUERY_H
#define COPENGLOCCLUSIONQUERY_H

#include <IOcclusionQuery.h>
#include "COpenGLDriver.h"

namespace irr
{
namespace video
{

class COpenGLOcclusionQuery : public IOcclusionQuery
{
    public:
        COpenGLOcclusionQuery(COpenGLDriver* driver_in, GLenum type_in);
        virtual ~COpenGLOcclusionQuery();
		virtual u32 getOcclusionQueryResult() const;
		virtual GLenum getType() const;
		virtual GLuint getGLHandle() const;
		virtual GLint* getCounterPointer() {return &counter;}
		virtual void setCondWaitMode(const E_CONDITIONAL_RENDERING_WAIT_MODE& mode);
		virtual GLenum getCondWaitModeGL() const {return condModeGL;}
    protected:
    private:
        GLuint object;
        GLenum type,condModeGL;
        GLint counter;
        COpenGLDriver* driver;
};


}
}

#endif // COPENGLOCCLUSIONQUERY_H
