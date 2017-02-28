#ifndef __C_OPEN_GL_VAO_H_INCLUDED__
#define __C_OPEN_GL_VAO_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "COpenGLBuffer.h"
#include "IMeshBuffer.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_

namespace irr
{
namespace video
{
    static GLint eComponentsPerAttributeToGLint[scene::ECPA_COUNT] =
    {
        GL_BGRA,
        1,2,3,4
    };

    static GLenum eComponentTypeToGLenum[scene::ECT_COUNT] =
    {
        GL_FLOAT,
        GL_HALF_FLOAT,
        GL_DOUBLE,
        GL_UNSIGNED_INT_10F_11F_11F_REV,
        //normalized ints
        GL_INT_2_10_10_10_REV,
        GL_UNSIGNED_INT_2_10_10_10_REV,
        GL_BYTE,
        GL_UNSIGNED_BYTE,
        GL_SHORT,
        GL_UNSIGNED_SHORT,
        GL_INT,
        GL_UNSIGNED_INT,
        //unnorm ints
        GL_INT_2_10_10_10_REV,
        GL_UNSIGNED_INT_2_10_10_10_REV,
        GL_BYTE,
        GL_UNSIGNED_BYTE,
        GL_SHORT,
        GL_UNSIGNED_SHORT,
        GL_INT,
        GL_UNSIGNED_INT,
        //native ints
        GL_INT_2_10_10_10_REV,
        GL_UNSIGNED_INT_2_10_10_10_REV,
        GL_BYTE,
        GL_UNSIGNED_BYTE,
        GL_SHORT,
        GL_UNSIGNED_SHORT,
        GL_INT,
        GL_UNSIGNED_INT,
        //native double
        GL_DOUBLE
    };


    class COpenGLVAO : public scene::IGPUMeshDataFormatDesc
    {
            //vertices
            COpenGLBuffer* mappedAttrBuf[scene::EVAI_COUNT];
            //indices
            COpenGLBuffer* mappedIndexBuf;

            inline bool formatCanBeAppended(const COpenGLVAO* other) const
            {
                bool retVal = true;
                for (size_t i=0; retVal&&i<scene::EVAI_COUNT; i++)
                {
                    if (mappedAttrBuf[i])
                    {
                        if (other->mappedAttrBuf[i])
                            retVal = retVal&&compntsPerAttr[i]==other->compntsPerAttr[i]&&attrType[i]==other->attrType[i];
                        else
                            return false;
                    }
                    else
                    {
                        if (other->mappedAttrBuf[i])
                            return false;
                        else
                            retVal = retVal&&compntsPerAttr[i]==other->compntsPerAttr[i]&&attrType[i]==other->attrType[i];
                    }
                }
                return retVal;
            }


            GLuint vao;
            uint64_t lastValidated;
        public:
            COpenGLVAO();
            virtual ~COpenGLVAO();

            bool formatCanBeAppended(const scene::IGPUMeshDataFormatDesc* other) const
            {
                return formatCanBeAppended(reinterpret_cast<const COpenGLVAO*>(other));
            }


            virtual void mapIndexBuffer(IGPUBuffer* ixbuf);

            virtual const video::IGPUBuffer* getIndexBuffer() const
            {
                return mappedIndexBuf;
            }

            virtual void mapVertexAttrBuffer(IGPUBuffer* attrBuf, const scene::E_VERTEX_ATTRIBUTE_ID& attrId, scene::E_COMPONENTS_PER_ATTRIBUTE components, scene::E_COMPONENT_TYPE type, const size_t &stride=0, size_t offset=0, uint32_t divisor=0);

            virtual void setMappedBufferOffset(const scene::E_VERTEX_ATTRIBUTE_ID& attrId, const size_t &offset);

            virtual const video::IGPUBuffer* getMappedBuffer(const scene::E_VERTEX_ATTRIBUTE_ID& attrId) const
            {
                if (attrId>=scene::EVAI_COUNT)
                    return NULL;

                return mappedAttrBuf[attrId];
            }



            inline const GLuint& getOpenGLName() const {return vao;}
            bool rebindRevalidate();


            void swapVertexAttrBuffer(IGPUBuffer* attrBuf, const scene::E_VERTEX_ATTRIBUTE_ID& attrId, const size_t& newOffset, const size_t& newStride)
            {
                if (!mappedAttrBuf[attrId] || !attrBuf)
                    return;

                attrBuf->grab();
                mappedAttrBuf[attrId]->drop();

                COpenGLBuffer* asGLBuf = dynamic_cast<COpenGLBuffer*>(attrBuf);
                mappedAttrBuf[attrId] = asGLBuf;
                attrOffset[attrId] = newOffset;
                attrStride[attrId] = newStride;

                COpenGLExtensionHandler::extGlVertexArrayVertexBuffer(vao,attrId,asGLBuf->getOpenGLName(),attrOffset[attrId],attrStride[attrId]);
            }
    };


} // end namespace video
} // end namespace irr

#endif
#endif

