// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_C_OPENGL_QUERY_H_INCLUDED__
#define __NBL_C_OPENGL_QUERY_H_INCLUDED__

#include "IQueryObject.h"

#ifdef _NBL_COMPILE_WITH_OPENGL_
#include "COpenGLExtensionHandler.h"

namespace nbl
{
namespace video
{

class COpenGLQuery : public IQueryObject
{
    protected:
        virtual ~COpenGLQuery();

    public:
        COpenGLQuery(const GLenum& type_in);

		virtual E_QUERY_OBJECT_TYPE getQueryObjectType() const {return cachedIrrType;}

		virtual void getQueryResult(uint32_t* queryResult);
		virtual void getQueryResult(uint64_t* queryResult);
		virtual bool getQueryResult32(IGPUBuffer* buffer, const size_t& offset=0, const bool& conditionalWrite=true);
		virtual bool getQueryResult64(IGPUBuffer* buffer, const size_t& offset=0, const bool& conditionalWrite=true);

		virtual bool isQueryReady();
		virtual void isQueryReady32(IGPUBuffer* buffer, const size_t& offset=0);
		virtual void isQueryReady64(IGPUBuffer* buffer, const size_t& offset=0);

		inline GLenum getType() const {return type;}
		inline GLuint getGLHandle() const {return object;}

        inline void flagBegun() {active=true; queryIsReady=false;}
        inline void flagEnded() {active=false; queryNeedsUpdate=true;}
        inline bool isActive() const {return active;}

    private:
        GLuint object;
        GLenum type;
        E_QUERY_OBJECT_TYPE cachedIrrType;

        void updateQueryResult();

        bool active;
        bool queryNeedsUpdate;
        bool queryIsReady;
        GLuint cachedCounter32;
        GLuint64 cachedCounter64;
};


}
}
#endif // _NBL_COMPILE_WITH_OPENGL_

#endif

