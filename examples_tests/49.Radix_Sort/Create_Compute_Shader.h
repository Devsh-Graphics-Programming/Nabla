#ifndef _CREATE_COMPUTE_SHADER_H_INCLUDED__
#define _CREATE_COMPUTE_SHADER_H_INCLUDED__

#include <cstdint>

#include <CReadFile.h>
#include <COpenGLExtensionHandler.h>

namespace Radix_Sort
{
    inline unsigned createComputeShader(const char* _src)
    {
        using namespace nbl;

        unsigned program = video::COpenGLExtensionHandler::extGlCreateProgram();
        unsigned cs = video::COpenGLExtensionHandler::extGlCreateShader(GL_COMPUTE_SHADER);

        video::COpenGLExtensionHandler::extGlShaderSource(cs, 1, const_cast<const char**>(&_src), NULL);
        video::COpenGLExtensionHandler::extGlCompileShader(cs);

        // check for compilation errors
        GLint success;
        GLchar infoLog[0x200];
        video::COpenGLExtensionHandler::extGlGetShaderiv(cs, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            video::COpenGLExtensionHandler::extGlGetShaderInfoLog(cs, sizeof(infoLog), nullptr, infoLog);
            os::Printer::log("CS COMPILATION ERROR:\n", infoLog, ELL_ERROR);
            video::COpenGLExtensionHandler::extGlDeleteShader(cs);
            video::COpenGLExtensionHandler::extGlDeleteProgram(program);
            return 0;
        }

        video::COpenGLExtensionHandler::extGlAttachShader(program, cs);
        video::COpenGLExtensionHandler::extGlLinkProgram(program);

        //check linking errors
        success = 0;
        video::COpenGLExtensionHandler::extGlGetProgramiv(program, GL_LINK_STATUS, &success);
        if (success == GL_FALSE)
        {
            video::COpenGLExtensionHandler::extGlGetProgramInfoLog(program, sizeof(infoLog), nullptr, infoLog);
            os::Printer::log("CS LINK ERROR:\n", infoLog, ELL_ERROR);
            video::COpenGLExtensionHandler::extGlDeleteShader(cs);
            video::COpenGLExtensionHandler::extGlDeleteProgram(program);
            return 0;
        }

        return program;
    }
	
	inline uint32_t Create_Compute_Shader_From_File(const char * const File_Path)
	{
        printf("BUILDING CS %s\n", File_Path);
        using namespace nbl;
        using namespace io;
        IReadFile* file = new CReadFile(File_Path);

        const size_t size = file->getSize();
        char* const src = (char*)malloc(size + 1);
        file->read(src, size);
        src[size] = 0;

        file->drop();
        unsigned cs = createComputeShader(src);
        free(src);

        return cs;
	}
}

#endif /* _CREATE_COMPUTE_SHADER_H_INCLUDED__ */