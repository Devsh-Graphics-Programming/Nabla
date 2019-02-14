// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_GPU_PROGRAMMING_SERVICES_H_INCLUDED__
#define __I_GPU_PROGRAMMING_SERVICES_H_INCLUDED__

#include "EMaterialTypes.h"
#include "irr/video/IGPUMeshBuffer.h"
#include "path.h"

namespace irr
{

namespace io
{
	class IReadFile;
} // end namespace io

namespace video
{

class IVideoDriver;
class IShaderConstantSetCallBack;


//! Interface making it possible to create and use programs running on the GPU.
class IRR_FORCE_EBO IGPUProgrammingServices
{
public:

	virtual int32_t addHighLevelShaderMaterial(
		const char* vertexShaderProgram,
		const char* controlShaderProgram,
		const char* evaluationShaderProgram,
		const char* geometryShaderProgram,
		const char* pixelShaderProgram,
		uint32_t patchVertices=3,
		E_MATERIAL_TYPE baseMaterial = video::EMT_SOLID,
		IShaderConstantSetCallBack* callback = 0,
		const char** xformFeedbackOutputs = NULL,
		const uint32_t& xformFeedbackOutputCount = 0,
		int32_t userData = 0,
		const char* vertexShaderEntryPointName="main",
		const char* controlShaderEntryPointName = "main",
		const char* evaluationShaderEntryPointName = "main",
		const char* geometryShaderEntryPointName = "main",
		const char* pixelShaderEntryPointName="main") = 0;

	virtual int32_t addHighLevelShaderMaterialFromFiles(
		const io::path& vertexShaderProgramFileName,
		const io::path& controlShaderProgramFileName,
		const io::path& evaluationShaderProgramFileName,
		const io::path& geometryShaderProgramFileName,
		const io::path& pixelShaderProgramFileName,
		uint32_t patchVertices=3,
		E_MATERIAL_TYPE baseMaterial = video::EMT_SOLID,
		IShaderConstantSetCallBack* callback = 0,
		const char** xformFeedbackOutputs = NULL,
		const uint32_t& xformFeedbackOutputCount = 0,
		int32_t userData = 0,
		const char* vertexShaderEntryPointName="main",
		const char* controlShaderEntryPointName = "main",
		const char* evaluationShaderEntryPointName = "main",
		const char* geometryShaderEntryPointName = "main",
		const char* pixelShaderEntryPointName="main") = 0;

	virtual int32_t addHighLevelShaderMaterialFromFiles(
		io::IReadFile* vertexShaderProgram,
		io::IReadFile* controlShaderProgram,
		io::IReadFile* evaluationShaderProgram,
		io::IReadFile* geometryShaderProgram,
		io::IReadFile* pixelShaderProgram,
		uint32_t patchVertices=3,
		E_MATERIAL_TYPE baseMaterial = video::EMT_SOLID,
		IShaderConstantSetCallBack* callback = 0,
		const char** xformFeedbackOutputs = NULL,
		const uint32_t& xformFeedbackOutputCount = 0,
		int32_t userData = 0,
		const char* vertexShaderEntryPointName="main",
		const char* controlShaderEntryPointName = "main",
		const char* evaluationShaderEntryPointName = "main",
		const char* geometryShaderEntryPointName = "main",
		const char* pixelShaderEntryPointName="main") = 0;


    virtual bool replaceHighLevelShaderMaterial(const int32_t &materialIDToReplace,
        const char* vertexShaderProgram,
        const char* controlShaderProgram,
        const char* evaluationShaderProgram,
        const char* geometryShaderProgram,
        const char* pixelShaderProgram,
        uint32_t patchVertices=3,
        E_MATERIAL_TYPE baseMaterial=video::EMT_SOLID,
        IShaderConstantSetCallBack* callback=0,
		const char** xformFeedbackOutputs = NULL,
		const uint32_t& xformFeedbackOutputCount = 0,
        int32_t userData=0,
        const char* vertexShaderEntryPointName="main",
        const char* controlShaderEntryPointName="main",
        const char* evaluationShaderEntryPointName="main",
        const char* geometryShaderEntryPointName="main",
        const char* pixelShaderEntryPointName="main") = 0;
};


} // end namespace video
} // end namespace irr

#endif

