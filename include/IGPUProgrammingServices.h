// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_GPU_PROGRAMMING_SERVICES_H_INCLUDED__
#define __I_GPU_PROGRAMMING_SERVICES_H_INCLUDED__

#include "EMaterialTypes.h"
#include "EPrimitiveTypes.h"
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
class IGPUProgrammingServices
{
public:

	//! Destructor
	virtual ~IGPUProgrammingServices() {}

	virtual s32 addHighLevelShaderMaterial(
		const c8* vertexShaderProgram,
		const c8* controlShaderProgram,
		const c8* evaluationShaderProgram,
		const c8* geometryShaderProgram,
		const c8* pixelShaderProgram,
		u32 patchVertices=3,
		E_MATERIAL_TYPE baseMaterial = video::EMT_SOLID,
		IShaderConstantSetCallBack* callback = 0,
		const char** xformFeedbackOutputs = NULL,
		const uint32_t& xformFeedbackOutputCount = 0,
		s32 userData = 0,
		const c8* vertexShaderEntryPointName="main",
		const c8* controlShaderEntryPointName = "main",
		const c8* evaluationShaderEntryPointName = "main",
		const c8* geometryShaderEntryPointName = "main",
		const c8* pixelShaderEntryPointName="main") = 0;

	virtual s32 addHighLevelShaderMaterialFromFiles(
		const io::path& vertexShaderProgramFileName,
		const io::path& controlShaderProgramFileName,
		const io::path& evaluationShaderProgramFileName,
		const io::path& geometryShaderProgramFileName,
		const io::path& pixelShaderProgramFileName,
		u32 patchVertices=3,
		E_MATERIAL_TYPE baseMaterial = video::EMT_SOLID,
		IShaderConstantSetCallBack* callback = 0,
		const char** xformFeedbackOutputs = NULL,
		const uint32_t& xformFeedbackOutputCount = 0,
		s32 userData = 0,
		const c8* vertexShaderEntryPointName="main",
		const c8* controlShaderEntryPointName = "main",
		const c8* evaluationShaderEntryPointName = "main",
		const c8* geometryShaderEntryPointName = "main",
		const c8* pixelShaderEntryPointName="main") = 0;

	virtual s32 addHighLevelShaderMaterialFromFiles(
		io::IReadFile* vertexShaderProgram,
		io::IReadFile* controlShaderProgram,
		io::IReadFile* evaluationShaderProgram,
		io::IReadFile* geometryShaderProgram,
		io::IReadFile* pixelShaderProgram,
		u32 patchVertices=3,
		E_MATERIAL_TYPE baseMaterial = video::EMT_SOLID,
		IShaderConstantSetCallBack* callback = 0,
		const char** xformFeedbackOutputs = NULL,
		const uint32_t& xformFeedbackOutputCount = 0,
		s32 userData = 0,
		const c8* vertexShaderEntryPointName="main",
		const c8* controlShaderEntryPointName = "main",
		const c8* evaluationShaderEntryPointName = "main",
		const c8* geometryShaderEntryPointName = "main",
		const c8* pixelShaderEntryPointName="main") = 0;


    virtual bool replaceHighLevelShaderMaterial(const s32 &materialIDToReplace,
        const c8* vertexShaderProgram,
        const c8* controlShaderProgram,
        const c8* evaluationShaderProgram,
        const c8* geometryShaderProgram,
        const c8* pixelShaderProgram,
        u32 patchVertices=3,
        E_MATERIAL_TYPE baseMaterial=video::EMT_SOLID,
        IShaderConstantSetCallBack* callback=0,
		const char** xformFeedbackOutputs = NULL,
		const uint32_t& xformFeedbackOutputCount = 0,
        s32 userData=0,
        const c8* vertexShaderEntryPointName="main",
        const c8* controlShaderEntryPointName="main",
        const c8* evaluationShaderEntryPointName="main",
        const c8* geometryShaderEntryPointName="main",
        const c8* pixelShaderEntryPointName="main") = 0;
};


} // end namespace video
} // end namespace irr

#endif

