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
		s32 userData = 0,
		const c8* vertexShaderEntryPointName="main",
		const c8* controlShaderEntryPointName = "main",
		const c8* evaluationShaderEntryPointName = "main",
		const c8* geometryShaderEntryPointName = "main",
		const c8* pixelShaderEntryPointName="main") = 0;

	//! Adds a new high-level shading material renderer to the VideoDriver.
	/** Currently only HLSL/D3D9 and GLSL/OpenGL are supported.
	\param vertexShaderProgram String containing the source of the vertex
	shader program. This can be 0 if no vertex program shall be used.
	\param vertexShaderEntryPointName Name of the entry function of the
	vertexShaderProgram (p.e. "main")
	\param vsCompileTarget Vertex shader version the high level shader
	shall be compiled to.
	\param pixelShaderProgram String containing the source of the pixel
	shader program. This can be 0 if no pixel shader shall be used.
	\param pixelShaderEntryPointName Entry name of the function of the
	pixelShaderProgram (p.e. "main")
	\param psCompileTarget Pixel shader version the high level shader
	shall be compiled to.
	\param geometryShaderProgram String containing the source of the
	geometry shader program. This can be 0 if no geometry shader shall be
	used.
	\param geometryShaderEntryPointName Entry name of the function of the
	geometryShaderProgram (p.e. "main")
	\param gsCompileTarget Geometry shader version the high level shader
	shall be compiled to.
	\param inType Type of vertices passed to geometry shader
	\param outType Type of vertices created by geometry shader
	\param verticesOut Maximal number of vertices created by geometry
	shader. If 0, maximal number supported is assumed.
	\param callback Pointer to an implementation of
	IShaderConstantSetCallBack in which you can set the needed vertex,
	pixel, and geometry shader program constants. Set this to 0 if you
	don't need this.
	\param baseMaterial Base material which renderstates will be used to
	shade the material.
	\param userData a user data int. This int can be set to any value and
	will be set as parameter in the callback method when calling
	OnSetConstants(). In this way it is easily possible to use the same
	callback method for multiple materials and distinguish between them
	during the call.
	\param shaderLang a type of shading language used in current shader.
	\return Number of the material type which can be set in
	SMaterial::MaterialType to use the renderer. -1 is returned if an error
	occured, e.g. if a shader program could not be compiled or a compile
	target is not reachable. The error strings are then printed to the
	error log and can be catched with a custom event receiver. */
	virtual s32 addHighLevelShaderMaterial(
		const c8* vertexShaderProgram,
		const c8* vertexShaderEntryPointName,
		const c8* pixelShaderProgram,
		const c8* pixelShaderEntryPointName,
		const c8* geometryShaderProgram = NULL,
		const c8* geometryShaderEntryPointName = "main",
		IShaderConstantSetCallBack* callback = 0,
		E_MATERIAL_TYPE baseMaterial = video::EMT_SOLID,
		s32 userData = 0) = 0;

	//! Like IGPUProgrammingServices::addShaderMaterial(), but loads from files.
	/** \param vertexShaderProgramFileName Text file containing the source
	of the vertex shader program. Set to empty string if no vertex shader
	shall be created.
	\param vertexShaderEntryPointName Name of the entry function of the
	vertexShaderProgram  (p.e. "main")
	\param vsCompileTarget Vertex shader version the high level shader
	shall be compiled to.
	\param pixelShaderProgramFileName Text file containing the source of
	the pixel shader program. Set to empty string if no pixel shader shall
	be created.
	\param pixelShaderEntryPointName Entry name of the function of the
	pixelShaderProgram (p.e. "main")
	\param psCompileTarget Pixel shader version the high level shader
	shall be compiled to.
	\param geometryShaderProgramFileName Name of the source of
	the geometry shader program. Set to empty string if no geometry shader
	shall be created.
	\param geometryShaderEntryPointName Entry name of the function of the
	geometryShaderProgram (p.e. "main")
	\param gsCompileTarget Geometry shader version the high level shader
	shall be compiled to.
	\param inType Type of vertices passed to geometry shader
	\param outType Type of vertices created by geometry shader
	\param verticesOut Maximal number of vertices created by geometry
	shader. If 0, maximal number supported is assumed.
	\param callback Pointer to an implementation of
	IShaderConstantSetCallBack in which you can set the needed vertex,
	pixel, and geometry shader program constants. Set this to 0 if you
	don't need this.
	\param baseMaterial Base material which renderstates will be used to
	shade the material.
	\param userData a user data int. This int can be set to any value and
	will be set as parameter in the callback method when calling
	OnSetConstants(). In this way it is easily possible to use the same
	callback method for multiple materials and distinguish between them
	during the call.
	\param shaderLang a type of shading language used in current shader.
	\return Number of the material type which can be set in
	SMaterial::MaterialType to use the renderer. -1 is returned if an error
	occured, e.g. if a shader program could not be compiled or a compile
	target is not reachable. The error strings are then printed to the
	error log and can be catched with a custom event receiver. */
	virtual s32 addHighLevelShaderMaterialFromFiles(
		const io::path& vertexShaderProgramFileName,
		const c8* vertexShaderEntryPointName,
		const io::path& pixelShaderProgramFileName,
		const c8* pixelShaderEntryPointName,
		const io::path& geometryShaderProgramFileName = "",
		const c8* geometryShaderEntryPointName = "main",
		IShaderConstantSetCallBack* callback = 0,
		E_MATERIAL_TYPE baseMaterial = video::EMT_SOLID,
		s32 userData = 0) = 0;

	//! Like IGPUProgrammingServices::addShaderMaterial(), but loads from files.
	/** \param vertexShaderProgram Text file handle containing the source
	of the vertex shader program. Set to 0 if no vertex shader shall be
	created.
	\param vertexShaderEntryPointName Name of the entry function of the
	vertexShaderProgram
	\param vsCompileTarget Vertex shader version the high level shader
	shall be compiled to.
	\param pixelShaderProgram Text file handle containing the source of
	the pixel shader program. Set to 0 if no pixel shader shall be created.
	\param pixelShaderEntryPointName Entry name of the function of the
	pixelShaderProgram (p.e. "main")
	\param psCompileTarget Pixel shader version the high level shader
	shall be compiled to.
	\param geometryShaderProgram Text file handle containing the source of
	the geometry shader program. Set to 0 if no geometry shader shall be
	created.
	\param geometryShaderEntryPointName Entry name of the function of the
	geometryShaderProgram (p.e. "main")
	\param gsCompileTarget Geometry shader version the high level shader
	shall be compiled to.
	\param inType Type of vertices passed to geometry shader
	\param outType Type of vertices created by geometry shader
	\param verticesOut Maximal number of vertices created by geometry
	shader. If 0, maximal number supported is assumed.
	\param callback Pointer to an implementation of
	IShaderConstantSetCallBack in which you can set the needed vertex and
	pixel shader program constants. Set this to 0 if you don't need this.
	\param baseMaterial Base material which renderstates will be used to
	shade the material.
	\param userData a user data int. This int can be set to any value and
	will be set as parameter in the callback method when calling
	OnSetConstants(). In this way it is easily possible to use the same
	callback method for multiple materials and distinguish between them
	during the call.
	\param shaderLang a type of shading language used in current shader.
	\return Number of the material type which can be set in
	SMaterial::MaterialType to use the renderer. -1 is returned if an
	error occured, e.g. if a shader program could not be compiled or a
	compile target is not reachable. The error strings are then printed to
	the error log and can be catched with a custom event receiver. */
	virtual s32 addHighLevelShaderMaterialFromFiles(
		io::IReadFile* vertexShaderProgram,
		const c8* vertexShaderEntryPointName,
		io::IReadFile* pixelShaderProgram,
		const c8* pixelShaderEntryPointName,
		io::IReadFile* geometryShaderProgram = NULL,
		const c8* geometryShaderEntryPointName = "main",
		IShaderConstantSetCallBack* callback = 0,
		E_MATERIAL_TYPE baseMaterial = video::EMT_SOLID,
		s32 userData = 0) = 0;


    virtual bool replaceHighLevelShaderMaterial(const s32 &materialIDToReplace,
        const c8* vertexShaderProgram,
        const c8* controlShaderProgram,
        const c8* evaluationShaderProgram,
        const c8* geometryShaderProgram,
        const c8* pixelShaderProgram,
        u32 patchVertices=3,
        E_MATERIAL_TYPE baseMaterial=video::EMT_SOLID,
        IShaderConstantSetCallBack* callback=0,
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

