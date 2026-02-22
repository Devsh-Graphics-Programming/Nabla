<?xml version="1.0" encoding="utf-8"?>
<Project ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003" DefaultTargets="Build">
  <PropertyGroup>
    <Configuration Condition=" '$(Configuration)' == '' ">Debug</Configuration>
    <SchemaVersion>2.0</SchemaVersion>
    <ProjectGuid>{@_NBL_PF_PROJECT_GUID_@}</ProjectGuid>
    <ProjectHome>@_NBL_PF_PROJECT_HOME_@</ProjectHome>
    <StartupFile>@_NBL_PF_STARTUP_FILE_@</StartupFile>
    <SearchPath>@_NBL_PF_SEARCH_PATH_@</SearchPath>
    <WorkingDirectory>@_NBL_PF_WORKING_DIRECTORY_@</WorkingDirectory>
    <OutputPath>.</OutputPath>
    <ProjectTypeGuids>{@_NBL_PF_PROJECT_TYPE_GUIDS_@}</ProjectTypeGuids>
    <LaunchProvider>Standard Python launcher</LaunchProvider>
    <InterpreterId />
    <InterpreterArguments>-m @_NBL_PF_MODULE_TRAVERSAL_@</InterpreterArguments>
    <EnableNativeCodeDebugging>False</EnableNativeCodeDebugging>
    <IsWindowsApplication>False</IsWindowsApplication>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)' == 'Debug'" />
  <PropertyGroup Condition="'$(Configuration)' == 'Release'" />
  <PropertyGroup>
    <VisualStudioVersion Condition=" '$(VisualStudioVersion)' == '' ">10.0</VisualStudioVersion>
  </PropertyGroup>
  <ItemGroup>
    @_NBL_PF_IG_COMPILE_INCLUDE_@
  </ItemGroup>
  <ItemGroup>
    @_NBL_PF_IG_FOLDER_INCLUDE_@
  </ItemGroup>
  <Import Project="$(MSBuildExtensionsPath32)\Microsoft\VisualStudio\v$(VisualStudioVersion)\Python Tools\Microsoft.PythonTools.targets" />
</Project>