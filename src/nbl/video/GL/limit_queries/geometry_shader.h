
m_features.geometryShader = true;
GetIntegerv(GLENUM_WITH_SUFFIX(GL_MAX_GEOMETRY_SHADER_INVOCATIONS), reinterpret_cast<GLint*>(&m_properties.limits.maxGeometryShaderInvocations));
GetIntegerv(GLENUM_WITH_SUFFIX(GL_MAX_GEOMETRY_INPUT_COMPONENTS), reinterpret_cast<GLint*>(&m_properties.limits.maxGeometryInputComponents));
GetIntegerv(GLENUM_WITH_SUFFIX(GL_MAX_GEOMETRY_OUTPUT_COMPONENTS), reinterpret_cast<GLint*>(&m_properties.limits.maxGeometryOutputComponents));
GetIntegerv(GLENUM_WITH_SUFFIX(GL_MAX_GEOMETRY_OUTPUT_VERTICES), reinterpret_cast<GLint*>(&m_properties.limits.maxGeometryOutputVertices));
GetIntegerv(GLENUM_WITH_SUFFIX(GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS), reinterpret_cast<GLint*>(&m_properties.limits.maxGeometryTotalOutputComponents));