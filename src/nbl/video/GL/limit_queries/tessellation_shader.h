m_features.tessellationShader = true;
GetIntegerv(GLENUM_WITH_SUFFIX(GL_MAX_TESS_GEN_LEVEL), reinterpret_cast<GLint*>(&m_properties.limits.maxTessellationGenerationLevel));
// GL_MAX_PATCH_VERTICES => maximum patch size
GetIntegerv(GLENUM_WITH_SUFFIX(GL_MAX_PATCH_VERTICES), reinterpret_cast<GLint*>(&m_properties.limits.maxTessellationPatchSize));
// GL_MAX_TESS_CONTROL_INPUT_COMPONENTS => num. components for per-vertex inputs in tess.
GetIntegerv(GLENUM_WITH_SUFFIX(GL_MAX_TESS_CONTROL_INPUT_COMPONENTS), reinterpret_cast<GLint*>(&m_properties.limits.maxTessellationControlPerVertexInputComponents));
// GL_MAX_TESS_CONTROL_OUTPUT_COMPONENTS => num. components for per-vertex outputs in tess. 
GetIntegerv(GLENUM_WITH_SUFFIX(GL_MAX_TESS_CONTROL_OUTPUT_COMPONENTS), reinterpret_cast<GLint*>(&m_properties.limits.maxTessellationControlPerVertexOutputComponents)); 
// GL_MAX_TESS_PATCH_COMPONENTS => num. components for per-patch output varyings
GetIntegerv(GLENUM_WITH_SUFFIX(GL_MAX_TESS_PATCH_COMPONENTS), reinterpret_cast<GLint*>(&m_properties.limits.maxTessellationControlPerPatchOutputComponents));
GetIntegerv(GLENUM_WITH_SUFFIX(GL_MAX_TESS_CONTROL_TOTAL_OUTPUT_COMPONENTS), reinterpret_cast<GLint*>(&m_properties.limits.maxTessellationControlTotalOutputComponents));
GetIntegerv(GLENUM_WITH_SUFFIX(GL_MAX_TESS_EVALUATION_INPUT_COMPONENTS), reinterpret_cast<GLint*>(&m_properties.limits.maxTessellationEvaluationInputComponents));
GetIntegerv(GLENUM_WITH_SUFFIX(GL_MAX_TESS_EVALUATION_OUTPUT_COMPONENTS), reinterpret_cast<GLint*>(&m_properties.limits.maxTessellationEvaluationOutputComponents));