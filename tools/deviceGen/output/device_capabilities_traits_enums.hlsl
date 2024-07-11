// Limits Enums
// VK 1.0
// VK 1.1
static nbl::hlsl::ShaderStage subgroupOpsShaderStages() { return (nbl::hlsl::ShaderStage)subgroupOpsShaderStagesBitPattern; }

static nbl::hlsl::PointClippingBehavior pointClippingBehavior() { return (nbl::hlsl::PointClippingBehavior)pointClippingBehaviorBitPattern; }

// VK 1.2
static nbl::hlsl::ResolveModeFlags supportedDepthResolveModes() { return (nbl::hlsl::ResolveModeFlags)supportedDepthResolveModesBitPattern; }
static nbl::hlsl::ResolveModeFlags supportedStencilResolveModes() { return (nbl::hlsl::ResolveModeFlags)supportedStencilResolveModesBitPattern; }

// VK 1.3
static nbl::hlsl::ShaderStage requiredSubgroupSizeStages() { return (nbl::hlsl::ShaderStage)requiredSubgroupSizeStagesBitPattern; }

// Nabla Core Extensions
// Extensions
static nbl::hlsl::SampleCountFlags sampleLocationSampleCounts() { return (nbl::hlsl::SampleCountFlags)sampleLocationSampleCountsBitPattern; }

static nbl::hlsl::ShaderStage cooperativeMatrixSupportedStages() { return (nbl::hlsl::ShaderStage)cooperativeMatrixSupportedStagesBitPattern; }

// Nabla
static nbl::hlsl::SpirvVersion spirvVersion() { return (nbl::hlsl::SpirvVersion)spirvVersionBitPattern; }

// Features Enums
// VK 1.0
// VK 1.1
// VK 1.2
// VK 1.3
// Nabla Core Extensions
// Extensions
static nbl::hlsl::SwapchainMode swapchainMode() { return (nbl::hlsl::SwapchainMode)swapchainModeBitPattern; }

// Nabla
