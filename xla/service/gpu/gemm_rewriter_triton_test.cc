/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/gemm_rewriter_triton.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/cublas_padding_requirements.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/verified_hlo_module.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::FieldsAre;

namespace m = ::xla::match;

class GemmRewriterTritonTest : public HloTestBase {
 public:
  GemmRewriterTritonTest()
      : HloTestBase(/*verifier_layout_sensitive=*/true,
                    /*allow_mixed_precision_in_hlo_verifier=*/false) {}

  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_triton_gemm_any(false);
    return debug_options;
  }

  se::GpuComputeCapability gpu_version_{
      se::CudaComputeCapability{se::CudaComputeCapability::AMPERE, 0}};

  void MatchHloModule(HloModule& module, absl::string_view pattern) {
    TF_ASSERT_OK_AND_ASSIGN(bool filecheck_result,
                            RunFileCheck(module.ToString(), pattern));
    EXPECT_TRUE(filecheck_result);
  }
};

TEST_F(GemmRewriterTritonTest, TransposeSubdimensionGroup) {
  // This HLO is artificial because unnecessary reshapes get optimized
  // out during compilation. It tests the ability of GemmRewriterTriton
  // to handle transposes of groups of subdimensions.
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m

ENTRY e {
  p0 = f32[32,3] parameter(0)
  t1 = f32[3,32] transpose(p0), dimensions={1,0}
  r1 = f32[3,8,4] reshape(t1)
  r0 = f32[3,32] reshape(r1)
  p1 = f16[32,7] parameter(1)
  c1 = f32[32,7] convert(p1)
  ROOT d = f32[3,7] dot(r0, c1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})")
                    .value();
  EXPECT_TRUE(GemmRewriterTriton(gpu_version_).Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter(), m::Parameter())));
}

TEST_F(GemmRewriterTritonTest, UnsupportedTransposeIsNotFused) {
  auto module = ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = f16[1,512,8,1024]{3,1,0,2} parameter(0)
  c = f16[1,512,8,1024]{3,2,1,0} copy(p0)
  b = f16[4096,1024]{1,0} bitcast(c)
  p1 = f16[128,1024]{1,0} parameter(1)
  ROOT d = f16[4096,128]{1,0} dot(b, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
})")
                    .value();
  EXPECT_FALSE(GemmRewriterTriton(gpu_version_).Run(module.get()).value());
}

TEST_F(GemmRewriterTritonTest, BitcastChain) {
  // This HLO is artificial because unnecessary reshapes get optimized
  // out during compilation. It tests the ability of GemmRewriterTriton
  // to handle various kinds of bitcasts.
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m

ENTRY e {
  p0 = s8[60,5] parameter(0)
  r0 = s8[3,20,5] reshape(p0)
  c0 = f16[3,20,5] convert(r0)
  p1 = f16[3,200] parameter(1)
  r12 = f16[600] reshape(p1)
  r11 = f16[30,20] reshape(r12)
  r1 = f16[3,10,20] reshape(r11)
  ROOT d = f16[3,5,10] dot(c0, r1),
    lhs_contracting_dims={1}, rhs_contracting_dims={2},
    lhs_batch_dims={0}, rhs_batch_dims={0}
})")
                    .value();
  EXPECT_TRUE(GemmRewriterTriton(gpu_version_).Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter(), m::Parameter())));
}

TEST_F(GemmRewriterTritonTest, SplitDimensionTwice) {
  auto module = ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = s8[4,2,32,4,2] parameter(0)
  r1 = s8[8,32,8] reshape(p0)
  t1 = s8[32,8,8] transpose(r1), dimensions={1,0,2}
  r0 = s8[32,64] reshape(t1)
  p1 = s8[32,32] parameter(1)
  c0 = f16[32,32] convert(p1)
  ROOT d = f16[64,32] dot(r0, c0),
    lhs_contracting_dims={0}, rhs_contracting_dims={1}
})")
                    .value();
  EXPECT_TRUE(GemmRewriterTriton(gpu_version_).Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter(), m::Parameter())));
}

TEST_F(GemmRewriterTritonTest, DoNotTriggerOnUnsupportedOutputConversions) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = f16[128,256] parameter(0)
  p1 = f16[256,512] parameter(1)
  r = f16[128,512] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT c = u8[128,512] convert(r)
})"));
  EXPECT_FALSE(GemmRewriterTriton(gpu_version_).Run(module.get()).value());
}

TEST_F(GemmRewriterTritonTest, DoNotTriggerWhenTheLhsNoncontractingDimIs1) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = s8[1,256] parameter(0)
  p0c = f16[1,256] convert(p0)
  p1 = f16[256,512] parameter(1)
  ROOT r = f16[1,512] dot(p0c, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));
  EXPECT_FALSE(GemmRewriterTriton(gpu_version_).Run(module.get()).value());
}

TEST_F(GemmRewriterTritonTest, DoNotTriggerWhenTheRhsNoncontractingDimIs1) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = s8[128,256] parameter(0)
  p0c = f16[128,256] convert(p0)
  p1 = f16[256,1] parameter(1)
  ROOT r = f16[128,1] dot(p0c, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));
  EXPECT_FALSE(GemmRewriterTriton(gpu_version_).Run(module.get()).value());
}

using TritonDotAnalysisTest = HloTestBase;

TEST_F(TritonDotAnalysisTest, NopBitcasts) {
  const std::string hlo_text = R"(
HloModule t

triton_dot {
  param_0.1 = s8[48,4]{1,0} parameter(0)
  bitcast.18 = s8[1,48,4]{2,1,0} bitcast(param_0.1)
  bitcast.19 = s8[48,4]{1,0} bitcast(bitcast.18)
  convert.4 = bf16[48,4]{1,0} convert(bitcast.19)
  param_1.1 = bf16[4,3]{1,0} parameter(1)
  ROOT dot = bf16[48,3]{1,0} dot(convert.4, param_1.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = s8[48,4]{1,0} parameter(0)
  p1 = bf16[4,3]{1,0} parameter(1)
  custom-call = bf16[48,3]{1,0} custom-call(p0, p1),
    custom_call_target="__triton",
    called_computations={triton_dot}
  ROOT bitcast.2 = bf16[1,8,6,3]{3,2,1,0} bitcast(custom-call)
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  const HloComputation* dot_computation = module->entry_computation()
                                              ->root_instruction()
                                              ->operand(0)
                                              ->called_computations()[0];
  const HloInstruction* p0 = dot_computation->parameter_instruction(0);
  const HloInstruction* p1 = dot_computation->parameter_instruction(1);
  TF_ASSERT_OK_AND_ASSIGN(const auto analysis,
                          TritonFusionAnalysis::Execute(*dot_computation));
  EXPECT_EQ(*analysis.ScopeParameters(TritonFusionAnalysis::Scope::LHS).begin(),
            p0);
  EXPECT_EQ(*analysis.ScopeParameters(TritonFusionAnalysis::Scope::RHS).begin(),
            p1);
  EXPECT_THAT(
      *analysis.IterSpec(TritonFusionAnalysis::Scope::LHS, p0, 0),
      ElementsAre(FieldsAre(/*stride=*/4, /*count=*/48, /*slice_start=*/0,
                            /*slice_limit=*/48, ElementsAre(48))));
  EXPECT_THAT(
      *analysis.IterSpec(TritonFusionAnalysis::Scope::LHS, p0, 1),
      ElementsAre(FieldsAre(/*stride=*/1, /*count=*/4, /*slice_start=*/0,
                            /*slice_limit=*/4, ElementsAre(4))));
  EXPECT_THAT(
      *analysis.IterSpec(TritonFusionAnalysis::Scope::RHS, p1, 0),
      ElementsAre(FieldsAre(/*stride=*/3, /*count=*/4, /*slice_start=*/0,
                            /*slice_limit=*/4, ElementsAre(4))));
  EXPECT_THAT(
      *analysis.IterSpec(TritonFusionAnalysis::Scope::RHS, p1, 1),
      ElementsAre(FieldsAre(/*stride=*/1, /*count=*/3, /*slice_start=*/0,
                            /*slice_limit=*/3, ElementsAre(3))));
}

TEST_F(TritonDotAnalysisTest, Merge) {
  const std::string hlo_text = R"(
HloModule t

triton_dot {
  param_0.1 = s8[1,8,6,4]{3,2,1,0} parameter(0)
  bitcast.18 = s8[48,4]{1,0} bitcast(param_0.1)
  convert.4 = bf16[48,4]{1,0} convert(bitcast.18)
  param_1.1 = bf16[4,3]{1,0} parameter(1)
  ROOT dot = bf16[48,3]{1,0} dot(convert.4, param_1.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = s8[1,8,6,4]{3,2,1,0} parameter(0)
  p1 = bf16[4,3]{1,0} parameter(1)
  custom-call = bf16[48,3]{1,0} custom-call(p0, p1),
    custom_call_target="__triton",
    called_computations={triton_dot}
  ROOT bitcast.2 = bf16[1,8,6,3]{3,2,1,0} bitcast(custom-call)
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  const HloComputation* dot_computation = module->entry_computation()
                                              ->root_instruction()
                                              ->operand(0)
                                              ->called_computations()[0];
  const HloInstruction* p0 = dot_computation->parameter_instruction(0);
  const HloInstruction* p1 = dot_computation->parameter_instruction(1);
  TF_ASSERT_OK_AND_ASSIGN(const auto analysis,
                          TritonFusionAnalysis::Execute(*dot_computation));
  EXPECT_EQ(*analysis.ScopeParameters(TritonFusionAnalysis::Scope::LHS).begin(),
            p0);
  EXPECT_EQ(*analysis.ScopeParameters(TritonFusionAnalysis::Scope::RHS).begin(),
            p1);
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::LHS, p0, 0),
              ElementsAre(FieldsAre(/*stride=*/4, /*count=*/6 * 8,
                                    /*slice_start=*/0, /*slice_limit=*/6 * 8,
                                    /*subfragments=*/ElementsAre(6, 8))));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::LHS, p0, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/4,
                                    /*slice_start=*/0, /*slice_limit=*/4,
                                    /*subfragments=*/ElementsAre(4))));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::RHS, p1, 0),
              ElementsAre(FieldsAre(/*stride=*/3, /*count=*/4,
                                    /*slice_start=*/0, /*slice_limit=*/4,
                                    /*subfragments=*/ElementsAre(4))));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::RHS, p1, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/3,
                                    /*slice_start=*/0, /*slice_limit=*/3,
                                    /*subfragments=*/ElementsAre(3))));
}

TEST_F(TritonDotAnalysisTest, Split) {
  const std::string hlo_text = R"(
HloModule t

triton_dot {
  %parameter_1 = f32[24000,2]{1,0} parameter(1)
  %convert.15 = f16[24000,2]{1,0} convert(%parameter_1)
  %parameter_0 = f16[4]{0} parameter(0)
  %bitcast.45 = f16[2,2]{1,0} bitcast(%parameter_0)
  ROOT %dot.26 = f16[24000,2]{1,0} dot(%convert.15, %bitcast.45),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f16[4]{0} parameter(0)
  p1 = f32[24000,2]{1,0} parameter(1)
  ROOT r = f16[24000,2]{1,0} custom-call(p0, p1),
    custom_call_target="__triton",
    called_computations={triton_dot}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  const HloComputation* dot_computation =
      module->entry_computation()->root_instruction()->called_computations()[0];
  const HloInstruction* p0 = dot_computation->parameter_instruction(0);
  const HloInstruction* p1 = dot_computation->parameter_instruction(1);
  TF_ASSERT_OK_AND_ASSIGN(const auto analysis,
                          TritonFusionAnalysis::Execute(*dot_computation));
  EXPECT_EQ(*analysis.ScopeParameters(TritonFusionAnalysis::Scope::LHS).begin(),
            p1);
  EXPECT_EQ(*analysis.ScopeParameters(TritonFusionAnalysis::Scope::RHS).begin(),
            p0);
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::LHS, p1, 0),
              ElementsAre(FieldsAre(/*stride=*/2, /*count=*/24000,
                                    /*slice_start=*/0, /*slice_limit=*/24000,
                                    /*subfragments=*/ElementsAre(24000))));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::LHS, p1, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/2,
                                    /*slice_start=*/0, /*slice_limit=*/2,
                                    /*subfragments=*/ElementsAre(2))));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::RHS, p0, 0),
              ElementsAre(FieldsAre(/*stride=*/2, /*count=*/2,
                                    /*slice_start=*/0, /*slice_limit=*/2,
                                    /*subfragments=*/ElementsAre(2))));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::RHS, p0, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/2,
                                    /*slice_start=*/0, /*slice_limit=*/2,
                                    /*subfragments=*/ElementsAre(2))));
}

TEST_F(TritonDotAnalysisTest, TransposeMerge) {
  const std::string hlo_text = R"(
HloModule t

triton_dot {
  param_0.1 = s8[1,4,8,6]{3,2,1,0} parameter(0)
  transpose.3 = s8[1,8,6,4]{3,2,1,0} transpose(param_0.1), dimensions={0,2,3,1}
  bitcast.18 = s8[48,4]{1,0} bitcast(transpose.3)
  convert.4 = bf16[48,4]{1,0} convert(bitcast.18)
  param_1.1 = bf16[4,3]{1,0} parameter(1)
  ROOT dot = bf16[48,3]{1,0} dot(convert.4, param_1.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = s8[1,4,8,6]{3,2,1,0} parameter(0)
  p1 = bf16[4,3]{1,0} parameter(1)
  custom-call = bf16[48,3]{1,0} custom-call(p0, p1),
    custom_call_target="__triton",
    called_computations={triton_dot}
  ROOT bitcast.2 = bf16[1,8,6,3]{3,2,1,0} bitcast(custom-call)
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  const HloComputation* dot_computation = module->entry_computation()
                                              ->root_instruction()
                                              ->operand(0)
                                              ->called_computations()[0];
  const HloInstruction* p0 = dot_computation->parameter_instruction(0);
  const HloInstruction* p1 = dot_computation->parameter_instruction(1);
  TF_ASSERT_OK_AND_ASSIGN(const auto analysis,
                          TritonFusionAnalysis::Execute(*dot_computation));
  EXPECT_EQ(*analysis.ScopeParameters(TritonFusionAnalysis::Scope::LHS).begin(),
            p0);
  EXPECT_EQ(*analysis.ScopeParameters(TritonFusionAnalysis::Scope::RHS).begin(),
            p1);
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::LHS, p0, 0),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/8 * 6,
                                    /*slice_start=*/0, /*slice_limit=*/8 * 6,
                                    /*subfragments=*/ElementsAre(6, 8))));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::LHS, p0, 1),
              ElementsAre(FieldsAre(/*stride=*/8 * 6, /*count=*/4,
                                    /*slice_start=*/0, /*slice_limit=*/4,
                                    /*subfragments=*/ElementsAre(4))));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::RHS, p1, 0),
              ElementsAre(FieldsAre(/*stride=*/3, /*count=*/4,
                                    /*slice_start=*/0, /*slice_limit=*/4,
                                    /*subfragments=*/ElementsAre(4))));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::RHS, p1, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/3,
                                    /*slice_start=*/0, /*slice_limit=*/3,
                                    /*subfragments=*/ElementsAre(3))));
}

TEST_F(TritonDotAnalysisTest, CopyMerge) {
  const std::string hlo_text = R"(
HloModule t

triton_dot {
  param_0.1 = s8[1,4,8,6]{3,2,1,0} parameter(0)
  bitcast.99 = s8[1,8,6,4]{2,1,3,0} bitcast(param_0.1)
  copy.3 = s8[1,8,6,4]{3,2,1,0} copy(bitcast.99)
  bitcast.18 = s8[48,4]{1,0} bitcast(copy.3)
  convert.4 = bf16[48,4]{1,0} convert(bitcast.18)
  param_1.1 = bf16[4,3]{1,0} parameter(1)
  ROOT dot = bf16[48,3]{1,0} dot(convert.4, param_1.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = s8[1,4,8,6]{3,2,1,0} parameter(0)
  p1 = bf16[4,3]{1,0} parameter(1)
  custom-call = bf16[48,3]{1,0} custom-call(p0, p1),
    custom_call_target="__triton",
    called_computations={triton_dot}
  ROOT bitcast.2 = bf16[1,8,6,3]{3,2,1,0} bitcast(custom-call)
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  const HloComputation* dot_computation = module->entry_computation()
                                              ->root_instruction()
                                              ->operand(0)
                                              ->called_computations()[0];
  const HloInstruction* p0 = dot_computation->parameter_instruction(0);
  const HloInstruction* p1 = dot_computation->parameter_instruction(1);
  TF_ASSERT_OK_AND_ASSIGN(const auto analysis,
                          TritonFusionAnalysis::Execute(*dot_computation));
  EXPECT_EQ(*analysis.ScopeParameters(TritonFusionAnalysis::Scope::LHS).begin(),
            p0);
  EXPECT_EQ(*analysis.ScopeParameters(TritonFusionAnalysis::Scope::RHS).begin(),
            p1);
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::LHS, p0, 0),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/8 * 6,
                                    /*slice_start=*/0, /*slice_limit=*/8 * 6,
                                    /*subfragments=*/ElementsAre(6, 8))));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::LHS, p0, 1),
              ElementsAre(FieldsAre(/*stride=*/8 * 6, /*count=*/4,
                                    /*slice_start=*/0, /*slice_limit=*/4,
                                    /*subfragments=*/ElementsAre(4))));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::RHS, p1, 0),
              ElementsAre(FieldsAre(/*stride=*/3, /*count=*/4,
                                    /*slice_start=*/0, /*slice_limit=*/4,
                                    /*subfragments=*/ElementsAre(4))));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::RHS, p1, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/3,
                                    /*slice_start=*/0, /*slice_limit=*/3,
                                    /*subfragments=*/ElementsAre(3))));
}

TEST_F(TritonDotAnalysisTest, TransposeMergeNCN) {
  const std::string hlo_text = R"(
HloModule t

triton_dot {
  param_0.1 = bf16[3,4,8,1]{3,2,1,0} parameter(0)
  transpose.3 = bf16[3,8,1,4]{3,2,1,0} transpose(param_0.1), dimensions={0,2,3,1}
  bitcast.18 = bf16[24,4]{1,0} bitcast(transpose.3)
  param_1.1 = bf16[4,3]{1,0} parameter(1)
  ROOT dot = bf16[24,3]{1,0} dot(bitcast.18, param_1.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = bf16[3,4,8,1]{3,2,1,0} parameter(0)
  p1 = bf16[4,3]{1,0} parameter(1)
  custom-call = bf16[24,3]{1,0} custom-call(p0, p1),
    custom_call_target="__triton", called_computations={triton_dot}
  ROOT bitcast.2 = bf16[3,8,1,3]{3,2,1,0} bitcast(custom-call)
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  const HloComputation* dot_computation = module->entry_computation()
                                              ->root_instruction()
                                              ->operand(0)
                                              ->called_computations()[0];
  const HloInstruction* p0 = dot_computation->parameter_instruction(0);
  const HloInstruction* p1 = dot_computation->parameter_instruction(1);
  TF_ASSERT_OK_AND_ASSIGN(const auto analysis,
                          TritonFusionAnalysis::Execute(*dot_computation));
  EXPECT_EQ(*analysis.ScopeParameters(TritonFusionAnalysis::Scope::LHS).begin(),
            p0);
  EXPECT_EQ(*analysis.ScopeParameters(TritonFusionAnalysis::Scope::RHS).begin(),
            p1);
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::LHS, p0, 0),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/8,
                                    /*slice_start=*/0, /*slice_limit=*/8,
                                    /*subfragments=*/ElementsAre(8)),
                          FieldsAre(/*stride=*/4 * 8, /*count=*/3,
                                    /*slice_start=*/0, /*slice_limit=*/3,
                                    /*subfragments=*/ElementsAre(3))));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::LHS, p0, 1),
              ElementsAre(FieldsAre(/*stride=*/8, /*count=*/4,
                                    /*slice_start=*/0, /*slice_limit=*/4,
                                    /*subfragments=*/ElementsAre(4))));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::RHS, p1, 0),
              ElementsAre(FieldsAre(/*stride=*/3, /*count=*/4,
                                    /*slice_start=*/0, /*slice_limit=*/4,
                                    /*subfragments=*/ElementsAre(4))));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::RHS, p1, 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/3,
                                    /*slice_start=*/0, /*slice_limit=*/3,
                                    /*subfragments=*/ElementsAre(3))));
}

TEST_F(TritonDotAnalysisTest, TransposeOutput) {
  const std::string hlo_text = R"(
HloModule t

triton_dot {
  p0 = bf16[24,4]{1,0} parameter(0)
  p1 = bf16[4,3]{1,0} parameter(1)
  dot = bf16[24,3]{1,0} dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bc = bf16[12,2,3]{2,1,0} bitcast(dot)
  ROOT t = bf16[3,12,2]{2,1,0} transpose(bc), dimensions={2,0,1}
}

ENTRY e {
  p0 = bf16[24,4]{1,0} parameter(0)
  p1 = bf16[4,3]{1,0} parameter(1)
  ROOT r = bf16[3,12,2]{2,1,0} fusion(p0, p1), kind=kCustom,
    calls=triton_dot
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  const HloComputation* dot_computation =
      module->entry_computation()->root_instruction()->called_computations()[0];
  const HloInstruction* dot_output = dot_computation->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(const auto analysis,
                          TritonFusionAnalysis::Execute(*dot_computation));
  EXPECT_THAT(
      *analysis.IterSpec(TritonFusionAnalysis::Scope::OUTPUT, dot_output, 0),
      ElementsAre(FieldsAre(/*stride=*/1, /*count=*/24, /*slice_start=*/0,
                            /*slice_limit=*/24,
                            /*subfragments=*/ElementsAre(2, 12))));
  EXPECT_THAT(
      *analysis.IterSpec(TritonFusionAnalysis::Scope::OUTPUT, dot_output, 1),
      ElementsAre(FieldsAre(/*stride=*/24, /*count=*/3, /*slice_start=*/0,
                            /*slice_limit=*/3,
                            /*subfragments=*/ElementsAre(3))));
}

TEST_F(TritonDotAnalysisTest, OutputParameterIsHandled) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule t

triton_dot {
  p0 = bf16[24,4]{1,0} parameter(0)
  p1 = bf16[4,3]{1,0} parameter(1)
  dot = bf16[24,3]{1,0} dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  p2 = f16[3,24]{1,0} parameter(2)
  p2t = f16[24,3]{1,0} transpose(p2), dimensions={1,0}
  p2tc = bf16[24,3]{1,0} convert(p2t)
  ROOT r = bf16[24,3]{1,0} divide(p2tc, dot)
}

ENTRY e {
  p0 = bf16[24,4]{1,0} parameter(0)
  p1 = bf16[4,3]{1,0} parameter(1)
  p2 = f16[3,24]{1,0} parameter(2)
  ROOT r = bf16[24,3]{1,0} fusion(p0, p1, p2), kind=kCustom,
    calls=triton_dot
})"));
  const HloComputation* dot_computation =
      module->entry_computation()->root_instruction()->called_computations()[0];
  const HloInstruction* output_param =
      dot_computation->parameter_instruction(2);
  TF_ASSERT_OK_AND_ASSIGN(const auto analysis,
                          TritonFusionAnalysis::Execute(*dot_computation));
  EXPECT_EQ(
      analysis.IterSpec(TritonFusionAnalysis::Scope::OUTPUT, output_param, 0)
          ->size(),
      1);
  EXPECT_THAT(
      *analysis.IterSpec(TritonFusionAnalysis::Scope::OUTPUT, output_param, 0),
      ElementsAre(FieldsAre(/*stride=*/1, /*count=*/24, /*slice_start=*/0,
                            /*slice_limit=*/24,
                            /*subfragments=*/ElementsAre(24))));
  EXPECT_EQ(
      analysis.IterSpec(TritonFusionAnalysis::Scope::OUTPUT, output_param, 1)
          ->size(),
      1);
  EXPECT_THAT(
      *analysis.IterSpec(TritonFusionAnalysis::Scope::OUTPUT, output_param, 1),
      ElementsAre(FieldsAre(/*stride=*/24, /*count=*/3, /*slice_start=*/0,
                            /*slice_limit=*/3,
                            /*subfragments=*/ElementsAre(3))));
}

TEST_F(TritonDotAnalysisTest, InputBroadcastFromScalarIsHandled) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule t

triton_dot {
  p0 = bf16[24,4]{1,0} parameter(0)
  p1 = bf16[] parameter(1)
  p1b = bf16[4,3] broadcast(p1)
  ROOT dot = bf16[24,3]{1,0} dot(p0, p1b),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = bf16[24,4]{1,0} parameter(0)
  p1 = bf16[] parameter(1)
  ROOT r = bf16[24,3]{1,0} fusion(p0, p1), kind=kCustom,
    calls=triton_dot
})"));
  const HloComputation* dot_computation =
      module->entry_computation()->root_instruction()->called_computations()[0];
  const HloInstruction* scalar = dot_computation->parameter_instruction(1);
  TF_ASSERT_OK_AND_ASSIGN(const auto analysis,
                          TritonFusionAnalysis::Execute(*dot_computation));
  EXPECT_EQ(analysis.IterSpec(TritonFusionAnalysis::Scope::RHS, scalar, 0),
            nullptr);
  EXPECT_EQ(analysis.IterSpec(TritonFusionAnalysis::Scope::RHS, scalar, 1),
            nullptr);
}

TEST_F(TritonDotAnalysisTest, InputBroadcastFromVectorIsHandled) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule t

triton_dot {
  p0 = bf16[24,4]{1,0} parameter(0)
  p1 = bf16[4] parameter(1)
  p1b = bf16[4,3] broadcast(p1), dimensions={0}
  ROOT dot = bf16[24,3]{1,0} dot(p0, p1b),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = bf16[24,4]{1,0} parameter(0)
  p1 = bf16[4] parameter(1)
  ROOT r = bf16[24,3]{1,0} fusion(p0, p1), kind=kCustom,
    calls=triton_dot
})"));
  const HloComputation* dot_computation =
      module->entry_computation()->root_instruction()->called_computations()[0];
  const HloInstruction* vector = dot_computation->parameter_instruction(1);
  TF_ASSERT_OK_AND_ASSIGN(const auto analysis,
                          TritonFusionAnalysis::Execute(*dot_computation));
  EXPECT_EQ(
      analysis.IterSpec(TritonFusionAnalysis::Scope::RHS, vector, 0)->size(),
      1);
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::RHS, vector, 0),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/4,
                                    /*slice_start=*/0, /*slice_limit=*/4,
                                    /*subfragments=*/ElementsAre(4))));
}

TEST_F(TritonDotAnalysisTest, OutputBroadcastIsNotAccepted) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule t

ENTRY e {
  p0 = f16[2,35] parameter(0)
  p0c = bf16[2,35] convert(p0)
  p1 = bf16[35,2] parameter(1)
  dot = bf16[2,2] dot(p0c, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT bc = bf16[2,2,100] broadcast(dot), dimensions={0,1}
})"));
  EXPECT_TRUE(GemmRewriterTriton(se::CudaComputeCapability{
                                     se::CudaComputeCapability::AMPERE, 0})
                  .Run(module.get())
                  .value());
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kBroadcast);
}

TEST_F(TritonDotAnalysisTest, DegenerateSplitFragmentIsHandled) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
triton_gemm_r {
  Arg_0.1 = s8[30,913,8,21]{3,2,1,0} parameter(0)
  bitcast.6 = s8[30,8,21,913]{2,1,3,0} bitcast(Arg_0.1)
  copy.7 = s8[30,8,21,913]{3,2,1,0} copy(bitcast.6)
  bitcast.8 = s8[5040,913]{1,0} bitcast(copy.7)
  convert.9 = bf16[5040,913]{1,0} convert(bitcast.8)
  bitcast.32 = bf16[58,913]{1,0} parameter(1)
  dot.33 = bf16[5040,58]{1,0} dot(convert.9, bitcast.32),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
  bitcast.34 = bf16[30,8,21,58]{3,2,1,0} bitcast(dot.33)
  copy.35 = bf16[30,8,21,58]{2,1,3,0} copy(bitcast.34)
  ROOT bitcast.41 = bf16[30,1,58,8,21]{4,3,2,1,0} bitcast(copy.35)
}

ENTRY e {
  Arg_0.1 = s8[30,913,8,21]{3,2,1,0} parameter(0)
  Arg_1.2 = bf16[58,913]{1,0} parameter(1)
  ROOT r = bf16[30,1,58,8,21]{4,3,2,1,0} fusion(Arg_0.1, Arg_1.2), kind=kCustom,
    calls=triton_gemm_r,
    backend_config={kind: "__triton_gemm"}
})"));
  const HloComputation* dot_computation =
      module->entry_computation()->root_instruction()->called_computations()[0];
  TF_ASSERT_OK_AND_ASSIGN(const auto analysis,
                          TritonFusionAnalysis::Execute(*dot_computation));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::OUTPUT,
                                 dot_computation->root_instruction(), 0),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/8 * 21,
                                    /*slice_start=*/0, /*slice_limit=*/8 * 21,
                                    /*subfragments=*/ElementsAre(21, 8)),
                          FieldsAre(/*stride=*/8 * 21 * 58, /*count=*/30,
                                    /*slice_start=*/0, /*slice_limit=*/30,
                                    /*subfragments=*/ElementsAre(30))));
}

using TritonSoftmaxAnalysisTest = HloTestBase;

TEST_F(TritonSoftmaxAnalysisTest, DegenerateBatchDimensionIsSupported) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
max {
  p1 = f32[] parameter(1)
  p0 = f32[] parameter(0)
  ROOT m = f32[] maximum(p0, p1)
}

triton_softmax_computation {
  p0 = f32[1,97]{1,0} parameter(0)
  bitcast = f32[97]{0} bitcast(p0)
  constant = f32[] constant(-inf)
  reduce = f32[] reduce(bitcast, constant), dimensions={0}, to_apply=max
  broadcast = f32[1,97]{1,0} broadcast(reduce), dimensions={}
  ROOT subtract = f32[1,97]{1,0} subtract(p0, broadcast)
}

ENTRY e {
  p0 = f32[1,97]{1,0} parameter(0)
  ROOT r = f32[1,97]{1,0} fusion(p0), kind=kCustom,
    calls=triton_softmax_computation,
    backend_config={"kind":"__triton_softmax"}
})"));
  const HloComputation* computation =
      module->entry_computation()->root_instruction()->called_computations()[0];
  TF_ASSERT_OK_AND_ASSIGN(const auto analysis,
                          TritonFusionAnalysis::Execute(*computation));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::OUTPUT,
                                 computation->root_instruction(), 0),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/97,
                                    /*slice_start=*/0, /*slice_limit=*/97,
                                    /*subfragments=*/ElementsAre(97))));
  EXPECT_EQ(analysis.IterSpec(TritonFusionAnalysis::Scope::OUTPUT,
                              computation->root_instruction(), 1),
            nullptr);
}

TEST_F(TritonSoftmaxAnalysisTest, BroadcastIntoBatchDimensionIsSupported) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
c {
  p1 = f32[127]{0} parameter(0)
  ROOT b = f32[125,127]{1,0} broadcast(p1), dimensions={1}
}

ENTRY e {
  p0 = f32[127]{0} parameter(0)
  ROOT t = f32[125,127]{1,0} fusion(p0), kind=kCustom, calls=c
})"));
  const HloComputation* computation =
      module->entry_computation()->root_instruction()->called_computations()[0];
  TF_ASSERT_OK_AND_ASSIGN(const auto analysis,
                          TritonFusionAnalysis::Execute(*computation));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::OUTPUT,
                                 computation->root_instruction(), 0),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/127,
                                    /*slice_start=*/0, /*slice_limit=*/127,
                                    /*subfragments=*/ElementsAre(127))));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::OUTPUT,
                                 computation->root_instruction(), 1),
              ElementsAre(FieldsAre(/*stride=*/127, /*count=*/125,
                                    /*slice_start=*/0, /*slice_limit=*/125,
                                    /*subfragments=*/ElementsAre(125))));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::OUTPUT,
                                 computation->parameter_instruction(0), 0),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/127,
                                    /*slice_start=*/0, /*slice_limit=*/127,
                                    /*subfragments=*/ElementsAre(127))));
  EXPECT_EQ(analysis.IterSpec(TritonFusionAnalysis::Scope::OUTPUT,
                              computation->parameter_instruction(0), 1),
            nullptr);
}

TEST_F(GemmRewriterTritonTest, HandleDotIfCublasRequiresPadding) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m

ENTRY e {
  p0 = f16[5,3] parameter(0)
  p1 = f16[5,7] parameter(1)
  ROOT d = f16[3,7] dot(p0, p1),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
})"));

  const se::CudaComputeCapability cc{se::CudaComputeCapability::VOLTA, 0};
  EXPECT_TRUE(CublasRequiresPadding(
      *xla::Cast<HloDotInstruction>(
          module->entry_computation()->root_instruction()),
      cc));
  EXPECT_TRUE(GemmRewriterTriton(cc).Run(module.get()).value());
}

TEST_F(GemmRewriterTritonTest, FuseSliceOfParameterWithOtherUsers) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = f32[97,121] parameter(0)
  s0 = f32[7,101] slice(p0), slice={[3:10], [10:111]}
  p1 = f32[101,16] parameter(1)
  d = f32[16,7] dot(p1, s0),
    lhs_contracting_dims={0}, rhs_contracting_dims={1}
  s1 = f32[3,33] slice(p0), slice={[10:13], [20:53]}
  ROOT t = tuple(d, s1)
})"));

  const se::CudaComputeCapability cc{se::CudaComputeCapability::VOLTA, 0};
  EXPECT_TRUE(GemmRewriterTriton(cc).Run(module.get()).value());
}

TEST_F(GemmRewriterTritonTest, DoNotFuseSliceOfMixedDimensions) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = bf16[768,64] parameter(0)
  s0 = bf16[768,32] slice(p0), slice={[0:768], [0:32]}
  b0 = bf16[256,3,32] reshape(s0)
  b1 = bf16[256,96] reshape(b0)
  p1 = bf16[256,96] parameter(1)
  ROOT d = bf16[96,96] dot(b1, p1),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
})"));

  const se::CudaComputeCapability cc{se::CudaComputeCapability::AMPERE, 0};
  EXPECT_FALSE(GemmRewriterTriton(cc).Run(module.get()).value());
}

TEST_F(GemmRewriterTritonTest, DoNotFuseSlicesOfNonMajorFragments) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = f32[2,2,256,256] parameter(0)
  s0 = f32[1,1,256,256] slice(p0),
    slice={[0:1], [0:1], [0:256], [0:256]}
  r0 = f32[256,256] reshape(s0)
  p1 = f16[2,2,256,256] parameter(1)
  s1 = f16[1,1,256,256] slice(p1),
    slice={[0:1], [0:1], [0:256], [0:256]}
  r1 = f16[256,256] reshape(s1)
  ROOT d = f32[256,256] dot(r0, r1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));

  const se::CudaComputeCapability cc{se::CudaComputeCapability::AMPERE, 0};
  EXPECT_FALSE(GemmRewriterTriton(cc).Run(module.get()).value());
}

TEST_F(GemmRewriterTritonTest, SliceToDegenerateIsSkipped) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p = f32[3] parameter(0)
  s = f32[1] slice(p), slice={[2:3]}
  r = f32[] reshape(s)
  b = f32[3,3] broadcast(r), dimensions={}
  ROOT d = f32[3,3] dot(b, b),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)"));
  const se::CudaComputeCapability cc{se::CudaComputeCapability::AMPERE, 0};

  ASSERT_TRUE(GemmRewriterTriton(cc).Run(module.get()).value());

  // Slice is not fused.
  MatchHloModule(*module, R"(
; CHECK-NOT: slice
; CHECK: ENTRY
; CHECK: slice
)");
}

TEST_F(GemmRewriterTritonTest, MultipleUsesAreHandled) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  c = f32[] constant(1)
  b = f32[6,8] broadcast(c), dimensions={}
  p0 = f32[6,8] parameter(0)
  a1 = f32[6,8] add(p0, b)
  e = f32[6,8] exponential(a1)
  a2 = f32[6,8] add(e, b)
  d = f32[6,8] divide(b, a2)
  p2 = f16[8,6] parameter(1)
  cv = f32[8,6] convert(p2)
  ROOT r = f32[6,6] dot(d, cv),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));
  const se::CudaComputeCapability cc{se::CudaComputeCapability::AMPERE, 0};
  EXPECT_TRUE(GemmRewriterTriton(cc).Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter(), m::Parameter())));
}

TEST_F(GemmRewriterTritonTest, BinaryElementwiseOfBroadcastIsFused) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p2 = f32[3072] parameter(2)
  b = f32[8192,3072] broadcast(p2), dimensions={1}
  p0 = f16[8192,3072] parameter(0)
  p0c = f32[8192,3072] convert(p0)
  a = f32[8192,3072] add(p0c, b)
  p1 = f32[3072,768] parameter(1)
  ROOT r = f32[8192,768] dot(a, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));
  const se::CudaComputeCapability cc{se::CudaComputeCapability::AMPERE, 0};
  EXPECT_TRUE(GemmRewriterTriton(cc).Run(module.get()).value());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(), m::Parameter(), m::Parameter())));
}

TEST_F(GemmRewriterTritonTest,
       BinaryElementwiseOfUnsupportedBroadcastIsNotFused) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p2 = f32[768] parameter(2)
  b = f32[8192,768,4] broadcast(p2), dimensions={1}
  s = f32[8192,3072] bitcast(b)
  p0 = f16[8192,3072] parameter(0)
  p0c = f32[8192,3072] convert(p0)
  a = f32[8192,3072] add(p0c, s)
  p1 = f32[3072,768] parameter(1)
  ROOT r = f32[8192,768] dot(a, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));
  const se::CudaComputeCapability cc{se::CudaComputeCapability::AMPERE, 0};
  EXPECT_FALSE(GemmRewriterTriton(cc).Run(module.get()).value());
}

class GemmRewriterTritonLevel2Test : public GemmRewriterTritonTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options =
        GemmRewriterTritonTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_triton_fusion_level(2);
    return debug_options;
  }
};

TEST_F(GemmRewriterTritonLevel2Test, ReshapeToScalarIsHandled) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = s8[5,3] parameter(0)
  c = f16[5,3] convert(p0)
  p1 = f16[1] parameter(1)
  r = f16[] reshape(p1)
  b = f16[5,7] broadcast(r)
  ROOT d = f16[3,7] dot(c, b),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
})"));

  EXPECT_TRUE(GemmRewriterTriton(gpu_version_).Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter(), m::Parameter())));
}

TEST_F(GemmRewriterTritonLevel2Test, DoNotFuseIncompatibleDimensionSplits) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p1 = s8[5,7,2,3]{3,2,1,0} parameter(1)
  t1 = s8[7,5,2,3]{3,2,1,0} transpose(p1), dimensions={1,0,2,3}
  r1 = s8[7,30]{1,0} reshape(t1)
  cvt = f16[7,30]{1,0} convert(r1)
  p2 = f16[2,7,5,3]{3,2,1,0} parameter(2)
  t2 = f16[7,2,5,3]{3,2,1,0} transpose(p2), dimensions={1,0,2,3}
  r2 = f16[7,30]{1,0} reshape(t2)
  a = f16[7,30]{1,0} add(cvt, r2)
  p0 = f16[7,79]{1,0} parameter(0)
  ROOT dot = f16[30,79]{1,0} dot(a, p0),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
})"));

  EXPECT_TRUE(GemmRewriterTriton(gpu_version_).Run(module.get()).value());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Transpose(), m::Parameter(), m::Parameter())));
}

TEST_F(GemmRewriterTritonLevel2Test, DoNotFuseTooManyParameters) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  tmp_0 = f32[] constant(1)
  tmp_1 = f32[3,49]{1,0} broadcast(tmp_0), dimensions={}
  tmp_2 = f32[3,49]{1,0} parameter(6)
  tmp_3 = f32[] constant(0)
  tmp_4 = f32[3,49]{1,0} broadcast(tmp_3), dimensions={}
  tmp_5 = pred[3,49]{1,0} compare(tmp_2, tmp_4), direction=GT
  tmp_6 = f32[3,49]{1,0} convert(tmp_5)
  tmp_7 = f32[3,49]{1,0} subtract(tmp_1, tmp_6)
  tmp_8 = s32[] parameter(13)
  tmp_9 = f32[] convert(tmp_8)
  tmp_10 = f32[] maximum(tmp_9, tmp_0)
  tmp_11 = f32[] divide(tmp_3, tmp_10)
  tmp_12 = f32[3,49]{1,0} broadcast(tmp_11), dimensions={}
  tmp_13 = pred[3,49]{1,0} parameter(7)
  tmp_14 = pred[3,49]{1,0} parameter(10)
  tmp_15 = pred[3,49]{1,0} and(tmp_13, tmp_14)
  tmp_16 = f32[3,49]{1,0} convert(tmp_15)
  tmp_17 = f32[3,49]{1,0} multiply(tmp_12, tmp_16)
  tmp_18 = f32[3,49]{1,0} negate(tmp_17)
  tmp_19 = f32[3,49]{1,0} multiply(tmp_7, tmp_18)
  tmp_20 = f32[3,49]{1,0} parameter(19)
  tmp_21 = f32[3,49]{1,0} subtract(tmp_1, tmp_20)
  tmp_22 = f32[3,49]{1,0} divide(tmp_19, tmp_21)
  tmp_23 = f32[3,49]{1,0} negate(tmp_22)
  tmp_24 = f32[3,49]{1,0} negate(tmp_6)
  tmp_25 = f32[3,49]{1,0} multiply(tmp_24, tmp_17)
  tmp_26 = f32[3,49]{1,0} divide(tmp_25, tmp_20)
  tmp_27 = f32[3,49]{1,0} add(tmp_23, tmp_26)
  tmp_28 = f32[3,49]{1,0} parameter(18)
  tmp_29 = f32[3,49]{1,0} multiply(tmp_27, tmp_28)
  tmp_30 = f32[3,49]{1,0} parameter(17)
  tmp_31 = f32[3,49]{1,0} multiply(tmp_29, tmp_30)
  tmp_32 = f32[3,49]{1,0} parameter(16)
  tmp_33 = f32[3,49]{1,0} multiply(tmp_31, tmp_32)
  tmp_34 = f32[3,49]{1,0} parameter(15)
  tmp_35 = f32[3,49]{1,0} add(tmp_33, tmp_34)
  tmp_36 = f32[3,49]{1,0} parameter(14)
  tmp_37 = f32[3,49]{1,0} add(tmp_35, tmp_36)
  tmp_38 = f32[1,1]{1,0} constant({ {0} })
  tmp_39 = f32[1,1]{1,0} broadcast(tmp_38), dimensions={0,1}
  tmp_40 = f32[] reshape(tmp_39)
  tmp_41 = f32[3,32]{1,0} broadcast(tmp_40), dimensions={}
  tmp_42 = u32[48]{0} parameter(11)
  tmp_43 = u32[48]{0} parameter(5)
  tmp_44 = u32[96]{0} concatenate(tmp_42, tmp_43), dimensions={0}
  tmp_45 = u32[3,32]{1,0} reshape(tmp_44)
  tmp_46 = u32[96]{0} reshape(tmp_45)
  tmp_47 = u32[] constant(1)
  tmp_48 = u32[3,32]{1,0} broadcast(tmp_47), dimensions={}
  tmp_49 = u32[96]{0} reshape(tmp_48)
  tmp_50 = u32[96]{0} shift-right-logical(tmp_46, tmp_49)
  tmp_51 = u32[3,32]{1,0} reshape(tmp_50)
  tmp_52 = u32[3,32]{1,0} or(tmp_51, tmp_48)
  tmp_53 = f32[3,32]{1,0} bitcast-convert(tmp_52)
  tmp_54 = f32[3,32]{1,0} broadcast(tmp_0), dimensions={}
  tmp_55 = f32[3,32]{1,0} subtract(tmp_53, tmp_54)
  tmp_56 = f32[1,1]{1,0} constant({ {1} })
  tmp_57 = f32[1,1]{1,0} broadcast(tmp_56), dimensions={0,1}
  tmp_58 = f32[] reshape(tmp_57)
  tmp_59 = f32[3,32]{1,0} broadcast(tmp_58), dimensions={}
  tmp_60 = f32[3,32]{1,0} multiply(tmp_55, tmp_59)
  tmp_61 = f32[3,32]{1,0} add(tmp_60, tmp_41)
  tmp_62 = f32[3,32]{1,0} maximum(tmp_41, tmp_61)
  tmp_63 = f32[3,32]{1,0} broadcast(tmp_3), dimensions={}
  tmp_64 = pred[3,32]{1,0} compare(tmp_62, tmp_63), direction=LT
  tmp_65 = f32[3,32]{1,0} convert(tmp_64)
  tmp_66 = f32[3,49]{1,0} parameter(9)
  tmp_67 = f32[49]{0} parameter(4)
  tmp_68 = f32[3,49]{1,0} broadcast(tmp_67), dimensions={1}
  tmp_69 = f32[3,49]{1,0} add(tmp_66, tmp_68)
  tmp_70 = f32[1,49]{1,0} parameter(12)
  tmp_71 = f32[1,49]{1,0} broadcast(tmp_0), dimensions={}
  tmp_72 = f32[1,49]{1,0} divide(tmp_70, tmp_71)
  tmp_73 = f32[1,49]{1,0} broadcast(tmp_72), dimensions={0,1}
  tmp_74 = f32[49]{0} reshape(tmp_73)
  tmp_75 = f32[3,49]{1,0} broadcast(tmp_74), dimensions={1}
  tmp_76 = f32[3,49]{1,0} subtract(tmp_69, tmp_75)
  tmp_77 = f32[1,49]{1,0} parameter(3)
  tmp_78 = f32[1,49]{1,0} parameter(8)
  tmp_79 = f32[1,49]{1,0} divide(tmp_78, tmp_71)
  tmp_80 = f32[1,49]{1,0} multiply(tmp_72, tmp_72)
  tmp_81 = f32[1,49]{1,0} subtract(tmp_79, tmp_80)
  tmp_82 = f32[1,49]{1,0} add(tmp_81, tmp_71)
  tmp_83 = f32[1,49]{1,0} rsqrt(tmp_82)
  tmp_84 = f32[1,49]{1,0} multiply(tmp_77, tmp_83)
  tmp_85 = f32[1,49]{1,0} broadcast(tmp_84), dimensions={0,1}
  tmp_86 = f32[49]{0} reshape(tmp_85)
  tmp_87 = f32[3,49]{1,0} broadcast(tmp_86), dimensions={1}
  tmp_88 = f32[3,49]{1,0} multiply(tmp_76, tmp_87)
  tmp_89 = f32[1,49]{1,0} parameter(2)
  tmp_90 = f32[1,49]{1,0} broadcast(tmp_89), dimensions={0,1}
  tmp_91 = f32[49]{0} reshape(tmp_90)
  tmp_92 = f32[3,49]{1,0} broadcast(tmp_91), dimensions={1}
  tmp_93 = f32[3,49]{1,0} add(tmp_88, tmp_92)
  tmp_94 = f32[49,32]{1,0} parameter(1)
  tmp_95 = f32[3,32]{1,0} dot(tmp_93, tmp_94), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  tmp_96 = f32[32]{0} parameter(0)
  tmp_97 = f32[3,32]{1,0} broadcast(tmp_96), dimensions={1}
  tmp_98 = f32[3,32]{1,0} add(tmp_95, tmp_97)
  tmp_99 = f32[3,32]{1,0} multiply(tmp_65, tmp_98)
  tmp_100 = f32[3,32]{1,0} divide(tmp_99, tmp_63)
  tmp_101 = f32[3,32]{1,0} maximum(tmp_100, tmp_63)
  ROOT tmp_102 = f32[49,32]{1,0} dot(tmp_37, tmp_101), lhs_contracting_dims={0}, rhs_contracting_dims={0}
})"));

  EXPECT_TRUE(GemmRewriterTriton(gpu_version_).Run(module.get()).value());
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kFusion);
  EXPECT_EQ(module->entry_computation()->root_instruction()->fusion_kind(),
            HloInstruction::FusionKind::kCustom);
  EXPECT_LE(module->entry_computation()->root_instruction()->operand_count(),
            TritonFusionAnalysis::kMaxParameterPerDotScope * 2);
}

TEST_F(GemmRewriterTritonLevel2Test,
       DoNotFuseTooManyParametersWhenAnInstructionWouldAddMultipleParameters) {
  static_assert(TritonFusionAnalysis::kMaxParameterPerDotScope == 4,
                "We have to update this test.");
  // If we fuse the select, it adds 2 additional parameters at once (not 3,
  // because the select instruction itself is removed from the parameters).
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  a = f32[3,49]{1,0} parameter(0)
  b = f32[3,49]{1,0} parameter(1)
  c = pred[3,49]{1,0} parameter(2)
  d = f32[3,49]{1,0} parameter(3)
  e = f32[3,49]{1,0} parameter(4)
  add0 = f32[3,49]{1,0} add(a, b)
  select = f32[3,49]{1,0} select(c, d, e)
  add1 = f32[3,49]{1,0} add(add0, select)
  f = f32[3,32]{1,0} parameter(5)
  ROOT tmp_102 = f32[49,32]{1,0} dot(add1, f), lhs_contracting_dims={0}, rhs_contracting_dims={0}
})"));

  EXPECT_TRUE(GemmRewriterTriton(gpu_version_).Run(module.get()).value());
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kFusion);
  EXPECT_EQ(module->entry_computation()->root_instruction()->fusion_kind(),
            HloInstruction::FusionKind::kCustom);
  EXPECT_LE(module->entry_computation()->root_instruction()->operand_count(),
            TritonFusionAnalysis::kMaxParameterPerDotScope + 1);
}

TEST_F(GemmRewriterTritonLevel2Test,
       InstructionsReachableFromMultipleOperandsAreHandledCorrectly) {
  static_assert(TritonFusionAnalysis::kMaxParameterPerDotScope == 4,
                "We have to update this test.");
  // There was a bug that some dead code was generated into some fusions in a
  // specific edge case. When some instructions were reachable both through the
  // LHS and the RHS operands, the BFS (Breadth-first search) through the LHS1
  // operand "marked" one operation as non-fusible because it would exceed the
  // limit on fusion parameters per operand. But the BFS through the RHS operand
  // went through that node and fused some more operands. So the resulting
  // fusion was not connected and caused errors. This test case checks that such
  // configurations generate a correct HLO now.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  a = f32[2,4]{1,0} parameter(0)
  b = f32[2,4]{1,0} parameter(1)
  c = f32[2,4]{1,0} parameter(2)
  d = f32[2,4]{1,0} parameter(3)
  e = f32[2,4]{1,0} parameter(4)
  add0 = f32[2,4]{1,0} add(a, b)
  add1 = f32[2,4]{1,0} add(add0, c)
  add2 = f32[2,4]{1,0} add(add1, d)
  add3 = f32[2,4]{1,0} add(add2, e)
  ROOT r = f32[2,2]{1,0} dot(add3, add0),
           lhs_contracting_dims={1}, rhs_contracting_dims={1}
})"));

  EXPECT_TRUE(GemmRewriterTriton(gpu_version_).Run(module.get()).value());
  // ~VerifiedHloModule() will verify the module.
}

TEST_F(GemmRewriterTritonLevel2Test, EachScopeIsFusedToASeparateSubgraph) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  a = f32[2,4]{1,0} parameter(0)
  b = f32[2,4]{1,0} parameter(1)
  add = f32[2,4]{1,0} add(a, b)
  ROOT r = f32[2,2]{1,0} dot(add, add),
           lhs_contracting_dims={1}, rhs_contracting_dims={1}
})"));

  EXPECT_TRUE(GemmRewriterTriton(gpu_version_).Run(module.get()).value());

  MatchHloModule(*module, R"(
CHECK-DAG: %[[P0:.*]] = f32[2,4]{1,0} parameter(0)
CHECK-DAG: %[[P1:.*]] = f32[2,4]{1,0} parameter(1)
CHECK-DAG: %[[ADD0:.*]] = f32[2,4]{1,0} add(f32[2,4]{1,0} %[[P0]], f32[2,4]{1,0} %[[P1]])
CHECK-DAG: %[[P2:.*]] = f32[2,4]{1,0} parameter(2)
CHECK-DAG: %[[P3:.*]] = f32[2,4]{1,0} parameter(3)
CHECK-DAG: %[[ADD1:.*]] = f32[2,4]{1,0} add(f32[2,4]{1,0} %[[P2]], f32[2,4]{1,0} %[[P3]])
CHECK-DAG: ROOT {{.*}} = f32[2,2]{1,0} dot(f32[2,4]{1,0} %[[ADD0]], f32[2,4]{1,0} %[[ADD1]])
CHECK: ENTRY
CHECK-DAG: %[[P0:.*]] = f32[2,4]{1,0} parameter(0)
CHECK-DAG: %[[P1:.*]] = f32[2,4]{1,0} parameter(1)
CHECK-DAG: ROOT {{.*}} = f32[2,2]{1,0}
CHECK-SAME: fusion(f32[2,4]{1,0} %[[P0]], f32[2,4]{1,0} %[[P1]], f32[2,4]{1,0} %[[P0]], f32[2,4]{1,0} %[[P1]]),
CHECK-SAME: kind=kCustom
CHECK-SAME: __triton_gemm
})");
}

TEST_F(GemmRewriterTritonLevel2Test,
       OperationsAddingMoreParametersGetMultipleTries) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
e {
  p0 = f32[2,2] parameter(0)
  c0 = f32[] constant(12345)
  b0 = f32[2,2] broadcast(c0), dimensions={}
  m0 = f32[2,2] multiply(p0, b0)
  c1 = f32[] constant(34567)
  b1 = f32[2,2] broadcast(c1), dimensions={}
  a0 = f32[2,2] add(m0, b1)
  b3 = f32[2,2,2] broadcast(a0), dimensions={0,1}
  p2 = f32[2,2,2] parameter(2)
  m2 = f32[2,2,2] multiply(p2, b3)
  p1 = f32[2]{0} parameter(1)
  c2 = f32[] constant(5678)
  b2 = f32[2] broadcast(c2), dimensions={}
  a1 = f32[2]{0} add(p1, b2)
  b4 = f32[2,2,2] broadcast(a1), dimensions={2}
  m1 = f32[2,2,2] multiply(m2, b4)
  b = f32[4,2] bitcast(m1)
  p3 = f16[2,2] parameter(3)
  p3c = f32[2,2] convert(p3)
  ROOT r = f32[4,2] dot(b, p3c),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));

  EXPECT_TRUE(GemmRewriterTriton(gpu_version_).Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch((m::Fusion(m::Parameter(), m::Parameter(),
                                    m::Parameter(), m::Parameter()))));
}

TEST_F(GemmRewriterTritonLevel2Test, FusionLevelIsLimitedOnVolta) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = f32[2,53] parameter(0)
  p0e = f32[2,53] exponential(p0)
  p1 = s16[53,2] parameter(1)
  p1c = f32[53,2] convert(p1)
  ROOT dot = f32[2,2] dot(p0e, p1c),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));
  EXPECT_TRUE(GemmRewriterTriton(se::CudaComputeCapability{
                                     se::CudaComputeCapability::VOLTA, 0})
                  .Run(module.get())
                  .value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch((m::Fusion(m::Parameter(), m::Exp()))));
}

TEST_F(GemmRewriterTritonLevel2Test, ParameterUsedElementwiseTwiceIsFused) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule t

ENTRY e {
  p0 = f32[2,35] parameter(0)
  p0n = f32[2,35] negate(p0)
  p0e = f32[2,35] exponential(p0)
  a = f32[2,35] add(p0e, p0n)
  p1 = f16[35,2] parameter(1)
  p1c = f32[35,2] convert(p1)
  ROOT dot = f32[2,2] dot(a, p1c),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));
  EXPECT_TRUE(GemmRewriterTriton(se::CudaComputeCapability{
                                     se::CudaComputeCapability::AMPERE, 0})
                  .Run(module.get())
                  .value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch((m::Fusion(m::Parameter(), m::Parameter()))));
  TF_ASSERT_OK_AND_ASSIGN(
      const auto analysis,
      TritonFusionAnalysis::Execute(*module->entry_computation()
                                         ->root_instruction()
                                         ->called_computations()[0]));
  EXPECT_EQ(analysis.ScopeParameters(TritonFusionAnalysis::Scope::LHS).size(),
            1);
  EXPECT_EQ(analysis.ScopeParameters(TritonFusionAnalysis::Scope::RHS).size(),
            1);
}

TEST_F(GemmRewriterTritonLevel2Test,
       ParameterUsedNonElementwiseTwiceIsFusedOnlyOnOnePath) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule t

ENTRY e {
  p0 = f32[4,4] parameter(0)
  p0t = f32[4,4] transpose(p0), dimensions={1,0}
  a = f32[4,4] add(p0, p0t)
  p1 = f16[4,5] parameter(1)
  p1c = f32[4,5] convert(p1)
  ROOT dot = f32[4,5] dot(a, p1c),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));
  EXPECT_TRUE(GemmRewriterTriton(se::CudaComputeCapability{
                                     se::CudaComputeCapability::AMPERE, 0})
                  .Run(module.get())
                  .value());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch((m::Fusion(m::Parameter(), m::Transpose(), m::Parameter()))));
}

TEST_F(GemmRewriterTritonLevel2Test,
       ComputationParameterWithMultipleUsersIsNotTrivialToFuse) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = f32[400,400] parameter(0)

  c0 = f16[400,400] convert(p0)
  p1 = f16[400,400] parameter(1)
  dot0 = f16[400,400] dot(c0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}

  c1 = f16[400,400] convert(p0)
  p2 = f16[400,400] parameter(2)
  dot1 = f16[400,400] dot(c1, p2),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}

  ROOT a = f16[400,400] add(dot0, dot1)
})"));
  EXPECT_FALSE(GemmRewriterTriton(se::CudaComputeCapability{
                                      se::CudaComputeCapability::AMPERE, 0})
                   .Run(module.get())
                   .value());
}

TEST_F(GemmRewriterTritonLevel2Test, NarrowingConversionIsAlwaysBetterToFuse) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = s8[512,512] parameter(0)
  c0 = f16[512,512] convert(p0)
  p1 = f16[512,512] parameter(1)
  dot0 = f16[512,512] dot(c0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}

  n = f16[512,512] negate(c0)
  ROOT a = f16[512,512] add(dot0, n)
})"));
  EXPECT_TRUE(GemmRewriterTriton(se::CudaComputeCapability{
                                     se::CudaComputeCapability::AMPERE, 0})
                  .Run(module.get())
                  .value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch((m::Add(m::Fusion(m::Parameter(), m::Parameter()),
                                 m::Negate()))));
}

TEST_F(GemmRewriterTritonLevel2Test, NestedSlicingIsAnalyzedCorrectly) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
triton_gemm_d_computation {
  p0 = f32[6,24]{1,0} parameter(0)
  s1 = f32[5,20]{1,0} slice(p0), slice={[1:6], [3:23]}
  n1 = f32[5,20]{1,0} negate(s1)
  s2 = f32[3,7]{1,0} slice(n1), slice={[1:4], [13:20]}
  p1 = f32[7,37]{1,0} parameter(1)
  ROOT d = f32[3,37]{1,0} dot(s2, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[7,37]{1,0} parameter(0)
  p1 = f32[6,24]{1,0} parameter(1)
  ROOT triton_gemm_d = f32[3,37]{1,0} fusion(p1, p0), kind=kCustom,
    calls=triton_gemm_d_computation
})"));
  const HloComputation* computation =
      module->entry_computation()->root_instruction()->called_computations()[0];
  TF_ASSERT_OK_AND_ASSIGN(const auto analysis,
                          TritonFusionAnalysis::Execute(*computation));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::LHS,
                                 computation->parameter_instruction(0), 0),
              ElementsAre(FieldsAre(/*stride=*/24, /*count=*/6,
                                    /*slice_start=*/2, /*slice_limit=*/5,
                                    /*subfragments=*/ElementsAre(3))));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::LHS,
                                 computation->parameter_instruction(0), 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/24,
                                    /*slice_start=*/16, /*slice_limit=*/23,
                                    /*subfragments=*/ElementsAre(7))));
}

TEST_F(GemmRewriterTritonLevel2Test, FusedConcatenationIsAnalyzedCorrectly) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
e {
  p0 = s8[153,1536] parameter(0)
  p1 = s8[153,128] parameter(1)
  p2 = s8[153,256] parameter(2)
  cat = s8[153,1920] concatenate(p0, p1, p2), dimensions={1}
  cvt = bf16[153,1920] convert(cat)
  p3 = bf16[16,153] parameter(3)
  ROOT d = bf16[16,1920] dot(p3, cvt),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));
  EXPECT_TRUE(GemmRewriterTriton(se::CudaComputeCapability{
                                     se::CudaComputeCapability::AMPERE, 0})
                  .Run(module.get())
                  .value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch((m::Fusion(m::Parameter(), m::Parameter(),
                                    m::Parameter(), m::Parameter()))));
  const HloComputation* computation =
      module->entry_computation()->root_instruction()->called_computations()[0];
  TF_ASSERT_OK_AND_ASSIGN(const auto analysis,
                          TritonFusionAnalysis::Execute(*computation));

  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::RHS,
                                 computation->parameter_instruction(0), 0),
              ElementsAre(FieldsAre(/*stride=*/1536, /*count=*/153,
                                    /*slice_start=*/0, /*slice_limit=*/153,
                                    /*subfragments=*/ElementsAre(153))));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::RHS,
                                 computation->parameter_instruction(0), 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/1536,
                                    /*slice_start=*/0, /*slice_limit=*/1536,
                                    /*subfragments=*/ElementsAre(1536))));

  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::RHS,
                                 computation->parameter_instruction(1), 0),
              ElementsAre(FieldsAre(/*stride=*/128, /*count=*/153,
                                    /*slice_start=*/0, /*slice_limit=*/153,
                                    /*subfragments=*/ElementsAre(153))));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::RHS,
                                 computation->parameter_instruction(1), 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/128,
                                    /*slice_start=*/0, /*slice_limit=*/128,
                                    /*subfragments=*/ElementsAre(128))));

  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::RHS,
                                 computation->parameter_instruction(2), 0),
              ElementsAre(FieldsAre(/*stride=*/256, /*count=*/153,
                                    /*slice_start=*/0, /*slice_limit=*/153,
                                    /*subfragments=*/ElementsAre(153))));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::RHS,
                                 computation->parameter_instruction(2), 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/256,
                                    /*slice_start=*/0, /*slice_limit=*/256,
                                    /*subfragments=*/ElementsAre(256))));
}

TEST_F(GemmRewriterTritonLevel2Test, IndivisibleConcatenationIsNotFused) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
e {
  p0 = s8[124,1024] parameter(0)
  p1 = s8[124,1001] parameter(1)
  cat = s8[124,2025] concatenate(p0, p1), dimensions={1}
  cvt = f16[124,2025] convert(cat)
  p2 = f16[123,124] parameter(2)
  ROOT d = f16[2025,123] dot(cvt, p2),
    lhs_contracting_dims={0}, rhs_contracting_dims={1}
})"));
  EXPECT_TRUE(GemmRewriterTriton(se::CudaComputeCapability{
                                     se::CudaComputeCapability::AMPERE, 0})
                  .Run(module.get())
                  .value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch((m::Fusion(m::Concatenate(), m::Parameter()))));
}

TEST_F(GemmRewriterTritonLevel2Test, ConcatenationOfContractingIsNotFused) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
e {
  p0 = s8[124,1024] parameter(0)
  p1 = s8[124,1024] parameter(1)
  cat = s8[124,2048] concatenate(p0, p1), dimensions={1}
  cvt = f16[124,2048] convert(cat)
  p2 = f16[123,2048] parameter(2)
  ROOT d = f16[124,123] dot(cvt, p2),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
})"));
  EXPECT_TRUE(GemmRewriterTriton(se::CudaComputeCapability{
                                     se::CudaComputeCapability::AMPERE, 0})
                  .Run(module.get())
                  .value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch((m::Fusion(m::Concatenate(), m::Parameter()))));
}

TEST_F(GemmRewriterTritonLevel2Test, ConcatenationOfBatchIsNotFused) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
e {
  p0 = s8[124,1024,50] parameter(0)
  p1 = s8[124,1024,50] parameter(1)
  cat = s8[124,2048,50] concatenate(p0, p1), dimensions={1}
  cvt = f16[124,2048,50] convert(cat)
  p2 = f16[123,2048,50] parameter(2)
  ROOT d = f16[2048,124,123] dot(cvt, p2),
    lhs_batch_dims={1}, rhs_batch_dims={1},
    lhs_contracting_dims={2}, rhs_contracting_dims={2}
})"));
  EXPECT_TRUE(GemmRewriterTriton(se::CudaComputeCapability{
                                     se::CudaComputeCapability::AMPERE, 0})
                  .Run(module.get())
                  .value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch((m::Fusion(m::Concatenate(), m::Parameter()))));
}

TEST_F(GemmRewriterTritonLevel2Test,
       TwoConcatenationsOfSameParametersAreNotFused) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
e {
  p0 = s8[128,2] parameter(0)
  p1 = s8[128,2] parameter(1)
  cat0 = s8[256,2] concatenate(p0, p1), dimensions={0}
  cvt0 = f16[256,2] convert(cat0)
  cat1 = s8[256,2] concatenate(p1, p0), dimensions={0}
  n1 = s8[256,2] negate(cat1)
  cvt1 = f16[256,2] convert(n1)
  a = f16[256,2] add(cvt1, cvt0)
  p2 = f16[2,18] parameter(2)
  ROOT d = f16[18,256] dot(p2, a),
    lhs_contracting_dims={0}, rhs_contracting_dims={1}
})"));

  EXPECT_TRUE(GemmRewriterTriton(se::CudaComputeCapability{
                                     se::CudaComputeCapability::AMPERE, 0})
                  .Run(module.get())
                  .value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch((m::Fusion(m::Concatenate(), m::Concatenate(),
                                    m::Parameter()))));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
