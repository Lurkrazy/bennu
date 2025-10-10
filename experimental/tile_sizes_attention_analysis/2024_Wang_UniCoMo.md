---
title: "2024_Wang_UniCoMo"
source_path: "2024_Wang_UniCoMo.pdf"
md5: "f0c41bc4dfe69fa5c6b7bdb03af5b42b"
page_count: 9
created_at: "2025-10-09T19:30:17.463579-06:00"
toolchain: ["pdfminer.six"]
---

## Page 1

2024 IEEE 42nd International Conference on Computer Design (ICCD)

UniCoMo: A Unified Learning-Based Cost Model
for Tensorized Program Tuning

Zihan Wang, Lei Gong(cid:66), Wenqi Lou, Qianyu Cheng, Xianglan Chen, Chao Wang(cid:66), Xuehai Zhou
University of Science and Technology of China
Suzhou Institute for Advanced Research, University of Science and Technology of China
{leigong0203, cswang}@ustc.edu.cn

Abstract—Tensorized programs use hardware intrinsics on
accelerators to significantly improve tensor computation per-
formance. The trend of hardware customization has led to the
emergence of massive hardware accelerators and intrinsics, which
poses a significant engineering challenge for manually coding
efficient tensorized programs. Tensorized program tuning in deep
learning compilers (DLCs) is considered an effective approach to
address the challenge. At the core of program tuning relies the
design of the cost model, but currently there is still a lack of
cost models specifically designed for tensorized programs, which
hampers the co-optimization of DLCs and hardware accelerators.
In this paper, we propose UniCoMo, a unified cost model for
tensorized program tuning on various hardware platforms and
hardware intrinsics. We first analyze the design challenges intro-
duced by tensorized programs for cost models in terms of feature
representation and transfer prediction. And then, for feature
representation, we propose a unified feature representation for
tensorized programs by using program behavior as a template,
mining program features with the schedule attention matrix,
and incorporating hardware intrinsic abstraction. For transfer
prediction, we propose a unified transfer prediction strategy for
tensorized program cost model based on lifelong learning and
transfer learning. To meet training and testing requirements, we
constructed a dataset dedicated to tensorized program tuning. Re-
sults show that UniCoMo maintains the state-of-the-art accuracy
while significantly improving adaptability to diverse execution
environments and enabling flexible transfer prediction. It can
speed up search time by 9.8× and improve inference speed by
1.9× in the state-of-the-art tensorized program tuning framework
(TVM MetaSchedule). To the best of our knowledge, UniCoMo
is the first cost model for tensorized programs. The code will be
open-sourced at https://github.com/ZhW-loop/UniCoMo.

Index Terms—cost model,tensorized program,lifelong learning

I. INTRODUCTION

With the rise of AI algorithms such as deep learning and
large models [1] [2], customized hardware accelerators like
NVIDIA Tensor Core [3], Intel AVX512/VNNI [4], and ARM
NEON/SDOT [5] have proven to be essential means to ad-
dress performance issues in related applications. Correspond-
ingly, these hardware backends offer intrinsics at the software
programming level
to invoke underlying customized units
for accelerating tensor computation. The trend of hardware
customization has led to the emergence of massive hardware
accelerators and intrinsics [6]. These hardware intrinsics typ-
ically operate on multi-dimensional tensor regions and effec-
tively perform tensor operations such as multi-dimensional
loads, dot product, and matrix multiplication. The process of
transforming and scheduling the original tensor program

TABLE I: Tensor Program vs. Tensorized Program

Comparison
Execution
Behavior
Rate of Change
Diversity

Tensor Program
Arithmetic Operations
Freedom
Gradual Change
Only Platform

Tensorized Program
Hardware Intrinsics
Relatively Fixed
Rapid Growth
Intrinsic & Platform

However,

using these hardware intrinsics is called tensorization, and
the corresponding output is called tensorized program [7].
To effectively leverage these hardware intrinsics for accel-
erating tensor computations, in recent years, some advanced
work has begun to pay attention to the related issues of
tensorized program tuning [8] [9] [10] [11]. Similar to tradi-
tional tensor program tuning, tensorized program tuning for
a specific workload involves a vast search space. Due to
differences in tensorization-aware schedules [11], all points
in the space are programs with the same functionality but
different performance, and a DLC needs to search for the
best performing program in this space. However, obtaining the
actual runtime latency of programs under different schedules
involves unacceptable time costs. To address this issue, many
DLCs resort to the cost model – using the predicted latency
from the cost model as the criterion [12] [13]. Therefore, as
the core of program tuning, the accuracy and the efficiency of
the cost model is crucial for searching for the optimal program.
there are significant differences between ten-
sorized programs and normal tensor programs, as shown in
Table I. First, while normal tensor programs run on general
arithmetic processing units (ALU on CPUs, and CUDA Core
on GPUs), tensorized programs mainly rely on customized
acceleration components. Thus, the configuration and invoca-
tion of hardware intrinsics become the main factor affecting
tensorized program performance. Second, tensorized programs
are affected by the diversity of hardware platforms and hard-
ware intrinsics, posing great challenges for their cost model’s
generalization across different hardware. This challenge will
increasingly acute due to the continuous iteration of various
NPUs [14]. In contrast, normal tensor programs are more sta-
ble due to mature general-purpose components. The above two
differences make it difficult to effectively apply a cost model
for tensor programs to accurately predict the performance
of tensorized programs. Last but most important, tensorized
programs generally exhibit unified behavior restricted by the
intrinsic, which potentially facilitates the proposal of a unified
cost model for tensorized program cost model.

2576-6996/24/$31.00 ©2024 IEEE
DOI 10.1109/ICCD63220.2024.00081

487

Authorized licensed use limited to: The University of Utah. Downloaded on October 05,2025 at 05:16:55 UTC from IEEE Xplore.  Restrictions apply.

.

.

1
8
0
0
0
4
2
0
2
0
2
2
3
6
D
C
C
I
/
9
0
1
1
0
1
:
I

.

O
D
|
E
E
E
I

.

4
2
0
2
©
0
0
1
3
$
/
4
2
/
8
-
0
4
0
8
-
3
0
5
3
-
8
-
9
7
9
|
)

D
C
C
I
(
n
g
i
s
e
D
r
e
t
u
p
m
o
C
n
o
e
c
n
e
r
e
f
n
o
C

l

a
n
o
i
t
a
n
r
e
t
n

I

d
n
2
4
E
E
E
I

4
2
0
2
---

## Page 2

According to the requirements of an efficient cost model
specialized for tensorized programs, taking into account the
program variance and customized hardware diversity induced
by tensorization, we posit that the efficient cost model involves
two key points: ❶ feature representation, i.e., how to represent
programs as features learnable by the model; ❷ transfer
prediction, i.e., how the model can support predictions on
untrained hardware at low cost and with flexibility. Focusing
on systematic sorting out and solving the above two points, to
the best of our knowledge, we propose the first unified cost
model for tensorized program tuning by introducing a unified
feature representation and unified transfer prediction strategy.
The main contributions are as follows:

• We systematically and comprehensively analyze the de-
sign challenges of the cost model introduced by ten-
sorized programs in terms of feature representation and
transfer prediction.

• We propose a unified feature representation for tensorized
programs by using program behavior as a template, min-
ing program features with the schedule attention matrix,
and incorporating hardware intrinsic abstraction.

• We propose a unified transfer prediction strategy for
tensorized program cost model based on lifelong learning
and transfer learning.

• We self-produce and open-source a tensorized program
dataset called TensorizeSet, which contains 10 hardware
intrinsics and 84 common neural network structures. It
effectively meets the training and testing requirements of
the cost model for tensorized program tuning.

• We propose UniCoMo, the first cost model for tensorized
program tuning. Results show that UniCoMo maintains
the state-of-the-art (SOTA) accuracy while significantly
improving adaptability to diverse execution environments
and enabling flexible transfer prediction. It can speed up
search time by 9.8× and improve inference speed by
1.9× in the SOTA tensorized program tuning framework
(TVM MetaSchedule).

Fig. 1: Tensorized program space generation.
the specific workload from users and hardware intrinsics from
one or more hardware platforms (heterogeneous computing
cluster) as input. Hardware intrinsics are generally abstracted
into two parts, semantics and implementations, where seman-
tics describe the computational function and implementations
describe the calling interface to the underlying acceleration
core. First, the tensorization step, which does not present in
traditional tensor program tuning, matches the workload
to one or more semantics through special schedule primitives,
such as ReIndex and TransformLayout [11]. Each successful
match on each semantic will generate a tensorization candidate
which manifests as forms of loop calls for tensorized opera-
tions. The tensorization step introduces hardware platform di-
versity and intrinsic diversity to the search space. Furthermore,
for each tensorization candidate, the scheduling step varies
how hardware intrinsics are invoked by applying basic sched-
ule primitives, which are often used in tensor program tuning,
e.g., loop tiling, and then adds corresponding data movement
optimization. Rich scheduling strategies further expand the
search space, but programs in the space exhibit relatively
unified behavior restricted by fixed semantics. The above
two steps generate a tensorized program space with hardware
platform diversity, intrinsic diversity, and rich schedules.

II. BACKGROUND AND MOTIVATION

B. Motivation

A. Background

Tensor program tuning is gradually coming into the lime-
light with the TVM series of work [7] [12] [13] [15]. Tuning
involves search space generation and exploration of the space.
Guided by search algorithms (genetic algorithm, simulated
annealing, etc.), the compiler applies various combinations
tensor program. In
of schedule primitives to the original
the process, massive tensor programs are generated, and the
compiler uses a cost model to filter out candidates with the
best possible performance. After that, all the candidates will
be deployed on the target hardware and actual execution
performance will be obtained to find the best-performing one.
Except for search space generation, the process of tensorized
program tuning is similar to traditional tensor program tuning.
As shown in Fig. 1, DLC adopts a tensorization-aware schedul-
ing method to generate the tensorized program space. It takes

We analyze the design challenges introduced by tensorized
programs for cost models from two aspects, feature repre-
sentation and transfer prediction, combining deficiencies of
TLP [16] and TenSet [17],
two currently most advanced
and representative works on cost models for tensor program
tuning1, and explain why designing a new cost model for
tensorized programs is necessary.

1) Feature Representation: Considering the diversity of
hardware platforms and intrinsics, the ideal feature represen-
tation for tensorized programs should pursue generality while
ensuring prediction accuracy. The top and bottom of Fig. 2
(a) respectively show the feature extraction process of TLP
and TenSet. It is difficult to construct effective feature rep-
resentations for tensorized programs by directly transferring
existing methods from TLP and TenSet. For prediction accu-
racy, TenSet constructs program features based on AST, which

1They are for tensor program tuning. TLP has SOTA accuracy. TenSet is used
in mainstream DLCs. Their datasets only contain float32 tensor programs.

Authorized licensed use limited to: The University of Utah. Downloaded on October 05,2025 at 05:16:55 UTC from IEEE Xplore.  Restrictions apply.

488

Tensorization StepIntrinsicsonGPUTensorCorei, j, k in(64, 256, 128):C[i, j] += A[i, k] * B[k, j]Input Tensor ProgramHardware Intrinsicstensorcore.m8n32k16.nn.fp16# Semanticsi, j, k in(8, 32, 16):C[i, j] += A[i, k] * B[k, j]IntrinsicsonX86CPUx86.avx512/vnni.dot.int8# Semanticsi, k in(16, 4):C[i] += A[k] * B[i, k]IntrinsicsonARMCPUarm.neon/sdot.dot.int8# Semanticsi, k in(4, 4):C[i] += A[k] * B[i, k]TensorizationCandidatesi, j, k in(8, 8, 8):Matmul(8, 32, 16)Tensorized Program Spacei0, j0 in (?,?):i1, j1, k0 in(?,?,?):read_optimization(A, B)i2, j2, k1 in(?,?,?):i3, j3 in (?,?):tensorcore.m8n32k16.nn.fp16write_optimization(C)// Scheduling on GPU// Apply Basic Schedules// “?”: optional Tiling// Program Transformation// Semantic Matching// One Tensorization CandidateScheduling Step
---

## Page 3

Fig. 2: Analysis of design requirements for tensorized program cost model.

hardware intrinsics mainly in two directions: (1) Extending se-
mantics to flexibly support tensor computation workloads. This
includes different semantics on the same platform (Tensor
Core supporting matrix multiplication of various data types,
shapes, and layouts, X86 AVX512 supporting dot product of
multiple data types), and different semantics on different plat-
forms. (2) Improving performance while maintaining seman-
tics unchanged. Generally, microarchitectural upgrades change
the implementation of hardware intrinsics (from AVX512 to
VNNI on X86, from NEON to SDOT on ARM).

Based on the above development trends, we decompose
the originally complex transfer prediction into two orthogonal
requirements: Req 4, for semantic expansion,
it requires
the prediction to support new semantics at
low cost and
with flexibility while always maintaining the generalization
ability across various semantics, thereby coping with multiple
semantics (multiple tensorization candidates) in the tuning
process. We refer to such issues as cross-semantic transfer pre-
diction. Req 5, for performance improvement with semantic
invariance, it requires to transfer cost model originally trained
on low-performance implementations to high-performance im-
plementations at low cost and with flexibility, i.e., allowing
the cost model to perceive the impact of changes in intrinsic
implementations on optimal schedule. We refer to such issues
as cross-performance transfer prediction.

III. OVERVIEW AND KEY IDEAS
Regarding the motivation above, we present the key ideas

for implementing UniCoMo.

A. Insight into Tensorized Program Characteristic

As shown in Fig. 3, to make the feature representation meet
the first three requirements in the motivation, we have insight

involves complex and cumbersome expert knowledge of hard-
ware architecture. Limited by prior knowledge, expert features
cannot sufficiently reflect the impact of schedule differences on
program performance. To address the above two issues, TLP
directly treats schedule primitives as an NLP task, simplifying
feature design while achieving SOTA accuracy. For hardware
intrinsic diversity, neither of them contains hardware intrinsic
abstraction in the program features, thus being unable to work
in search space with multiple intrinsics (multiple tensorization
candidates), as confirmed in the experiment section. Besides,
the NLP-based feature approach makes it difficult for TLP
to incorporate hardware intrinsic abstraction and is prone to
cause to the issue of ”out of vocabulary” when embedding
tensorization schedule primitives. Although AST contains
hardware intrinsic information, TenSet does not propose a
feature representation for intrinsics. For hardware platform
diversity, although AST itself is platform-independent, expert
features vary greatly across platforms. Schedule primitives also
vary greatly across hardware platforms and customized DLCs.
Considering the background of tensorized programs and
deficiencies of existing works, the ideal feature representation
for tensorized programs should meet
the following three
requirements: Req 1, it requires features to effectively reflect
the impact of schedule differences on program performance
to ensure the cost model has high prediction accuracy in
specific hardware environments. Req 2, it requires features
to include hardware intrinsic abstractions to reflect the impact
of tensorization on program performance, enabling the cost
model to cope with the hardware intrinsic diversity. Req 3, it
requires features to unifiedly represent tensorized programs
running on various hardware platforms, enabling the cost
model to cope with the hardware platform diversity.

2) Transfer Prediction: Transfer prediction refers to how
the model can support predictions on untrained hardware at
low cost and with flexibility. The top and bottom of Fig. 2
(b) compare the transfer prediction problems faced by tensor
programs and tensorized programs. The cost model for normal
tensor programs typically focuses only on prediction between
different hardware platforms of the same type, while the cost
model for tensorized programs needs to further consider the
hardware intrinsic diversity, making their transfer prediction
more complex. On the other side, hardware vendors develop

Fig. 3: Key ideas for unified feature representation.

Authorized licensed use limited to: The University of Utah. Downloaded on October 05,2025 at 05:16:55 UTC from IEEE Xplore.  Restrictions apply.

489

Workload Program:…Matmul(64, 256, 128)i,j,kin(64,256,128):C[i, j] += A[i, k]*B[k, j]……Schedule Primitives:…transformlayout(vj,vk:(vk,vj))split(loop7, 16)transformlayout(i0:i0//8//(v27*v28))…1:Accuracy 2:Intrinsic Diversity 3:Platform Diversity4: Cross Semantic  5: Cross Performancetransformlayout[1,0,0...]split           [0,1,0...]vj,vk:(vk,vj)            1loop72i0:i0//8//(v27*v28)      3Cons: Out of Vocabulary[1,0,0,..., 1][0,1,0,..., 2,16][1,0,0,..., 3]Cons: No IntrinsicPros: Accurate AttentionPrimitives-based Feature Representation such as TLPReq 1 √Req 2 ×Req 3 ×Program-based Feature Representation such as TenSet…ForNodeRegionNodeBufferNode…Abstract Syntax TreePros: HW Independent Feature ExtractorCons: Complex&LimitedExpert AST Featuresreuse distance, arithmetic intensity,cache line …NLP workflowAST-learn workflowMLP/XgboostCons: InaccurateReq 1 ×Req 2 ×Req 3 √Token Embedding TableToken Features(a) Feature Representation(b) Transfer Predicting (TP)Figure 1 Analysis of design requirements for Tensorized Program Tensor ProgramPlatformDiversityTP within same classc1c2c3g1g2g3Requirements:CPUs(c1-c3): Intel, AMD, ARMGPUs(g1-g3): A100, T4, V100Req 4 Req 5Tensorized ProgramCross-Semantic TPCross-Performance TPi3i4i5i6i1i2i3i5X86 Intrinsic:i3.avx512.dot.int8i4.vnni.dot.int8ARM Intrinsic:i5.neon.dot.int8i6.sdot.dot.int8GPU Intrinsic:i1.tscore.m8n32k16i2.tscore.m16n16k16Req 4 Req 5Feature ExtractorLoadComputeStoreBehaviorTemplateAST-basedFeaturesMine Key ScheduleIntrinsicAbstractionKey MethodKey ObservationReq2 √Req1 √ASTIntrinsicKey PremiseReq3 √Unified Feature RepresentationSSchedulePrimitivesAttentionQueryKeyAttention MatrixApplyTLPon Tensorized ProgramSTensorizationDatasetFig 3. Key ideas for unified feature representation
---

## Page 4

into the tensorized program characteristic. As mentioned,
tensorized programs exhibit a unified program behavior, which
can be abstracted into a three-stage execution model: first, the
program reads data from outermost storage through multi-level
caches into dedicated storage, then performs the tensorized
operations specified by intrinsics, and finally writes back the
results to outermost storage. This unified behavior remains
consistent across hardware platforms and customized DLCs.
Therefore, the behavior template allows tensorized programs
running on different platforms to be unifiedly represented, thus
coping with hardware platform diversity (meeting Req 3). We
describe any tensorized program by combining the above three
behaviors, and then abstract a cross-platform unified program
feature template. Based on the template, we embed program
features with different hardware intrinsics and schedules,
thereby constructing a unified feature representation.

On this basis, considering AST as a comprehensive pro-
gram abstraction, it not only intuitively demonstrates program
behavior but also incorporates hardware intrinsic informa-
tion. This can further cope with hardware intrinsic diversity
(meeting Req 2). The semantics of intrinsics are included as
sub-AST in the complete program AST. Therefore, we can
reuse feature extraction method, ensuring feature alignment
while avoiding extra design efforts. We extract features from
AST to cope with the diversity of execution environments.

The final but most crucial issue is how to design AST
features to meet Req 1. The target AST features should effec-
tively reflect the impact of schedule differences on program
performance. As we mentioned in the motivation, although
NLP methods based on schedule primitives have limitations in
generality, TLP has SOTA prediction accuracy. We can utilize
the attention matrix of TLP to mine the key schedules affecting
program performance, and then construct target AST features
based on these key schedules. Specifically, we transfer TLP
to tensorized programs, called TDTLP. The TDTLP model
includes an attention layer that takes schedule primitives as
token sequence input. Because the attention matrix can repre-
sent the degree of attention to tokens [18], we can leverage the
scheduling attention matrix to filter out key schedules affecting
performance. Since schedules generate different programs with
different performance by transforming AST, transformations
introduced by key schedules can be regarded as the target
AST features (meeting Req 1).

Fig. 4: Key ideas for unified learning strategy to transfer prediction.

Fig. 5: The organization structure of TensorizeSet.

TABLE II: Platforms and Intrinsics in TensorizeSet.

Platforms

NVIDIA A100 GPU

Xeon Platinum 8369B
ARM Yitian710

Intrinsics
(tensorcore.)m16n16k16.nn(.fp16),
m16n16k16.nt, m32n8k16.nn,
m32n8k16.nt, m8n32k16.nn
m8n32k16.nt [n-row, t-col]
avx512.dot(.int8), vnni.dot
neon.dot(.int8), sdot.dot

B. Learning Strategy Comparison in Transfer Prediction

Based on the unified feature representation, we next dis-
cuss which learning strategies to adopt
to address cross-
semantic and cross-performance transfer predictions. Cross-
performance represents hardware intrinsic upgrades, generally
focusing only on post-upgrade prediction. Hence,
transfer
learning is appropriate (meeting Req 5). However, cross-
semantic must simultaneously consider existing massive se-
mantics and potential new semantics. Multiple learning meth-
ods are compared in Fig. 4. Train ALL pours data from all
semantics into the model for training, and retraining after
expanding the new semantics leads to huge storage and
training costs. Multi-task learning has been used in TLP. The
premise is that a data sample needs to have labels on multiple
tasks. However, a tensorized program on one semantic cannot
run on other semantics to obtain latency labels, so multi-task
learning cannot be applied in cross-semantic scenarios. The
transfer learning adopted in TenSet suffers from catastrophic
forgetting [19], where the model forgets old semantics when
transferring to new semantics, leading to poorer flexibility.
As contrast, lifelong learning [20] as an incremental learning
method is particularly suitable for cross-semantic scenarios
(meeting Req 4). Models do not forget old semantics when
learning new ones, significantly reducing storage and training
costs and allowing flexible adaptation to new semantics.

IV. IMPLEMENTATION DETAILS

A. TensorizeSet: A Dataset for Tensorized Program Tuning

To meet the training and testing requirements of the cost
model for tensorized program tuning, we first construct Ten-
sorizeSet, a tensorized program dataset, which includes 10
hardware intrinsics from 3 hardware platforms and 84 common
neural network structures. As shown in Table II, the hardware
intrinsics consist of eight semantics (6 on GPU, 1 each on X86
and ARM in Fig. 1), 10 implementations (6 implementations

Authorized licensed use limited to: The University of Utah. Downloaded on October 05,2025 at 05:16:55 UTC from IEEE Xplore.  Restrictions apply.

490

ComparisonTrain CostStorageCostFlexibilityTrain AllHighDataset, highHighTransferLowParams, highLowMulti-Task———Life LongLowLowHighHigh PerfNew SemanticsCrossSemanticsTPCrossPerformanceTPReq4 √Req5 √Fig 4. Key ideas for selecting effective learning methods to achieve transfer prediction  Req5 √Req4 √Neural NetworkHardware IntrinsicGraph PartitionGraph PartitionSubgraph 0Subgraph 1Subgraph nSearch Space 0Search Space 1Search Space nMeasured Record 0Measured Record 1Measured Record n………Meta-ScheduleExecution
---

## Page 5

Fig. 6: Attention for tensorized programs on tensor core.m16n16k16.nn, arm.sdot, x86.avx512.

of schedule primitives embedding, other configurations remain
identical to TLP. After TDTLP convergence, we mine AST
features through its schedule attention matrix. Fig. 6 shows
the schedule attention matrix of TDTLP on three different
hardware intrinsics. The vertical axis represents query, while
the horizontal axis represents key. Deeper colors indicate
the higher importance of the primitive in affecting program
performance [18]. Despite differing platforms and intrinsics,
the key schedule primitives remain largely consistent. The
key schedules include Split, SamplePerfectTile, Sample-
Categorical, CacheWrite, ComputeInline. Since schedules
generate different programs with different performance by
transforming AST, transformations introduced by key sched-
ules can be regarded as the target AST features.

Combining Figure 7, we analyze how key schedules trans-
form the AST structure, and then determine the target AST fea-
tures. SamplePerfectTile and Split jointly determine tiling.
Tiling transforms the extent of loops in the AST, i.e., the
tiling size. After Split, the loop iteration transforms from (128,
128) to (2, 64, 128). Additionally, tiling also transforms the
affine of iterators [22]. As the access to the first dimension
of Buffer A transforms from i to i0 * 64 + i1, such affine
transformations are particularly critical for performance in
parallel scenarios. For example, 64 threads handle the data.
When the data is divided into 64 or 128 pieces, the data
each thread handles is different, which is reflected through the
thread affine. The integers sampled by SampleCategorical
are applied to vectorization, unroll, parallel, etc., which will
transform the schedules bound to loops in the AST. Bind
selects iteration binding threads according to the designation of
SampleCategorical. The remaining important schedule prim-
itives are related to program behavior, such as CacheWrite
and ComputeInline, which increase and decrease behavior,
respectively. The CacheWrite for Buffer B generates a new
behavior, which writes Buffer B to a new Buffer C, and the
description of program behavior is already covered in the
feature template. In summary, the target AST features should
include loops, loop-bound schedules, and affine.

C. Unified Feature Representation for Tensorized Program

Fig. 8 (a) and (b) respectively show the unified feature
representation and encoding method for tensorized programs.
The unified behavior template consists of a sequence of three

Fig. 7: AST transformations induced by key schedules.

with 6 semantics on GPU, 2 implementations with the same
semantic on X86 and ARM), and 2 data types (float16, int8).
In the dataset, tensorized programs on the same semantics
share the same schedules (but can also be different), while
those on different semantics have different schedules (and
cannot be the same). The network structures include CV
and NLP models with varying batch sizes and input shapes.
The organization structure of TensorizeSet is shown in Fig.
5. We run graph partitioning algorithms for each pair of
(network, hardware intrinsic) to obtain subgraphs, the smallest
execution units after operator fusion, with approximately 1000
subgraphs per hardware intrinsic. For each subgraph, we use
TVM MetaSchedule [15] to generate the search space and
sample up to 2000 programs, which are then actually run
to obtain measured records. Thus, the entire dataset contains
approximately 20 million records.

Compared to TenSet, the largest dataset currently used for
tensor program tuing, TensorizeSet has the same type of hard-
ware platforms, a comparable number of program samples,
and the same network structures. Furthermore, TensorizeSet
enriches the diversity of hardware intrinsic. Experimental
results show that TensorizeSet can meet the training and testing
requirements of cost models for tensorized program tuning. In
addition, program tuning is based on subgraphs or operators
[21] [7] as basic units. The network structure of TensorizeSet
includes common basic subgraphs and operators, so the cost
models trained and tested on this dataset can be generalized
to complex network built from basic subgraphs and operators.

B. AST Feature Mining with Attention

We transfer TLP to tensorized programs and train it on Ten-
sorizeSet, called TDTLP. Except for reasonable customization

Authorized licensed use limited to: The University of Utah. Downloaded on October 05,2025 at 05:16:55 UTC from IEEE Xplore.  Restrictions apply.

491

AnnotateBindBlockizeCacheReadCacheWriteComputeAtComputeInlineDecomposeReductionFuseGetBlockGetChildBlocksGetConsumersGetLoopsGetProducersPadEinsumReIndexReorderReverseComputeAtReverseComputeInlineSampleCategoricalSamplePerfectTileSplitStorageAlignTensorizeTransformBlockLayoutTransformLayoutUnannotateVectorizeAnnotateBlockizeCacheWriteComputeInlineDecomposeReductionFuseGetBlockGetChildBlocksGetConsumersGetLoopsParallelReorderReverseComputeAtSampleCategoricalSamplePerfectTileSplitTensorizeUnannotateVectorizeAnnotateBlockizeCacheWriteComputeInlineDecomposeReductionFuseGetBlockGetChildBlocksGetConsumersGetLoopsParallelReorderReverseComputeAtSampleCategoricalSamplePerfectTileSplitTensorizeUnannotateVectorize0.0000.0250.0500.0750.1000.1250.1500.000.010.020.030.040.050.060.000.020.040.060.080.10fori, j in(128, 128):B[i, j] = A[i, j]OriginalProgramTensorizationCandidatesfori, j, k in(8, 8, 8):Matmul(8, 32, 16)Tensorized Program Spacefori0, j0 in (?,?):fori1, j1, k0 in(?,?,?):read_optimization(A, B)fori2, j2, k1 in(?,?,?):fori3, j3 in (?,?):tensorcore.m8n32k16.nn.fp16write_optimization(C)// Scheduling on GPU// Apply Basic Schedules// “?”: optional Tiling// Program Transformation// Semantic Matching// One Tensorization CandidateSplit(i, factor=[2, 64])fori0, i1, j in(2, 64, 128):B[i, j] = A[i0*64+i1, j]SampleCategorical(x=64)Bind(j, threadIdx.x)fori0 in(2):fori1 in(64,threadIdx.x):fori0 in(128):B[i, j] = A[i0*64+i1, j]fori, j in(128, 128):B[i, j] = A[i, j]fori, j in(128, 128):C[i, j] = B[i, j]CacheWrite(Buffer=B)
---

## Page 6

Fig. 8: The unified feature representation and encoding method for tensorized programs.
(a) Unified Feature Representation, (b) Feature Encoding, (c) Example.

behaviors: load, compute, and store, which originates from
the simplified execution model of tensorized programs. Each
behavior is represented by three vectors, all of which start with
one hot vector indicating the type of behavior. The loopnest
vector represents all loops of the current behavior. Each four
dimensions describe a loop, which are loopid, extent, whether
parallel, and bound schedule. The affine vector consists of
loopid and five operations. Affine may appear in multiple
indices of multiple buffers. We encode from reading buffers
to writing buffers, from left to right. The loopnest and affine
vectors are derived from the schedule attention matrix mining.
The third vector is the buffer or intrinsic vector, depending
on whether the current behavior relies on hardware intrinsics.
The buffer vector consists of arithops and loop touched bytes,
used to describe the computation intensity of the innermost
assignment statement and the number of bytes accessed by
each loop per buffer. The Intrinsic vector consists of semantic
and implementation. The semantic sub-AST can be further
represented as loopnest and buffer.

As shown in Fig. 8 (c), AST features for the compute
behavior are extracted from a tensorized program. Further,
AST features are encoded, resulting in the final extracted
features. The displayed AST Features are extracted from the
compiler intermediate representation, satisfying the symbolic
derivation in Fig. 8 (a). Features Encoding demonstrates the
encoding process for the target AST Features. For the loop
vector (line 4 in a), taking loop n as an example (line 2 in AST
Features), since it is the second occurring loop, its loopid=2.
The number of iterations for this loop is 16, so extent=16.
This loop is bound to a thread and is parallel, i.e., isparallel=1
(line 4 in b). The loop is bound to the block.x schedule, i.e.,
schedule=6 (line 5 in b). So, loop n is encoded as [2, 16, 1,
6] (line 1 in Features Encoding). Other loops are encoded in
the same way. For the affine vector (line 5 in a), taking the first
dimension access of Buffer A as an example (line 5 in AST
Features), there are five affine operations (+, -, *, /, %) encoded
as numbers 1 to 5 (line 3 in b), so n/2*8 is encoded as [2, 4, 2,
3, 8] (line 4 in Features Encoding). Other affines are encoded
in the same way. For the hardware intrinsic vector (line 7 in
a), its semantics is also an AST, so its encoding method is
similar to the above, further represented as loop vectors and
buffer access vectors (line 8 in a). Extracted Features Final

Fig. 9: The model architecture of UniCoMo.

shows the three vectors obtained by the final encoding. Since
the current behavior is Compute, all three vectors start with
[0, 1, 0] (line 1 in Extracted Features Final). In summary, we
encode a compute behavior into three vectors: loop, affine, and
hardware intrinsic.

The representation is extensible, allowing users to extend
three feature categories: the implementation of hardware in-
trinsics (such as throughput, frequency, and power consump-
tion), more schedules in customized DLC, and additional
AST features about buffer access. While this work’s feature
the
extractor implementation is based on TVM TensorIR,
approach can be generalized to any DLC.

D. UniCoMo Model Architecture and Training Strategy

1) Model Architecture: The input of UniCoMo is the uni-
fied feature representation encoding for tensorized programs.
Its model structure is shown in Fig. 9, same as TDTLP.
Users can modify or deepen this model structure, but it is
essential to retain the attention layer to enable the model to
learn the relationships between program behaviors. UniCoMo
is trained and tested on the TensorizeSet. The label for each
program sample is label = min latency/latency, where
min latency is the minimum latency among all programs in
a subgraph. UniCoMo employs lambda rank loss [23] as the
training loss, which has been proven to be the most effective
loss function for such tasks.

2) Life Long Learning for Cross-Semantics: To overcome
catastrophic forgetting, lifelong learning enables models to
continuously learn new tasks while avoiding forgetting old
tasks. We adopt the method of selective synaptic plasticity
[20], which is a regularization-based approach. Its basic as-
sumption is that some parameters in the model are important
for old tasks, so only the unimportant parameters are changed
when training new tasks. We use it to address the cross-
semantic transfer prediction. Specifically, UniCoMo is trained

Authorized licensed use limited to: The University of Utah. Downloaded on October 05,2025 at 05:16:55 UTC from IEEE Xplore.  Restrictions apply.

492

1tensorized program::= (||)*2||::= loopnestaffine(buffer|intrin)3loopnest::= loop*4loop ::= loopidextent isparallelschedule5affine ::= loopid(+|-|*|/|% loopid)*6buffer ::= arithopsloop_touched_bytes...7intrin::= semantic implementation8semantic ::= loopnestbuffer9 implementation::= ...1||-> \onehot(vec(loopnest) vec(affine) vec(buffer|intrin))2 loopid-> 1, 2, 3, 4 ..., n3 +, -, *, /, % -> 1, 2, 3, 4, 54 isparallel-> 1(parallel)|2(serial)5 schedule -> \1(unroll)|2(vectorize)|3(parallel)|4(block.z)|5(block.y)|6(block.x)|7(thread.z)|8(thread.y)…6 arithops-> intadd, intmul, floatadd, floatmul1m in(2, block.y):2n in(16, block.x):3 # one warp for Matmul(8,128,64)4p, q in(8, 2):5   read1(A[n/2*8, p*16])6   read2(B[p*16, m*128+n%2*64+q*32])7   write1(C[n/2, m*2+n%2, 0, q])8m8n32k16.nn.fp16(C, A, B)9  # Semantics10i, j, k in(8, 32, 16):11  C[i, j] += A[i, k] * B[k, j]12 # implementation ...ASTFeaturesFeaturesEncoding1loop{m}[1,2,1,5], loop{n}[2,16,1,6]2loop{p}[3,8,0,0], loop{q}[4,2,0,0]3loopnest{m,n,p,q}[1,2,1,5,2,16,1,6,3,8,0,0,4,2,0,0]4affine{read1}[2,4,2,3,8,4,3,16]5affine{read2}[3,3,16,1,3,128,2,5,2,3,64,1,4,3,32]6affine{write1}[2,4,2,1,3,2,1,2,5,2,0,4]7// intrin{semantics}8loopnest{1,j,k}[1,8,0,0,2,32,0,0,3,16,0,0]9  arithops{semantics}[0,0,4096,4096]10buffer[0,0,4096,4096,512,64,2,256,32,32,1024,1024,32]11// intrin{implementation}...ExtractedFeaturesFinal1 onehot{load}[1,0,0], onehot{compute}[0,1,0], onehot{store}[0,0,1]2 onehot(vec(loopnest))3 [0,1,0 1,2,1,5,2,16,1,6,3,8,0,0,4,2,0,0]4 onehot(vec(affine))5 [0,1,0  2,4,2,3,8,4,3,16,3,3,16,1,3,128,2,5,2,3,64,1,4,3,32,2,4,2,1,3,2,1,2,5,2,0,4]6 onehot(vec(intrin))7 [0,1,0  1,8,0,0,2,32,0,0,3,16,0,0,0,0,4096,4096,512,64,2,256,32,32,1024,1024,32]abcSumLinear×4Linear×4LinearLinearAttention
---

## Page 7

TABLE III: Task Definition for Evaluation.

TaskId
Task0-2
Task3-4
Task5
Task6
Task7
Task8
Task9

Intrinsic Group
m8n32k16.nn; m8m32k16.nt; m32n8k16.nt;
avx512.dot; sdot.dot;
ALL Tensor Core (6 Semantics);
avx512.dot, neon.dot;
m16n16k16.nn, avx512.dot, sdot.dot;
vnni.dot, sdot.dot;
m16n16k16.nn, m16n16k16.nt, m32n8k16.nn;

on the first task using lambda rank loss, and subsequent new
semantics use the following loss function,

L′(θ) = L(θ) + λ

bi(θi − θb

i )2

(cid:88)

i

Where L(θ) is the loss function for the current task, i.e.,
lambda rank loss. bi represents the importance of parameters.
θi is the parameter to be learned. θb
i is the model parameter
converged on old tasks. For bi, we adopt EWC [24],

bi =

1
D

(cid:88)

d∈D

(

∂L(d, θ)
∂θ

)2

Where D is the dataset of old tasks. When the model parame-
ters are near the extremum for old tasks (θ near θb), the square
average of gradients is used as an approximation of the second
derivative as the basis for importance.

3) Transfer Learning for Cross-Performance: We employ
transfer learning with pre-training and fine-tuning to address
cross-performance transfer prediction. Specifically, the model
is pre-trained on low-performance implementations. Follow-
ing hardware intrinsics upgrade, their semantics remain un-
changed, but performance improves. We reinitialize the last
four fully connected layers of UniCoMo, keeping the remain-
ing model parameters unchanged. Subsequently,
the entire
network is fine-tuned on high-performance implementations
using a small dataset, leveraging model parameters to perceive
hardware implementation change.

V. EXPERIMENT

A. Experiment Setup

We conduct two types of evaluation: dataset-based evalua-
tion and search-based evaluation. The dataset-based metrics
evaluate the model’s accuracy on a static dataset. Search-
based metrics evaluate the model’s tuning efficiency and tuned
latency in a real tuning framework. Tuning efficiency refers
to the search time, and tuned latency refers to the inference

latency of the final program obtained through tuning. Both
types of evaluations are tested on ResNet-50, MobileNet-
V2, BERT-tiny, and BERT-base, with batch size 1 and image
size 224 (or sequence length 128) (test dataset and target
tuning network), which will not be used as training sets
for either dataset-based evaluation or search-based evaluation.
This experimental setup aligns with the current mainstream
cost model works [17] [16].

The dataset-based evaluation is conducted based on Ten-
sorizeSet. We establish 10 tasks as shown in the Table III. The
model is trained and tested on datasets containing hardware
intrinsics included in a task. We use the top-1 score and top-5
score as evaluation criteria (averaging scores when multiple
hardware intrinsics are included in a task), which have been
proven effective [17]. The expression of top-k is as follows,

top-k =

(cid:80)
(cid:80)

m

(cid:80)
(cid:80)

m

s min latencym,s × weightm,s
s min(latencym,s,i) × weightm,s

, 1 ≤ i ≤ k

where min latencym,s is the minimum latency among all
tensoried programs of subgraph s of model m, weightm,s is
the number of times the subgraph s appears in model m, and
latencym,s,i is the latency corresponding to the i-th largest
value of the output score. In simple terms, top-k represents the
ratio of the latency of the true optimal program to the latency
of the model-predicted optimal program. Therefore, the closer
this value is to 1, the stronger the cost model’s capability.

Dataset-based evaluation needs to validate UniCoMo’s fea-
ture representation and transfer prediction effectiveness. For
feature representation, we transfer TenSet and TLP to Ten-
sorizeSet as baselines, referred to as TDTenSet and TDTLP.
Around the three requirements that the feature representation
for tensorized programs should meet, we choose reasonable
task configurations to demonstrate that UniCoMo can cope
with hardware intrinsic diversity and platform diversity while
ensuring SOTA accuracy. For transfer prediction, we simulate
cross-semantic and cross-performance scenarios by reasonably
configuring task sequences, thereby proving the effectiveness
of UniCoMo’s lifelong learning and transfer learning in the
development of hardware intrinsics.

the current state-of-the-art

The search-based evaluation is based on TVM MetaSched-
ule,
tensorized program tuning
framework. We verify the tuning efficiency and tuned latency
of UniCoMo based on lifelong learning and transfer learning.
Existing work lacks cost models for tensorized programs.
Current cost models for normal
tensor programs, such as
Tenset and TLP, cannot meet the requirements of tensorized

TABLE IV: Effectiveness of Feature Representation.

TDTenSet

TDTLP

UniCoMo

Task0
Task1
Task2
Task3
Task4

Task5
Task6

Task7

Top-1 Score
0.8326
0.8602
0.9040
0.8713
0.8305

0.7910
0.7953

——

Top-5 Score
0.9210
0.9115
0.9173
0.9058
0.8982

0.8119
0.8552

——

Top-1 Score
0.9228
0.9255
0.9420
0.9236
0.8953

0.8290
0.8437

——

493

Top-5 Score
0.9746
0.9868
0.9709
0.9600
0.9345

0.8516
0.9027

——

Top-1 Score
0.9276
0.9206
0.9418
0.9172
0.8851

0.9040
0.9073

0.8685

Top-5 Score
0.9793
0.9821
0.9764
0.9632
0.9416

0.9757
0.9525

0.9180

Authorized licensed use limited to: The University of Utah. Downloaded on October 05,2025 at 05:16:55 UTC from IEEE Xplore.  Restrictions apply.
---

## Page 8

Test On

Rand Init.

Task9
Task0
Task1
Task2
Task4

Test on

Rand Init.

Task9
Task0
Task1
Task2
Task4

g
n
i
n
i
a
r
T

r
e
t
f

A

g
n
i
n
i
a
r
T

r
e
t
f

A

Top-1
0.1285
0.9154
0.7797
0.7301
0.6599
0.4568

Top-1
0.1285
0.9154
0.9135
0.9051
0.9042
0.8802

TABLE V: Cross Semantics without Lifelong Learning.

Task9

Task0

Task1

Task2

Task4

Top-5
0.2745
0.9765
0.8965
0.8126
0.7925
0.6365

Top-1
0.1700
0.8355
0.9245
0.6551
0.6471
0.5656

Top-5
0.2094
0.9024
0.9722
0.8544
0.8217
0.6070

Top-1
0.1594
0.8107
0.7012
0.9252
0.8116
0.3146

Top-5
0.2447
0.8977
0.8414
0.9878
0.8835
0.6315

Top-1
0.1427
0.7944
0.7400
0.7391
0.9399
0.4953

Top-5
0.2447
0.8910
0.8124
0.8586
0.9764
0.7346

Top-1
0.1745
0.6306
0.4112
0.4246
0.5725
0.8842

TABLE VI: Cross Semantics with Lifelong Learning.

Task9

Task0

Task1

Task2

Task4

Top-5
0.2745
0.9765
0.9686
0.9641
0.9455
0.9246

Top-1
0.1700
0.8355
0.9203
0.9184
0.9021
0.8954

Top-5
0.2094
0.9024
0.9664
0.9733
0.9556
0.9406

Top-1
0.1594
0.8107
0.7997
0.9204
0.9147
0.8802

Top-5
0.2447
0.8977
0.8967
0.9739
0.9672
0.9566

Top-1
0.1427
0.7944
0.7850
0.8859
0.9217
0.9005

Top-5
0.2447
0.8910
0.9015
0.9046
0.9760
0.9644

Top-1
0.1745
0.6306
0.6053
0.6462
0.5048
0.8561

Top5
0.2651
0.7923
0.6845
0.7259
0.7188
0.9430

Top5
0.2651
0.7923
0.7802
0.7626
0.7939
0.9350

TABLE VII: Cross Performance with Transfer Learning.

Task6
Task8(transfer with half)
Task8(direct with all)

Top-1 Score
0.9073
0.8964
0.9172

Top-5 Score
0.9525
0.9537
0.9496

program tuning. As mentioned in motivation, for feature
representation, they cannot simultaneously consider accuracy,
hardware intrinsic and platform diversity, leading to differ-
entiated feature representations for specific hardware, which
cannot cope with diverse runtime environments. For transfer
prediction, they lack suitable learning strategies, leading to
train models one by one for tensorized programs on specific
hardware, which cannot cope with the rapid development of
customized hardware. These deficiencies are demonstrated in
Table IV and the related explanations. Therefore, we adopt the
default online model as the baseline.

B. Dataset-based Evaluation

1) Effectiveness of Feature Representation: Table IV shows
the accuracy of TDTenSet, TDTLP, and UniCoMo on mul-
tiple tasks. Tasks 0 to 4 are all single hardware intrinsic
tasks. UniCoMo outperforms TDTenSet, indicating that AST
features mined through the schedule attention matrix can
more effectively reflect the impact of schedule differences
on program performance than expert features. UniCoMo is
this indicates the
on par with TDTLP. On the one hand,
interpretability of using scheduling primitives as an NLP task.
On the other hand, UniCoMo can further explore generality
while maintaining SOTA accuracy. Tasks 5 and 6 involve
multiple hardware intrinsics from the same or similar hardware
platforms. Due to the inclusion of hardware intrinsic abstrac-
tions, UniCoMo has a significant advantage over TDTenSet
and TDTLP, indicating that UniCoMo significantly enhances
the ability to cope with hardware intrinsic diversity. Task 7
involves different hardware intrinsics from different hardware
platforms. TDTLP and TDTenSet cannot work across CPUs
and GPUs. Although UniCoMo has not achieved high accuracy

in this task, it does have the ability to cope with the hardware
platform diversity, which opens up possibilities in unified
cost model for tensorized program tuning on heterogeneous
computing cluster in the future.

2) Effectiveness of Transfer Prediction: We simulate cross-
semantic transfer prediction scenarios by sequentially learning
tasks 9, 0, 1, 2, and 4. We compare the effectiveness of
lifelong learning using Table V and Table VI. The vertical
direction in the table represents the order of task learning,
and the horizontal direction represents the testing accuracy.
Any task pair in the table is denoted as (taski, taskj).
i < j indicates the model’s zero-shot transfer ability to taskj,
while i > j indicates the degree of forgetting for taskj.
Table V shows the case without lifelong learning. Due to
effective feature representation, UniCoMo exhibits some zero-
shot
transfer ability to unseen tasks but shows significant
forgetting for old tasks. Table VI shows the case of lifelong
learning. UniCoMo ensures not to forget old semantics while
continuously learning new semantics. Overcoming forgetting
enhances its zero-shot transfer ability to some extent. It is
evident that through lifelong learning, UniCoMo can flexibly
expand new semantics at a low cost, thereby coping with
cross-semantic transfer prediction. As shown in Table VII, We
simulate cross-performance transfer prediction scenarios using
task 6 and task 8. We first trained UniCoMo on Task6 (the
first row in Table 7), and then used half of the data from Task8
for transfer learning (the second row in Table 7). Furthermore,
we directly training UniCoMo using all the data from Task8
(the third row in Table 7) as a control experiment. It is evident
that through transfer learning, UniCoMo can flexibly transfer
to high-performance implementations at a low cost, thereby
coping with cross-performance transfer prediction.

C. Search-based Evaluation

We conducted search-based evaluation (tensorized program
tuning) on TVM MetaSchedule. The hardware platforms and
hardware intrinsics are aligned with TensorizeSet. For the GPU

Authorized licensed use limited to: The University of Utah. Downloaded on October 05,2025 at 05:16:55 UTC from IEEE Xplore.  Restrictions apply.

494
---

## Page 9

Fig. 10: Tuning curves of UniCoMo and Online on tensor core and vnni.dot.

by the USTC Research Funds of the Double First-Class
Initiative under Grant YD2150002005 and YD2150002011.

REFERENCES

[1] H. Touvron et al., “Llama: Open and efficient foundation language

models,” arXiv preprint arXiv:2302.13971, 2023.

[2] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei et al., “Language
models are unsupervised multitask learners,” OpenAI blog, 2019.
[3] Nvidia. nvidia tensor cores. [Online]. Available: \protect\@normalcr\

relaxwww.nvidia.com/en-us/data-center/tensor-cores/

[4] Intel,

intrinsics. [Online]. Available: \protect\@normalcr\relaxwww.

intel.com/content/www/us/en/docs/intrinsics-guide/index.html

[5] Arm,

intrinsics.

[Online].

Available:

\protect\@normalcr\

relaxdeveloper.arm.com/architectures/instruction-sets/intrinsics/

[6] W. Lou et al., “Octcnn: A high throughput fpga accelerator for cnns
using octave convolution algorithm,” IEEE Trans on Computers, 2021.
[7] T. Chen et al., “Tvm: An automated end-to-end optimizing compiler for

deep learning,” in Proc. of OSDI, 2018.

[8] J. Weng, A. Jain, J. Wang, L. Wang, Y. Wang, T. Nowatzki et al., “Unit:

Unifying tensorized instruction compilation,” in Proc. of CGO, 2021.

[9] J. Zhao et al., “Akg: automatic kernel generation for neural processing
units using polyhedral transformations,” in Proc. of PLDI, 2021.
[10] S. Zheng, R. Chen, A. Wei, Y. Jin, Q. Han, L. Lu, B. Wu et al.,
“Amos: enabling automatic mapping for tensor computations on spatial
accelerators with hardware abstraction,” in Proc. of ISCA, 2022.
[11] S. Feng, B. Hou, H. Jin, W. Lin et al., “Tensorir: An abstraction for
automatic tensorized program optimization,” in Proc. of ASPLOS, 2023.
[12] T. Chen, L. Zheng, E. Yan, Z. Jiang, T. Moreau, L. Ceze, C. Guestrin
et al., “Learning to optimize tensor programs,” in Proc. of NIPS, 2018.
[13] L. Zheng et al., “Ansor: Generating high-performance tensor programs

for deep learning,” in Proc. of OSDI, 2020.

[14] Y. Chen, Y. Xie, L. Song, F. Chen, and T. Tang, “A survey of accelerator

architectures for deep neural networks,” Engineering, 2020.

[15] J. Shao, X. Zhou, S. Feng, B. Hou, R. Lai, H. Jin et al., “Tensor program
optimization with probabilistic programs,” in Proc. of NIPS, 2022.
[16] Y. Zhai, Y. Zhang, S. Liu, X. Chu, J. Peng et al., “Tlp: A deep learning-
based cost model for tensor program tuning,” in Proc. of ASPLOS, 2023.
[17] L. Zheng, R. Liu et al., “Tenset: A large-scale program performance

dataset for learned tensor compilers,” in Proc. of NIPS, 2021.

[18] A. Vaswani et al., “Attention is all you need,” in Proc. of NIPS, 2017.
[19] J. Kirkpatrick et al., “Overcoming catastrophic forgetting in neural

networks,” in Proceedings of the national academy of sciences, 2017.

[20] L. Wang et al., “A comprehensive survey of continual learning: Theory,
method and application,” IEEE Trans on Pattern Analysis and Machine
Intelligence, 2024.

[21] J. Roesch et al., “Relay: A high-level compiler for deep learning,” arXiv

preprint arXiv:1904.08368, 2019.

[22] C. Lattner et al., “Mlir: Scaling compiler infrastructure for domain

specific computation,” in Proc. of CGO, 2021.

[23] X. Wang et al., “The lambdaloss framework for ranking metric opti-
mization,” in Proceedings of the 27th ACM international conference on
information and knowledge management, 2018.

[24] J. Kirkpatrick et al., “Overcoming catastrophic forgetting in neural

networks,” in Proceedings of the national academy of sciences, 2017.

Fig. 11: The search time required for UniCoMo to reach the latency
of Online tuning 2000 times.

platform, each network is tensorized to 6 semantics of tensor
cores, meaning there are six different tensorization candidates
in the search space for each subgraph. For the CPU platform,
each network is tensorized to vnni.dot. Subgraphs that cannot
be tensorized by MetaSchedule are ignored. After 2000 rounds
of tuning (global=2000, per iter=10), the tuning efficiency and
tuned latency of UniCoMo (life-long learning with tasks 9,
0, 1, 2 in sequence and transfer learning from task 6 to 8)
are compared with the default online cost model. The tuning
curves for all workloads are shown in Figure 10. These curves
indicate that UniCoMo can converge to lower latency faster.
UniCoMo can improve the inference speed (tuned latency) by
an average of 1.9× compared to the Online. Figure 11 shows
the search time required for UniCoMo to achieve the latency of
Online tuning 2,000 times. UniCoMo can speed up the search
time (tuning efficiency) by an average of 9.8×.

VI. CONCLUSION AND FUTURE WORK

This work proposes a unified cost model for tensorizd
program tuning. For future work, based on our approach,
the direct prediction of the most suitable hardware intrinsic
and optimal task scheduling for workloads in heterogeneous
computing clusters can be explored.

VII. ACKNOWLEDGEMENTS

This work was supported in part by the National Key
R&D Program of China under Grants 2022YFB4501600 and
2022YFB4501603, in part by the National Natural Science
Foundation of China under Grants 62102383, 61976200, and
62172380, in part by Jiangsu Provincial Natural Science Foun-
dation under Grant BK20210123, in part by Youth Innovation
Promotion Association CAS under Grant Y2021121”, in part

Authorized licensed use limited to: The University of Utah. Downloaded on October 05,2025 at 05:16:55 UTC from IEEE Xplore.  Restrictions apply.

495

0500010000Search Time(s)100020003000Latency(us)BERT-base(Tensor Core)0250050007500Search Time(s)5001000BERT-tiny(Tensor Core)0500010000Search Time(s)5001000ResNet-50(Tensor Core)0500010000Search Time(s)100200MobileNet-V2(Tensor Core)0250050007500Search Time(s)255075Latency(ms)BERT-base(X86 VNNI)0200040006000Search Time(s)51015BERT-tiny(X86 VNNI)0250050007500Search Time(s)1020ResNet-50(X86 VNNI)0250050007500Search Time(s)12MobileNet-V2(X86 VNNI)OnlineUniCoMoBERT-base(Tensor Core)BERT-tiny(Tensor Core)ResNet-50(Tensor Core)MobileNet-V2(Tensor Core)BERT-base(X86 VNNI)BERT-tiny(X86 VNNI)ResNet-50(X86 VNNI)MobileNet-V2(X86 VNNI)02500500075001000012500Time (s)OnlineUniCoMo
---

