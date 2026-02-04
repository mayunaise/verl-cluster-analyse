Performance Tuning Guide on Ascend
==================================

Last updated:  01/29/2026.

Author:  `Xiaobo Hu <https://github.com/tardis-key>`_, `Haozhe Li <https://github.com/ZLiao097>`_

`Perf Tuning <https://github.com/verl-project/verl/blob/main/docs/perf/perf_tuning.rst>`_ 中介绍的性能调优方法在昇腾设备中同样适用。本文重点介绍**昇腾特有**的调优手段，包括**融合算子优化**、**特定硬件配置**和**昇腾亲和特性**等。

----

融合算子
--------------------------

常用融合算子列表
**********************************

融合算子的优化原理为：通过数学意义上的等价替换，将多个算子融为一个算子的计算，减少冗余计算，同时减少下发次数，从而提高性能。几个典型的 NPU 融合算子列举如下，目前均已在 ``npu_patch.py`` 中对 Qwen2、Qwen3 系列模型完成替换。

当前 verl 中使用的全量融合算子请查阅：
`npu_patch.py <https://github.com/verl-project/verl/blob/main/verl/models/transformers/npu_patch.py>`_

Matrix Computation-Communication operator fusion (MC2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MC2 是 CANN 中一系列**计算通信融合算子**的统称，这些算子将原本串行的通信和计算操作融合在一起，通过内部的切分和流水线并行执行来优化性能。

在 vllm-ascend 中，可以通过指定环境变量：

.. code-block:: sh

    export VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=1

在前向计算的 ``RowParallelLinear`` 中使能 ``torch_npu.npu_mm_all_reduce_base``，将分离的 ``matmul`` 和 ``allreduce`` 合并为一个融合算子。

`RotaryMul&RotaryMulGrad <https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/performance_tuning_0030.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**torch_npu 接口**：
``torch_npu.npu_rotary_mul(x, r1, r2)``

**参数说明**：

- ``x``: q, k，shape 要求输入为 4 维，一般为 ``[B, N, S, D]`` / ``[B, S, N, D]`` / ``[S, B, N, D]``
- ``r1``: cos 值，shape 要求输入为 4 维，一般为 ``[1, 1, S, D]`` / ``[1, S, 1, D]`` / ``[S, 1, 1, D]``
- ``r2``: sin 值，shape 要求输入为 4 维，一般为 ``[1, 1, S, D]`` / ``[1, S, 1, D]`` / ``[S, 1, 1, D]``

`RmsNorm&RmsNormGrad <https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/performance_tuning_0031.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**torch_npu 接口**：
``torch_npu.npu_rms_norm(self, gamma, epsilon=1e-06) -> (Tensor, Tensor)``

**参数说明**：

- ``self``: Tensor 类型，shape 支持 1–8 维
- ``gamma``: Tensor 类型，通常为 weight，shape 要求与 ``self`` 的后几维保持一致
- ``epsilon``: Float 类型，用于防止除 0 错误

**输出说明**：

- 第 1 个输出：最终计算结果 ``y``
- 第 2 个输出：中间结果 ``rstd``，用于反向计算

`Swiglu <https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/performance_tuning_0035.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**torch_npu 接口**：
``torch_npu.npu_swiglu(Tensor self, int dim=-1) -> Tensor``

**参数说明**：

- ``self``: Tensor 类型，shape 支持 1–8 维
- ``dim``: Int 类型，默认为 ``-1``

**输出说明**：

- 输出为最终计算结果 ``y``

`GroupMatMul <https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu_grouped_matmul.md>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**函数原型**：

.. code-block:: python

    npu_grouped_matmul(
        x,
        weight,
        *,
        bias=None,
        scale=None,
        offset=None,
        antiquant_scale=None,
        antiquant_offset=None,
        per_token_scale=None,
        group_list=None,
        activation_input=None,
        activation_quant_scale=None,
        activation_quant_offset=None,
        split_item=0,
        group_type=None,
        group_list_type=0,
        act_type=0,
        output_dtype=None,
        tuning_config=None
    ) -> List[Tensor]

详细使用方法见文档链接。

----

FSDP 后端融合算子使用方法
**********************************

在 ``verl/models/transformers/npu_patch.py`` 中，已通过 patch 形式自动替换可用融合算子，**无需额外操作，默认启用**。

Megatron 后端融合算子使用方法
**********************************

Megatron 的融合算子集成在 MindSpeed 中，需要添加特定参数开启：

1. **Flash Attention（必须开启）**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
       ++actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True

2. **RotaryMul**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
       +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True

3. **RMSNorm**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rmsnorm=True

4. **GroupMatMul**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True

5. **Swiglu**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True

6. **Permute/Unpermute**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.fused_permute_unpermute=True

7. **MC2**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.use_ascend_mc2=True

----

昇腾通用配置
--------------------------

`算子下发 <https://www.hiascend.com/document/detail/zh/Pytorch/730/comref/Envvariables/docs/zh/environment_variable_reference/TASK_QUEUE_ENABLE.md>`_
************************************************************************************************************************************************************************

通过 ``TASK_QUEUE_ENABLE`` 可配置 task_queue 算子下发队列优化等级，**默认为 Level 1**。

.. image:: https://github.com/verl-project/verl-data/blob/main/images/ascend/perf_tuning_task_queue.png
    :width: 500px
    :align: center

- **Level 0**：不开启下发流水优化
- **Level 1**：将算子下发分为两段，部分任务（如 aclnn 调用）放在二级流水并行执行，掩盖下发耗时
- **Level 2**：在 Level 1 基础上进一步均衡一、二级流水负载，将 workspace 相关任务迁移至二级流水，**性能收益更大**。该配置仅在二进制场景生效，**推荐设为 Level 2**。

`通讯算法编排展开 <https://www.hiascend.com/document/detail/zh/canncommercial/850/maintenref/envvar/envref_07_0096.html>`_
************************************************************************************************************************************************************************

使用环境变量 ``HCCL_OP_EXPANSION_MODE=AIV`` 配置通信算法的编排展开位置，支持取值：

- **AI_CPU**：通信算法编排展开在 Device 侧 AI_CPU
- **AIV**：通信算法编排展开在 Device 侧 Vector Core（推荐）
- **HOST**：通信算法编排展开在 Host 侧 CPU
- **HOST_TS**：Host 侧 CPU 编排，向 Device Task Scheduler 下发并调度执行

----

推理阶段调优
--------------------------

Chunked Prefill in V1
***************************

VLLM 当前版本已默认启用 VLLM V1，使用以下配置启用 Chunked Prefill：

.. code-block:: sh

    actor_rollout_ref.rollout.enable_chunked_prefill=True

原理参考 `VLLM 官方文档 <https://docs.vllm.ai/en/v0.4.2/models/performance.html>`_。

Graph Mode
***************************

与 CUDA 类似，NPU 通过以下配置启用 **ACL Graph**：

.. code-block:: sh

    actor_rollout_ref.rollout.enforce_eager=False

文档：`ACL Graph <https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/feature_guide/ACL_Graph.html>`_

.. note::
    ACL Graph 与 ``taskqueue Level 2`` 原理冲突，**二者无法同时开启**。

----

训练阶段调优
--------------------------

FSDP
***************************

.. csv-table:: FSDP 切分策略说明
   :header: "FSDP 模式", "说明"
   :widths: 30, 70

   "/",                  "仅切分优化器（Zero-1）"
   "SHARD_GRAD_OP",      "切分梯度和优化器（Zero-2）"
   "HYBRID_SHARD",       "切分权重、梯度和优化器（Zero-3）"
   "2D device_mesh+HYBRID_SHARD", "HSDP（FSDP+DDP），如 device_mesh=[2,8]"
   "2D device_mesh+HYBRID_SHARD_ZERO2", "HSDP 的 Zero-2 版本"
   "NO_SHARD",           "纯 DDP"

**说明**：

- FSDP 不支持 Zero-1，VeRL 会根据卡数和 ``actor_rollout_ref.actor.fsdp_config.fsdp_size`` 自动决定 device mesh，**默认使用 Zero-3**
- 模型较小时（建议 < 7B），可将 ``reshard_after_forward=True``，在 FSDP/FSDP2 上使用 Zero-2 优化性能

Megatron
***************************

模型较大时，使用 Megatron 作为训练后端可更灵活地进行性能调优。

通用并行策略建议：

- DP 显存不足时：优先开启 **TP** 切分模型权重
- 模型仍过大：再开启 **PP** 进一步切分
- 序列过长、激活过大：开启 **CP / SP**
- MoE 模型：开启 **EP** 切分专家；专家过小时开启 **ETP** 避免 MoE 部分 TP 切分过碎

**SP / CP 开启方式**：

- **SP (Sequence Parallel)**
  ::

      actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True

- **CP (Context Parallel)**（两个参数必须同时配置）
  ::

      actor_rollout_ref.actor.megatron.context_parallel_size
      actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size

Megatron-distributed optimizer
**********************************

大模型场景下，为节省显存，通常需要将优化器分片到 DP 域内每张卡。

NPU + Megatron 后端开启分布式优化器：

::

    +actor_rollout_ref.actor.megatron.override_transformer_config.use_distributed_optimizer=True
