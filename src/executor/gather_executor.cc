#include <mxnet/base.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/graph.h>
#include <nnvm/pass_functions.h>
#include <nnvm/graph_attr_types.h>
#include <mshadow/base.h>
#include <vector>
#include <algorithm>
#include <cstring>

#include <mxnet/perf-inl.h>

#include "./exec_pass.h"
#include "./gather_executor.h"
#include "../engine/profiler.h"
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "../operator/mkl/mkl_memory-inl.h"
#include "../operator/mkl/mkl_util-inl.h"
#endif

namespace mxnet { 
namespace exec { 

using mshadow::kInt32;
using mshadow::kInt64;

/*
 * Just rename FComputeExecutor to GatherOpExecutor
 */
class GatherOpExecutor : public OpExecutor {
 public:
  void Run(RunContext rctx) override {
    op_ctx.run_ctx = rctx;
    fcompute_(attrs_, op_ctx, in_data_, req, out_data_);
#if MKL_EXPERIMENTAL == 1
    mkl_tblobs_prv_to_cpu(in_data_);
    mkl_tblobs_prv_to_cpu(out_data_);
#endif
  }

  void SetInputTBlob(size_t idx, TBlob& blob) {
    //VLOG(1) << "GatherOpExecutor SetInputTBlob " << idx << " / " << in_data_.size();
    in_data_[idx] = blob;
  }

  void SetOutputTBlob(size_t idx, TBlob& blob) {
    //VLOG(1) << "GatherOpExecutor SetOutputTBlob " << idx << " / " << out_data_.size();
    out_data_[idx] = blob;
  }

  void SetAttrScalar(std::vector<double>& s) { 
    attrs_.scalars = s;
  }

  void Setup() override {
    in_data_.resize(in_array.size());
    out_data_.resize(out_array.size());
    auto get_blob =  [](const NDArray& nd) {
      return nd.data();
    };
    std::transform(in_array.begin(), in_array.end(), in_data_.begin(), get_blob);
    std::transform(out_array.begin(), out_array.end(), out_data_.begin(), get_blob);
  }

  Operator::ExecType exec_type() const override {
    return Operator::kSync;
  }

  explicit GatherOpExecutor(FCompute fcompute, const NodeAttrs& attrs)
      : fcompute_(fcompute), attrs_(attrs) {}

 private:
  FCompute fcompute_;
  NodeAttrs attrs_;
  std::vector<TBlob> in_data_, out_data_;
};


/*
 * TODO : Add callback to Executor to notify that the block execution is finished.
 */

GatherExecutor::~GatherExecutor() {
  if (op_node_.cached_opr != nullptr) {
    Engine::Get()->DeleteOperator(op_node_.cached_opr);
  }
}

void GatherExecutor::Init(Context& ctx, size_t max_gather) {
  ctx_ = ctx;
  max_gather_ = max_gather;
  idx_dev_ = NDArray(TShape({max_gather_}), ctx_, false,  kInt32);
  addr_dev_ = NDArray(TShape({max_gather_}), ctx_, false, kInt64);

  idx_host_ = NDArray(TShape({max_gather_}), Context::CPUPinned(0), false,  kInt32); // pinned 
  addr_host_ = NDArray(TShape({max_gather_}), Context::CPUPinned(0), false, kInt64); // pinned

  std::string op_name = "multi_gather";
  op_ = nnvm::Op::Get(op_name);

  // init OpExecutor
  FCompute fcompute;
  NodeAttrs attrs;

  if (ctx_.dev_mask() == cpu::kDevMask) {
    fcompute = nnvm::Op::GetAttr<FCompute>("FCompute<cpu>").get(op_, nullptr);
  } else if (ctx_.dev_mask() == gpu::kDevMask) {
    fcompute = nnvm::Op::GetAttr<FCompute>("FCompute<gpu>").get(op_, nullptr);
  } else { 
    LOG(FATAL) << "Unknown device mask";
  }
  attrs.op = op_;
  attrs.name = op_name;
  op_exec_ = std::make_shared<GatherOpExecutor>(fcompute, attrs);

  // init cached OpNode
  char *p_opr_name = new char[op_name.size() + 1];
  memcpy(p_opr_name, op_name.c_str(), op_name.size() + 1);
  op_node_.opr_name = p_opr_name;
  op_node_.ctx = ctx_;
  op_node_.exec = op_exec_;
  // cached_opr, use_vars, mutate_vars will be set when bind new input and output
}

void GatherExecutor::SetHostAddrAndIdx(std::vector<NDArray>& inputs, std::vector<int>& idxes) {
  int *h_idx = idx_host_.data().dptr<int>();
  std::memcpy(h_idx, idxes.data(), sizeof(int) * idxes.size());
  int64_t *h_addr = addr_host_.data().dptr<int64_t>();
  for (size_t i = 0;i < inputs.size();i ++) { 
    h_addr[i] = reinterpret_cast<int64_t>(inputs[i].data().dptr_);
  }
}



void GatherExecutor::Forward(std::vector<NDArray>& inputs, std::vector<int>& idxes, NDArray& output){ 
  uint64_t t0, t1, t2, t3, t4, t5;

  CHECK_EQ(inputs.size(), idxes.size());
  CHECK_LE(inputs.size(), max_gather_);

  //t0 = get_time();
  auto& ishape = inputs[0].shape();
  int M = ishape.Size() / ishape[0];
  int K = inputs.size();
  // 0. Set Node Attrs with M and K , 
  // M = scalars[0]   
  // K = scalars[1]
  std::vector<double> s {(double)(M), (double)(K)};
  op_exec_->SetAttrScalar(s);

  SetHostAddrAndIdx(inputs, idxes);

  // 1. copy idx and addr to device TODO if async, need callback to release memory
  //t1 = get_time();
  //CopyFromTo(idx_host_, &idx_dev_);
  //CopyFromToAsync(idx_host_, &idx_dev_);

  //t2 = get_time();
  //CopyFromTo(addr_host_, &addr_dev_);
  //CopyFromToAsync(addr_host_, &addr_dev_);

  //t3 = get_time();

  // 3. revoke gather kernel
  // setup exec, input : [idx_, addr_] output : output
  op_exec_->in_array.resize(2); 
  op_exec_->out_array.resize(1);
  op_exec_->in_array[0] = idx_dev_;
  op_exec_->in_array[1] = addr_dev_;
  op_exec_->out_array[0] = output;
  op_exec_->Setup();

  // setup exec vars
  std::vector<Engine::VarHandle> use_vars, mutate_vars;
  for(auto& nd : inputs) { use_vars.push_back(nd.var());} 
  use_vars.push_back(idx_dev_.var());
  use_vars.push_back(addr_dev_.var());
  mutate_vars.push_back(output.var());
  auto & op_exec = op_exec_;
  bool is_async = op_exec_->exec_type() == Operator::kAsync;
  bool is_gpu = ctx_.dev_mask() == gpu::kDevMask;
  auto exec_fun = [op_exec, is_async, is_gpu]( 
    RunContext ctx, Engine::CallbackOnComplete on_complete) {
    if (is_async) {
      op_exec->op_ctx.async_on_complete = on_complete;
    }
    //op_exec->Run(ctx); // (pin)
    // call on complete only if it is async op
    if (!is_async) {
      if (is_gpu) {
      #if MXNET_USE_CUDA
        // Wait GPU Kernel to Finish.
        //ctx.get_stream<gpu>()->Wait(); // (pin) do not wait
      #else
        LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
      #endif
      }
      on_complete(); // TODO Do not fully understand here
    }
  };

  op_node_.cached_opr = Engine::Get()->NewOperator(
    exec_fun, use_vars, mutate_vars, FnProperty::kNormal,
    PROFILER_MESSAGE(op_node_.opr_name));
  // no need to setup op_node_.mutate_vars and use_vars

  //t4 = get_time();

  // call engine  to launch kernel
#if MXNET_USE_PROFILER
  bool profiling = engine::Profiler::Get()->GetState() == engine::Profiler::kRunning;
#else
  bool profiling = false;
#endif
  Engine::Get()->Push(op_node_.cached_opr, op_node_.ctx, 0, profiling);
  //t5 = get_time();

  //LOG(INFO) << " num gather " << idxes.size()
  //          << " prepare time " << t1 - t0
  //          << " idx time " << t2 - t1
  //          << " addr time " << t3 - t2
  //          << " setup time " << t4 - t3
  //          << " push time " << t5 - t4;
}

} // namespace exec
} // namespace mxnet
