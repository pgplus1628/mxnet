#pragma once 

#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/executor.h>
#include <nnvm/graph.h>
#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "./exec_pass.h"

/*
 * Executor only works for MultiGather, 
 * Since it takes variable input NDArray, and we want it to be very efficient, 
 * So, we customized the GatherExecutor.
 */
namespace mxnet { 
using NodeOperatorMap = std::unordered_map<const nnvm::Node*,
    std::shared_ptr<Operator>>;

// forward declaration
namespace exec {
class GatherExecutor;
}


namespace exec { 
using nnvm::Graph;

//gather executors
class GatherExecutor { 
public : 
  virtual ~GatherExecutor();
  void Forward(std::vector<NDArray>& inputs, std::vector<int>& idxes, NDArray& output);

 // initialized the executor
  void Init(Context& default_ctx);

protected : 
  // information about operational node
  struct OpNode { 
    // The name of the operator
    const char* opr_name;
    // the context of the node
    Context ctx;
    // The executor
    std::shared_ptr<OpExecutor> exec;
    // skip the execution of this node
    bool skip_exec_node{false};
    // cached operator handle
    Engine::OprHandle cached_opr{nullptr};
    // cached const vars, used for seg ops creation
    std::vector<Engine::VarHandle> use_vars;
    // cached mutate vars, used for seg ops creation
    std::vector<Engine::VarHandle> mutate_vars;
  };

  // operator node
  OpNode op_node_;
  const Op * op_{nullptr}; // the gather op
  std::shared_ptr<OpExecutor> op_exec_;  // attach_op_exec_pass
  Context ctx_;

  NDArray idx_; // index array on device 
  NDArray addr_; // address array on device


}; //class GatherExecutor



} // namespace exec

} // namespace mxnet


