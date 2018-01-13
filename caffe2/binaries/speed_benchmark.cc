/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string>

#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/utils/string_utils.h"
#ifdef WITH_CUDA
#include <caffe2/core/context_gpu.h>
#endif

CAFFE2_DEFINE_string(net, "", "The given net to benchmark.");
CAFFE2_DEFINE_string(init_net, "",
    "The given net to initialize any parameters.");
CAFFE2_DEFINE_string(input, "",
    "Input that is needed for running the network. If "
    "multiple input needed, use comma separated string.");
CAFFE2_DEFINE_string(input_file, "",
    "Input file that contain the serialized protobuf for "
    "the input blobs. If multiple input needed, use comma "
    "separated string. Must have the same number of items "
    "as input does.");
CAFFE2_DEFINE_string(input_dims, "1,720,1080,4",
    "Alternate to input_files, if all inputs are simple "
    "float TensorCPUs, specify the dimension using comma "
    "separated numbers. If multiple input needed, use "
    "semicolon to separate the dimension of different "
    "tensors.");
CAFFE2_DEFINE_string(output, "",
    "Output that should be dumped after the execution "
    "finishes. If multiple outputs are needed, use comma "
    "separated string. If you want to dump everything, pass "
    "'*' as the output value.");
CAFFE2_DEFINE_string(output_folder, "",
    "The folder that the output should be written to. This "
    "folder must already exist in the file system.");
CAFFE2_DEFINE_int(warmup, 0, "The number of iterations to warm up.");
CAFFE2_DEFINE_int(iter, 10, "The number of iterations to run.");
CAFFE2_DEFINE_bool(run_individual, false,
    "Whether to benchmark individual operators.");

CAFFE2_DEFINE_bool(force_engine, false, "Force engine field for all operators");
CAFFE2_DEFINE_string(engine, "", "Forced engine field value");
CAFFE2_DEFINE_bool(force_algo, false, "Force algo arg for all operators");
CAFFE2_DEFINE_string(algo, "", "Forced algo arg value");

CAFFE2_DEFINE_string(device, "CPU", "Computation device: CPU|CUDA");
CAFFE2_DEFINE_string(net_type, "dag",
                     "simple, dag, async_{simple,dag,polling,scheduling}, "
                     "singlethread_async");
CAFFE2_DEFINE_int(num_workers, 4, "Level of parallelism (?).");

using std::string;
using std::unique_ptr;
using std::vector;

namespace {
std::vector<int> split_to_ints(char sep, const string& the_string) {
  vector<int> the_ints;
  vector<string> ints_str = caffe2::split(sep, the_string);
  for (const string& int_str : ints_str) {
    the_ints.push_back(caffe2::stoi(int_str));
  }
  return the_ints;
}

// From caffe2_cpp_tutorial/include/caffe2/util/cmd.h
bool cmd_setup_cuda() {
  static bool already_done = false;
  if (already_done) { return true; }
  already_done = true;
#ifdef WITH_CUDA
  DeviceOption option;
  option.set_device_type(CUDA);
  new CUDAContext(option);
  return true;
#else
  return false;
#endif
}

void SetDevice(const std::string& device_name, caffe2::NetDef* net_def) {
  caffe2::DeviceType device_type;
  if (!caffe2::DeviceType_Parse(device_name, &device_type)) {
    throw std::runtime_error("Invalid device type " + device_name);
  }
  if (device_type == caffe2::CUDA) {
    if (!cmd_setup_cuda()) {
      throw std::runtime_error("This build does not support CUDA");
    }
  }
  net_def->mutable_device_option()->set_device_type(device_type);
}
}  // namespace


int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  unique_ptr<caffe2::Workspace> workspace(new caffe2::Workspace());

  // Run initialization network.
  caffe2::NetDef net_def;
  CAFFE_ENFORCE(ReadProtoFromFile(caffe2::FLAGS_init_net, &net_def));
  SetDevice(caffe2::FLAGS_device, &net_def);
  CAFFE_ENFORCE(workspace->RunNetOnce(net_def));

  // Load the main network.
  CAFFE_ENFORCE(ReadProtoFromFile(caffe2::FLAGS_net, &net_def));
  net_def.set_type(caffe2::FLAGS_net_type);
  net_def.set_num_workers(caffe2::FLAGS_num_workers);
  SetDevice(caffe2::FLAGS_device, &net_def);

  // Load input.
  if (caffe2::FLAGS_input.size()) {
    vector<string> input_names = caffe2::split(',', caffe2::FLAGS_input);
    if (caffe2::FLAGS_input_file.size()) {
      vector<string> input_files = caffe2::split(',', caffe2::FLAGS_input_file);
      CAFFE_ENFORCE_EQ(
          input_names.size(),
          input_files.size(),
          "Input name and file should have the same number.");
      for (int i = 0; i < input_names.size(); ++i) {
        caffe2::BlobProto blob_proto;
        CAFFE_ENFORCE(caffe2::ReadProtoFromFile(input_files[i], &blob_proto));
        workspace->CreateBlob(input_names[i])->Deserialize(blob_proto);
      }
    } else if (caffe2::FLAGS_input_dims.size()) {
      vector<string> input_dims_list =
          caffe2::split(';', caffe2::FLAGS_input_dims);
      CAFFE_ENFORCE_EQ(
          input_names.size(),
          input_dims_list.size(),
          "Input name and dims should have the same number of items.");
      for (int i = 0; i < input_names.size(); ++i) {
        vector<int> input_dims = split_to_ints(',', input_dims_list[i]);
        if (!workspace->HasBlob(input_names[i])) {
          workspace->CreateBlob(input_names[i]);
        }
        caffe2::TensorCPU* tensor =
            workspace->GetBlob(input_names[i])->GetMutable<caffe2::TensorCPU>();
        tensor->Resize(input_dims);
        tensor->mutable_data<float>();
      }
    } else {
      CAFFE_THROW(
          "You requested input tensors, but neither input_file nor "
          "input_dims is set.");
    }
  } else {
    // This comes from https://github.com/caffe2/caffe2/issues/328:
    VLOG(4) << "DEBUG: GetBlob(" << net_def.external_input(0) << ")...";
    auto* b = workspace->GetBlob(net_def.external_input(0))
              ->GetMutable<caffe2::TensorCPU>();
    b->Resize(split_to_ints(',', caffe2::FLAGS_input_dims));
    b->mutable_data<float>();
    VLOG(4) << "DEBUG " << __func__ << ": TensorCPU b=" << b->DebugString();
  }

  // force changing engine and algo
  if (caffe2::FLAGS_force_engine) {
    LOG(INFO) << "force engine be: " << caffe2::FLAGS_engine;
    for (const auto& op : net_def.op()) {
      const_cast<caffe2::OperatorDef*>(&op)->set_engine(caffe2::FLAGS_engine);
    }
  }
  if (caffe2::FLAGS_force_algo) {
    LOG(INFO) << "force algo be: " << caffe2::FLAGS_algo;
    for (const auto& op : net_def.op()) {
      caffe2::GetMutableArgument(
          "algo", true, const_cast<caffe2::OperatorDef*>(&op))
          ->set_s(caffe2::FLAGS_algo);
    }
  }

  VLOG(4) << "DEBUG: CreateNet...";
  caffe2::NetBase* net = workspace->CreateNet(net_def);
  CHECK_NOTNULL(net);
  VLOG(4) << "DEBUG: net->Run()";
  CAFFE_ENFORCE(net->Run());
  VLOG(4) << "DEBUG: TEST_Benchmark...";
  net->TEST_Benchmark(
      caffe2::FLAGS_warmup, caffe2::FLAGS_iter, caffe2::FLAGS_run_individual);

  string output_prefix = caffe2::FLAGS_output_folder.size()
      ? caffe2::FLAGS_output_folder + "/"
      : "";
  if (caffe2::FLAGS_output.size()) {
    vector<string> output_names = caffe2::split(',', caffe2::FLAGS_output);
    if (caffe2::FLAGS_output == "*") {
      output_names = workspace->Blobs();
    }
    for (const string& name : output_names) {
      VLOG(4) << "DEBUG " << __func__ << ": output name=" << name;
      CAFFE_ENFORCE(
          workspace->HasBlob(name),
          "You requested a non-existing blob: ",
          name);
      string serialized = workspace->GetBlob(name)->Serialize(name);
      string output_filename = output_prefix + name;
      caffe2::WriteStringToFile(serialized, output_filename.c_str());
    }
  }

  return 0;
}
