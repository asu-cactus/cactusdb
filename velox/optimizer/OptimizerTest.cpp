#include <iostream>
#include <folly/init/Init.h>
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"
#include "velox/parse/Expressions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/Type.h"
#include "velox/expression/VectorFunction.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/tests/utils/VectorMaker.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

#include "velox/ml_functions/NNBuilder.h"
#include <fstream>
#include <sstream>
#include <random>

#include "velox/exec/FilterProject.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"

#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <json/json.h>

#include "RewriteAction.h"
#include "TorchNN2TwoLayerUDFRewriteAction.h"
// #include "Merge2SingleRewriteAction.h"
// #include "Mul2JoinAggRewriteAction.h"
// #include "JoinAgg2MulRewriteAction.h"

#include "PlanState.h"
#include "RuleManager.h"
#include "Register.h"
// #define EIGEN_USE_BLAS

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::test;


using exec::test::HiveConnectorTestBase;

constexpr int64_t KB = 1024L;
constexpr int64_t MB = 1024L * KB;
constexpr int64_t GB = 1024L * MB;

struct FileStructure;
struct DataFrame;
struct DynamicMetaData;
struct PlanBuilderExec;
struct OptOutput;



class MyFileTest;

static void waitForFinishedDrivers(const std::shared_ptr<exec::Task>& task);
std::vector<std::vector<float>> create_input_block(int total_size, std::vector<std::vector<float>>& values, int block_numbers);
std::vector<std::vector<float>> create_weight_block(int total_size, float* values, int block_numbers);
std::vector<std::vector<float>> create_block_index(int parts, int flag);
::FileStructure block_to_files(std::vector<std::vector<float>> valuesArray, int parts, int flag);
DataFrame data_generate(int features, int samples, int first_layer, int second_layer);
PlanBuilderExec build_plan_udf(DataFrame data, int features, int first_layer, int second_layer);
PlanBuilderExec build_plan_udf_torch(DataFrame data, int features, int first_layer, int second_layer, int memoryLimit, std::vector<std::vector<float>> feature, int splitsNum, int threadsNum);
void exec_plan_udf(PlanBuilderExec planBuilderExec, int memoryLimit, std::vector<std::vector<float>> features, int splitsNum, int threadsNum);
DynamicMetaData decision_maker(PlanBuilder& planBuilder);
PlanBuilderExec build_plan_op(float* weight, int row, int col, int samples, RowTypePtr inputs, RowTypePtr weights, std::vector<std::string> str, std::vector<std::string> targetString);
OptOutput optiming_plan(PlanBuilder& planBuilder, DataFrame data, int num_samples, int features_size, int first_layer_size, int second_layer_size);
void exec_plan_relational(PlanBuilderExec planBuilderOpt, int memoryLimit, std::vector<std::shared_ptr<TempFilePath>> inputPaths, 
std::vector<std::shared_ptr<TempFilePath>> weightPaths, int threadsNum);
void exec_pure_torch(int num_samples, int input_size, int layer1_size, int layer2_size,float* w1,float* w2,float* b1,float* b2,float* input_values);
void test_optimizer_demo(int argc, char** argv);
std::shared_ptr<PlanBuilderExec> rewriten_udf(PlanBuilder& udf_plan_builder, DataFrame data, int features, int first_layer, int second_layer, std::string test_action);
PlanBuilderExec rewriten_tradition(PlanBuilderExec plan_s1_builder);

auto pool_ = memory::MemoryManager::getInstance()->addLeafPool();
std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};
std::shared_ptr<core::QueryCtx> queryCtx_{
      std::make_shared<core::QueryCtx>(executor_.get())};
// std::shared_ptr<core::QueryCtx> newQueryCtx(
//       int64_t memoryCapacity) {
    
//     std::unordered_map<std::string, std::shared_ptr<Config>> configs;
//     std::shared_ptr<MemoryPool> pool = memory::MemoryManager::getInstance()->addRootPool(
//         "", memoryCapacity, MemoryReclaimer::create());
//    std::unordered_map<std::string, std::string> myMapWithValues = {{core::QueryConfig::kSpillEnabled, "true"}, 
//                                       {core::QueryConfig::kJoinSpillEnabled, "true"},  
//                                       {core::QueryConfig::kJoinSpillMemoryThreshold, "1"},
//                                        {core::QueryConfig::kSpillableReservationGrowthPct, "1"},
//                                       //  {core::QueryConfig::kSpillPartitionBits, "1"}
//                                       };
//     auto queryCtx = std::make_shared<core::QueryCtx>(
//         executor_.get(),
//         myMapWithValues,
//         configs,
//         memory::MemoryAllocator::getInstance(),
//         std::move(pool));
//     return queryCtx;
//   }

VectorMaker maker{pool_.get()};

struct FileStructure {
  std::vector<std::shared_ptr<TempFilePath>> paths;
  RowTypePtr schema;
};

struct DataFrame {
    std::vector<std::vector<float>> features;
    std::vector<float*> weights;
    std::vector<float*> bias;
    float* featuresFloat;
};

struct DynamicMetaData {
    bool replaceFlag;
    int blocksNum;
    std::vector<int> leftBlockSize;
    std::vector<int> rightBlockSize;
    std::vector<int> planIdShot;
    std::vector<std::string> targetStr;
};

struct PlanBuilderExec{
  std::shared_ptr<PlanBuilder> planBuilder;
  std::vector<core::PlanNodeId> p;
  PlanBuilderExec(std::shared_ptr<PlanBuilder> builder, std::vector<core::PlanNodeId> ids):planBuilder(builder), p(ids){}
};

struct OptOutput {
  bool flag;
  std::vector<std::shared_ptr<TempFilePath>> inputsPaths;
  std::vector<std::shared_ptr<TempFilePath>> weightPaths;
  PlanBuilderExec planBuilderExec;
  OptOutput(bool fl, std::vector<std::shared_ptr<TempFilePath>> iPaths, std::vector<std::shared_ptr<TempFilePath>> wPaths, PlanBuilderExec builderExec) : flag(fl), inputsPaths(iPaths), weightPaths(wPaths), planBuilderExec(builderExec) {}
};


class MyFileTest : public HiveConnectorTestBase {
  public:
  MyFileTest(){
    // SetUp();
  }
  ~MyFileTest() {
  }

  void SetUp() {
    HiveConnectorTestBase::SetUp();
  }

  void TestBody() override {}

};

static void waitForFinishedDrivers(const std::shared_ptr<exec::Task>& task) {
  while (!task->isFinished()) {     
    usleep(1000); // 0.01 second.
  }
}

std::vector<std::vector<float>> create_input_block(int total_size, std::vector<std::vector<float>>& values, int block_numbers){
    std::vector<std::vector<float>> valuesArray;
    std::vector<float> flattened;
    for (const auto& row : values) {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }

    auto block_size = total_size / block_numbers;
    for (int i = 0; i < block_numbers; i++) {
        std::vector<float> valuesArraySingleBlock;
        for (int j = 0; j < block_size; j++) {
                valuesArraySingleBlock.push_back(flattened[i*block_size+j]);
        }
        valuesArray.push_back(valuesArraySingleBlock);
    }
    return valuesArray;
}

std::vector<std::vector<float>> create_weight_block(int total_size, float* values, int block_numbers){
    std::vector<std::vector<float>> valuesArray;
    auto block_size = total_size / block_numbers;
    for (int i = 0; i < block_numbers; i++) {
        std::vector<float> valuesArraySingleBlock;
        for (int j = 0; j < block_size; j++) {
                valuesArraySingleBlock.push_back(values[i*block_size+j]);
        }
        valuesArray.push_back(valuesArraySingleBlock);
    }
    return valuesArray;
}

std::vector<std::vector<float>> create_block_index(int parts, int flag){
  std::vector<std::vector<float>> indexs;
  if (flag == 0){
    std::vector<float> indexs_row;
    std::vector<float> indexs_col;
    for (int i = 0; i < parts; i++){
      indexs_row.push_back(0);
    }
    for (int i = 0; i < parts; i++){
      indexs_col.push_back(i);
    }
    indexs.push_back(indexs_row);
    indexs.push_back(indexs_col);
  }
  else {
    std::vector<float> indexs_row;
    std::vector<float> indexs_col;
    for (int i = 0; i < parts; i++){
      indexs_row.push_back(i);
    }
    for (int i = 0; i < parts; i++){
      indexs_col.push_back(0);
    }
    indexs.push_back(indexs_row);
    indexs.push_back(indexs_col);
  }
  return indexs;
}

::FileStructure block_to_files(std::vector<std::vector<float>> valuesArray, int parts, int flag){
  ::MyFileTest myFile;
  ::FileStructure myFileStructure;
  std::vector<std::shared_ptr<TempFilePath>> paths;
  RowVectorPtr input;
  if (flag == 0){
    auto indexs = ::create_block_index(parts, flag);
    for (int i = 0; i < parts; i++){
      input = maker.rowVector({"v", "v_row", "v_col"}, 
      {maker.arrayVector<float>({valuesArray[i]}, REAL()), maker.flatVector({indexs[0][i]}), maker.flatVector({indexs[1][i]})});
      auto file = TempFilePath::create();
      myFile.writeToFile(file->path, {input});
      paths.push_back(file);
  }
  myFileStructure.paths = paths;
  myFileStructure.schema = asRowType(input->type());
  return myFileStructure;
  }
  else {
    auto indexs = ::create_block_index(parts, flag);
    for (int i = 0; i < parts; i++){
      input = maker.rowVector({"w", "w_row", "w_col"}, 
      {maker.arrayVector<float>({valuesArray[i]}, REAL()), maker.flatVector({indexs[0][i]}), maker.flatVector({indexs[1][i]})});
      auto file = TempFilePath::create();
      myFile.writeToFile(file->path, {input});
      paths.push_back(file);
  }
  myFileStructure.paths = paths;
  myFileStructure.schema = asRowType(input->type());
  return myFileStructure;
  }

}

DataFrame data_generate(int features, int samples, int first_layer, int second_layer){
  int input_features_size = features;
  int num_samples = samples;

  int first_layer_output_size = first_layer;
  int second_layer_output_size = second_layer;
  // ( 1000 * 597540 x 597540 * 1024 + 1000*1024) first layer
  // ( 1000 * 1024 x 1024 * 14588 + 1000*14588) second layer
  int input_total_size = input_features_size * num_samples;

  int weight_layer1_size = input_features_size * first_layer_output_size;
  int weight_layer2_size = first_layer_output_size * second_layer_output_size;

  int bias_layer1_size = num_samples * first_layer_output_size;
  int bias_layer2_size = num_samples * second_layer_output_size;

  std::random_device rd;  // Seed the random number generator
  std::mt19937 gen(rd()); // Initialize the Mersenne Twister engine
  // std::uniform_real_distribution<float> distribution(0.00000009, 0.00000011); // Define the range
  std::uniform_real_distribution<float> distribution(0.0009, 0.0011);

  //generate input
  std::vector<std::vector<float>> featureVectors;
  for (int i = 0; i < num_samples; i++) {
        std::vector<float> featureVector;
        for (int j = 0; j < input_features_size; j++) {
                featureVector.push_back(i*input_features_size+j);
                // float random_value = distribution(gen);
                // featureVector.push_back(0.0000001);
                // featureVector.push_back(random_value);
        }
        featureVectors.push_back(featureVector);
    }

  float* floatArray = new float[num_samples * input_features_size];
  int index = 0;

  for (const auto& row : featureVectors) {
      for (const float& value : row) {
          floatArray[index++] = value;
      }
  }

  //generate weight
  float* weight_layer1 = new float[weight_layer1_size];
  for (int i = 0; i < weight_layer1_size; ++i) {
      weight_layer1[i] = 0.000001; 
  }
  float* weight_layer2 = new float[weight_layer2_size];
  for (int i = 0; i < weight_layer2_size; ++i) {
      weight_layer2[i] = 0.000001; 
  }
  std::vector<float*> weights;
  weights.push_back(weight_layer1);
  weights.push_back(weight_layer2);

  //generate bias
  float* bias_layer1 = new float[bias_layer1_size];
  for (int i = 0; i < bias_layer1_size; ++i) {
      bias_layer1[i] = 0.00001; 
  }
  float* bias_layer2 = new float[bias_layer2_size];
  for (int i = 0; i < bias_layer2_size; ++i) {
      bias_layer2[i] = 0.00001; 
  }
  std::vector<float*> bias;
  bias.push_back(bias_layer1);
  bias.push_back(bias_layer2);
  // create dataframe
  DataFrame data;
  data.features = featureVectors;
  data.weights = weights;
  data.bias = bias;
  data.featuresFloat = floatArray;

  return data;
}

PlanBuilderExec build_plan_udf(DataFrame data, int features, int first_layer, int second_layer){
  auto featureArrayVector = maker.arrayVector<float>(data.features, REAL());
  auto inputRowVector = maker.rowVector({"v"}, {featureArrayVector});
  core::PlanNodeId p0;
  std::string compute =  NNBuilder()
                        .denseLayer(first_layer, features, data.weights[0], data.bias[0], NNBuilder::RELU)
                        .denseLayer(second_layer, first_layer, data.weights[1], data.bias[1], NNBuilder::SOFTMAX)
                        .build();


  std::cout << compute << std::endl; // softmax5(mat_add4(mat_mul3(relu2(mat_add1(mat_mul0({}))))))
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto planBuilder = exec::test::PlanBuilder(planNodeIdGenerator)
                .tableScan(asRowType(inputRowVector->type()))
                .capturePlanNodeId(p0)
                .project({fmt::format(compute, "v")}) 
                .planBuild();
  
  auto plan = planBuilder.planNode();
  auto projectNode = std::dynamic_pointer_cast<const facebook::velox::core::ProjectNode>(plan);
  auto str = projectNode->projections()[0]->toString();
  std::shared_ptr<PlanBuilder> planBuilderShared = std::make_shared<PlanBuilder>(planBuilder);
  PlanBuilderExec planBuilderExec(planBuilderShared, {p0});
  // exec_plan_udf(planBuilderExec, memoryLimit, feature, splitsNum, threadsNum);
  return planBuilderExec;
}


PlanBuilderExec build_plan_udf_torch(DataFrame data, int features, int first_layer, int second_layer, int memoryLimit, std::vector<std::vector<float>> feature, int splitsNum, int threadsNum){
  auto featureArrayVector = maker.arrayVector<float>(data.features, REAL());
  auto inputRowVector = maker.rowVector({"v"}, {featureArrayVector});
  core::PlanNodeId p0;

  std::vector<int> dimensions;
  dimensions.push_back(features);
  dimensions.push_back(first_layer);
  dimensions.push_back(second_layer);

  float* weights[2] = {data.weights[0], data.weights[1]};
  float* bias[2] = {data.bias[0], data.bias[1]};
  // torch::set_num_threads(8);
  // std::cout << torch::get_num_threads() << std::endl;
  // std::cout << "line 354" << std::endl;
  exec::registerVectorFunction(
    "torchDNN",
    TorchDNN2Level::signatures(),
    std::make_unique<TorchDNN2Level>(weights, bias, dimensions)
  );


  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto planBuilder = exec::test::PlanBuilder(planNodeIdGenerator)
                .tableScan(asRowType(inputRowVector->type()))
                .capturePlanNodeId(p0)
                .project({"torchDNN(v)"})
                .planBuild();
  std::shared_ptr<PlanBuilder> planBuilderShared = std::make_shared<PlanBuilder>(planBuilder);
  PlanBuilderExec planBuilderExec(planBuilderShared, {p0});
  exec_plan_udf(planBuilderExec, memoryLimit, feature, splitsNum, threadsNum);
  return planBuilderExec;
}

void exec_plan_udf(PlanBuilderExec planBuilderExec, int memoryLimit, std::vector<std::vector<float>> features, int splitsNum, int threadsNum){
  std::shared_ptr<memory::MemoryPool> rootPool{memory::MemoryManager::getInstance()->addRootPool("root", memoryLimit * MB)};
  auto planFragment = planBuilderExec.planBuilder->planFragment();
  queryCtx_->testingOverrideMemoryPool(rootPool);
  queryCtx_->testingOverrideConfigUnsafe({{core::QueryConfig::kSpillEnabled, "false"}});//may be this is the key factor (latency)
  //normal one data file
  auto featureArrayVector = maker.arrayVector<float>(features, REAL());
  auto inputRowVector = maker.rowVector({"v"}, {featureArrayVector});
  auto file = TempFilePath::create();
  ::MyFileTest myFile;

  auto config = std::make_shared<facebook::velox::dwrf::Config>();
  uint64_t kSizeKB = 1024UL;
  uint32_t rows = 2000;

  config->set(facebook::velox::dwrf::Config::STRIPE_SIZE, 100 * kSizeKB);
  config->set(facebook::velox::dwrf::Config::ROW_INDEX_STRIDE, rows);

  myFile.writeToFile(file->path, {inputRowVector}, config);

  auto hiveSplits =  myFile.makeHiveConnectorSplits(file->path, splitsNum, dwio::common::FileFormat::DWRF);

  // more data files

  // MyFileTest myFile;
  // std::vector<std::shared_ptr<TempFilePath>> paths;

  // int rowsPerPart = features.size() / 8;

  //   // Create four sub-vectors
  // std::vector<std::vector<std::vector<float>>> parts;

  // for (int i = 0; i < 8; i++) {
  //     int startRow = i * rowsPerPart;
  //     int endRow = (i + 1) * rowsPerPart;

  //     // Create a sub-vector by copying rows from the original vector
  //     std::vector<std::vector<float>> part(features.begin() + startRow, features.begin() + endRow);

  //     // Add the sub-vector to the parts vector
  //     parts.push_back(part);
  // }

  // for (int i = 0; i < 8; i++){
  //     auto featureArrayVector = maker.arrayVector<float>(parts[i], REAL());
  //     auto inputRowVector = maker.rowVector({"v"}, {featureArrayVector});
  //     auto file = TempFilePath::create();
  //     myFile.writeToFile(file->path, {inputRowVector});
  //     paths.push_back(file);
  // }
  // auto hiveSplits = myFile.makeHiveConnectorSplits(paths);

  boost::interprocess::interprocess_semaphore semaphore(threadsNum);
  // torch::set_num_threads(8);
  // Eigen::setNbThreads(1);
  // std::cout << Eigen::nbThreads() << " main"<< std::endl;
  // std::cout << torch::get_num_threads() << std::endl;
  // std::cout << "line 428" << std::endl;
  auto task = exec::Task::create("0", planFragment, 0, queryCtx_, 
        [&semaphore](RowVectorPtr result, ContinueFuture* /*unused*/) {
          if(result){
            semaphore.post();
          }
          return exec::BlockingReason::kNotBlocked;
  });

  task->start(threadsNum);
  std::cout << "Hive splits:" << std::endl;
  for(auto& split : hiveSplits) {
    semaphore.wait();
    std::cout << split->toString() << std::endl;
    task->addSplit(planBuilderExec.p[0], exec::Split(std::move(split)));
  }
  task->noMoreSplits(planBuilderExec.p[0]);
  std::cout << std::endl;
 
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  waitForFinishedDrivers(task);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Total time (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;

  // another execute method
  // auto planAssert = planBuilderExec.planBuilder->planNode();
  // DuckDbQueryRunner duckDbQueryRunner_;
  // auto results = exec::test::AssertQueryBuilder(planAssert, duckDbQueryRunner_)
  // .split(planBuilderExec.p[0], myFile.makeHiveConnectorSplit(file->path))
  // // .splits(planBuilderExec.p[0], myFile.makeHiveConnectorSplits(paths))
  // .copyResults(pool_.get());
  // std::cout << "test udf Results:" << results->toString() << std::endl;
  // std::cout << results->toString(0, 100) << std::endl;
}

DynamicMetaData decision_maker(PlanBuilder& planBuilder){
  //Todo: add true dynamic tool
    DynamicMetaData decisions;
    decisions.replaceFlag = true;
    decisions.blocksNum = 4;
    decisions.planIdShot = {0, 1, 0, 0, 1};//not used, Todo
    decisions.leftBlockSize.push_back(1000);
    decisions.leftBlockSize.push_back(200);//597540/4 = 149385
    decisions.rightBlockSize.push_back(200);
    decisions.rightBlockSize.push_back(1024);
    decisions.targetStr.push_back("mat_mul0(ROW[\"v\"])");// only test first layer, not auto determined
    return decisions;
}

PlanBuilderExec build_plan_op(float* weight, int row, int col, int samples, RowTypePtr inputs, RowTypePtr weights, std::vector<std::string> str, std::vector<std::string> targetString){
  exec::registerVectorFunction(
    "mat_mul_b",
    MatrixMultiply_b::signatures(),
    std::make_unique<MatrixMultiply_b>(row, col, samples, 4)
  );
  
  std::string searchString = targetString[0];
  std::string replaceString = "result";

  std::size_t found = str[0].find(searchString);
  while (found != std::string::npos) {
      str[0].replace(found, searchString.length(), replaceString);
      found = str[0].find(searchString, found + replaceString.length());
  }
  
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId p2;
  core::PlanNodeId p3;
  auto planOpt = exec::test::PlanBuilder(planNodeIdGenerator)
                  .tableScan(inputs)
                  // .values({input})
                  .capturePlanNodeId(p2)
                  .hashJoin(
                      {"v_col"},
                      {"w_row"},
                    exec::test::PlanBuilder(planNodeIdGenerator)
                   .tableScan(weights)
                  // .values({weightb})
                   .capturePlanNodeId(p3)
                   .planNode(),
                    "", // extra filter
                    {"v_row", "w_col", "v", "w"})
                  .project({"v_row", "w_col", "mat_mul_b(v, w) AS mp"})
                  .singleAggregation({"w_col","v_row"}, {"array_sum(mp) AS result"})
                  .project({str[0]})
                  .planBuild();
  std::shared_ptr<PlanBuilder> planBuilderOptShared = std::make_shared<PlanBuilder>(planOpt);
  PlanBuilderExec planOptBuilderExec(planBuilderOptShared, {p2, p3});
  return planOptBuilderExec;
}

OptOutput optiming_plan(PlanBuilder& planBuilder, DataFrame data, int num_samples, int features_size, int first_layer_size, int second_layer_size){
  auto dyDecision = decision_maker(planBuilder);
  if (dyDecision.replaceFlag) {
    auto inputBlocks = ::create_input_block(num_samples*features_size, data.features, dyDecision.blocksNum);
    //here is only for layer 1
    auto weightBlocks = ::create_weight_block(features_size*first_layer_size, data.weights[0], dyDecision.blocksNum);
    auto inputs = ::block_to_files(inputBlocks, dyDecision.blocksNum, 0);//0 denote values, 1 denote weight
    auto weights = ::block_to_files(weightBlocks, dyDecision.blocksNum, 1);
    auto nodeid = planBuilder.planNode()->id();
    // auto str = planBuilder.findExprStrings(nodeid);
    // "softmax5(mat_add4(mat_mul3(relu2(mat_add1(mat_mul0(ROW[\"v\"]))))))"
    // auto str = std::vector<std::string>{"softmax5(mat_add4(mat_mul3(relu2(mat_add1(mat_mul0(v))))))"};
    auto plan = planBuilder.planNode();
    auto projectNode = std::dynamic_pointer_cast<const facebook::velox::core::ProjectNode>(plan);
    auto str = std::vector<std::string>{};
    str.push_back(projectNode->projections()[0]->toString());
    auto planBuilderOpt = build_plan_op(data.weights[0], dyDecision.rightBlockSize[0], dyDecision.rightBlockSize[1], dyDecision.leftBlockSize[0], inputs.schema, weights.schema, str, dyDecision.targetStr);
    OptOutput optOutput(true, inputs.paths, weights.paths, planBuilderOpt);
    return optOutput;
  }
  else {
    std::shared_ptr<PlanBuilder> planBuilderShared = std::make_shared<PlanBuilder>(planBuilder);
    PlanBuilderExec planNoOptExec(planBuilderShared, {});
    OptOutput optOutput(false, {}, {}, planNoOptExec);
    return optOutput;
  }
}

void exec_plan_relational(PlanBuilderExec planBuilderOpt, int memoryLimit, std::vector<std::shared_ptr<TempFilePath>> inputPaths, 
std::vector<std::shared_ptr<TempFilePath>> weightPaths, int threadsNum){
  std::shared_ptr<memory::MemoryPool> rootPool{memory::MemoryManager::getInstance()->addRootPool("root_relational", memoryLimit * MB)}; // 280 pass for 4 threads, 40 for 1 thread
  queryCtx_->testingOverrideMemoryPool(rootPool);
  queryCtx_->testingOverrideConfigUnsafe({{core::QueryConfig::kSpillEnabled, "true"}});
  auto planFragmentOpt = planBuilderOpt.planBuilder->planFragment();
  ::MyFileTest myFile;
  auto inputHiveSplits = myFile.makeHiveConnectorSplits(inputPaths);
  auto weightHiveSplits = myFile.makeHiveConnectorSplits(weightPaths);

  boost::interprocess::interprocess_semaphore semaphore(threadsNum*8);
  auto task = exec::Task::create("0", planFragmentOpt, 0, queryCtx_, 
      [&semaphore](RowVectorPtr result, ContinueFuture* /*unused*/) {
        if(result){
          semaphore.post();
        }
        return exec::BlockingReason::kNotBlocked;
  });

  task->start(threadsNum);
  std::cout << "Hive splits:" << std::endl;
  for(auto& split : inputHiveSplits) {
    semaphore.wait();
    std::cout << split->toString() << std::endl;
    task->addSplit(planBuilderOpt.p[0], exec::Split(std::move(split)));
  }
  for(auto& split : weightHiveSplits) {
    semaphore.wait();
    std::cout << split->toString() << std::endl;
    task->addSplit(planBuilderOpt.p[1], exec::Split(std::move(split)));
  }
  task->noMoreSplits(planBuilderOpt.p[0]);
  task->noMoreSplits(planBuilderOpt.p[1]);

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  waitForFinishedDrivers(task);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Total time (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;

  //anthor execute method
  // auto planOptAssert = planBuilderOpt.planBuilder->planNode();
  // DuckDbQueryRunner duckDbQueryRunner_;
  // auto results = exec::test::AssertQueryBuilder(planOptAssert, duckDbQueryRunner_)
  // .splits(planBuilderOpt.p[0], myFile.makeHiveConnectorSplits(inputPaths))
  // .splits(planBuilderOpt.p[1], myFile.makeHiveConnectorSplits(weightPaths))
  // .copyResults(pool_.get());
  // std::cout << "relational Results:" << results->toString() << std::endl;
  // std::cout << results->toString(0, 100) << std::endl;

}

void exec_pure_torch(int num_samples, int input_size, int layer1_size, int layer2_size,float* w1,float* w2,float* b1,float* b2,float* input_values){
    torch::nn::Linear dense1(input_size, layer1_size);  
    torch::nn::Linear dense2(layer1_size,layer2_size);
    torch::nn::ReLU relu;
    
    torch::Tensor weightTensor1 = torch::from_blob(w1, {input_size, layer1_size}).t();
    torch::Tensor weightTensor2 = torch::from_blob(w2, {layer1_size, layer2_size}).t();
    torch::Tensor bias1 = torch::from_blob(b1, {layer1_size});
    torch::Tensor bias2 = torch::from_blob(b2, {layer2_size});

    dense1->weight.set_data(weightTensor1);
    dense2->weight.set_data(weightTensor2);
    dense1->bias.set_data(bias1);
    dense2->bias.set_data(bias2);

    int batch_size = 100;  // Adjust as needed
    int batch_count = 0;
    int c = 1000/batch_size;
    torch::Tensor input = torch::from_blob(input_values, {num_samples,  input_size});
    torch::data::datasets::TensorDataset dataset(input);
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(dataset), batch_size);

    for (auto& batch : *data_loader) {
        torch::Tensor x = input;
        torch::Tensor layer1_output = dense1->forward(x.reshape({x.size(0), input_size}));
        torch::Tensor reluOutput = relu->forward(layer1_output);
        torch::Tensor layer2_output = dense2->forward(reluOutput);
        torch::Tensor softmax_output = torch::nn::functional::softmax(layer2_output, 1);
        float* result = softmax_output.data_ptr<float>();
        if(--c == 0)
            break;   
    }

}

void test_optimizer_demo(int argc, char** argv){

  folly::init(&argc, &argv, false);
  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  parse::registerTypeResolver();
  const std::string kHiveConnectorId = "test-hive";
  auto hiveConnector =
      connector::getConnectorFactory(
          connector::hive::HiveConnectorFactory::kHiveConnectorName)
          ->newConnector(kHiveConnectorId, std::make_shared<core::MemConfig>());
  connector::registerConnector(hiveConnector);

  filesystems::registerLocalFileSystem();
  dwrf::registerDwrfReaderFactory();
  // int input_features_size = 597540;
  int input_features_size = 800;
  int num_samples = 1000;

  int first_layer_output_size = 1024;
  int second_layer_output_size = 14588;

  int memory_limit_udf = 1000;//mb
  int splits_num_udf = 4;
  int threads_num_udf = 4;

  int memory_limit_rela = 100000;//mb
  int splits_num_rela = 4;
  int threads_num_rela = 4;
  
  int flag = 2;
  auto data = data_generate(input_features_size, num_samples, first_layer_output_size, second_layer_output_size);
  if (flag == 0){
    auto udf_plan_builder = build_plan_udf(data, input_features_size, first_layer_output_size, second_layer_output_size);
    exec_plan_udf(udf_plan_builder, memory_limit_udf, data.features, splits_num_udf, threads_num_udf);
  }
  else if(flag == 1){
    auto udf_plan_torch_builder = build_plan_udf_torch(data, input_features_size, first_layer_output_size, second_layer_output_size, memory_limit_udf, data.features, splits_num_udf, threads_num_udf);
  }
  else if(flag == 2){
    auto udf_plan_builder = build_plan_udf(data, input_features_size, first_layer_output_size, second_layer_output_size);
    auto relational_plan = optiming_plan(*(udf_plan_builder.planBuilder), data, num_samples, input_features_size, first_layer_output_size, second_layer_output_size);
    exec_plan_relational(relational_plan.planBuilderExec, memory_limit_rela, relational_plan.inputsPaths, relational_plan.weightPaths, threads_num_rela);
  }
  else {
    const int numThreads = 1;
    std::vector<std::thread> threads;
   
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(exec_pure_torch, num_samples, input_features_size, first_layer_output_size, second_layer_output_size, 
        data.weights[0],data.weights[1], data.bias[0], data.bias[1], data.featuresFloat);
    }
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    for (std::thread& thread : threads) {
        thread.join();
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time for Test (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
  }
}

class Opt {
private:
    std::shared_ptr<PlanBuilder> originPlanBuilder;
    PlanBuilder possiblePlanBuilder;
    int clientSocket;
    DataFrame data;
     //TODO:get from planbuilder or querycontext directly

public:
    // Constructor
    Opt(std::shared_ptr<PlanBuilder> builder, int client, DataFrame data) : originPlanBuilder(builder), possiblePlanBuilder(), clientSocket(client), data(data) {}

    // Member function that uses the PlanBuilder
    void start() {
        std::cout << "Executing optimization." << std::endl;
        const char* start_str = "start";
        send(clientSocket, start_str, std::strlen(start_str), 0);
    }

    int costFunction(std::vector<std::string> actions) {
        auto action = actions[0];// fix more actions, sum up cost
        if (action == "Merge2Single" ){
          if (data.features[0].size() > 10000){
            return 5;
          }
          else {
            return 10;
          }
        }
        if (action == "Mul2JoinAgg"){
          if (data.features[0].size() > 10000){
            return 10;
          }
          else {
            return 5;
          }
        }
        return 0;
    }


    void rewriten_udf(std::string test_action){
      if (test_action == "Merge2Single"){
        core::QueryConfig config({});
        auto mat_mul0 = std::dynamic_pointer_cast<MatrixMultiply>(exec::getVectorFunction("mat_mul0", {ARRAY(REAL())}, {}, config));
        auto mat_mul3 = std::dynamic_pointer_cast<MatrixMultiply>(exec::getVectorFunction("mat_mul3", {ARRAY(REAL())}, {}, config));
        // std::vector<int> dimensions;
        auto dims0 = mat_mul0->getDims();
        auto dims3 = mat_mul3->getDims();
        std::vector<int> dimensions;
        dimensions.push_back(dims0[0]);
        dimensions.push_back(dims0[1]);
        dimensions.push_back(dims3[1]);

        float* weights[2] = {data.weights[0], data.weights[1]};
        float* bias[2] = {data.bias[0], data.bias[1]};

        exec::registerVectorFunction(
          "torchDNN",
          TorchDNN2Level::signatures(),
          std::make_unique<TorchDNN2Level>(weights, bias, dimensions)
        );

        auto oldplan = originPlanBuilder->planNode()->sources()[0];
        possiblePlanBuilder.replacePlan(oldplan);
        possiblePlanBuilder.project({"torchDNN(v)"});
        // originPlanBuilder->replacePlan(oldplan);
        // originPlanBuilder->project({"torchDNN(v)"});
      }
      else if (test_action == "Mul2JoinAgg"){
        core::QueryConfig config({});
        auto mat_mul0 = std::dynamic_pointer_cast<MatrixMultiply>(exec::getVectorFunction("mat_mul0", {ARRAY(REAL())}, {}, config));
        auto mat_mul3 = std::dynamic_pointer_cast<MatrixMultiply>(exec::getVectorFunction("mat_mul3", {ARRAY(REAL())}, {}, config));
        int blocks = 4;
        auto dims0 = mat_mul0->getDims();
        exec::registerVectorFunction(
        "mat_mul_b",
        MatrixMultiply_b::signatures(),
        std::make_unique<MatrixMultiply_b>(dims0[0]/blocks, dims0[1], data.features.size(), 4)
      );

      std::string searchString = "mat_mul0(ROW[\"v\"])";
      std::string replaceString = "result";

      auto plan = originPlanBuilder->planNode();
      auto projectNode = std::dynamic_pointer_cast<const facebook::velox::core::ProjectNode>(plan);
      auto str = std::vector<std::string>{};
      str.push_back(projectNode->projections()[0]->toString());

      std::size_t found = str[0].find(searchString);
      while (found != std::string::npos) {
          str[0].replace(found, searchString.length(), replaceString);
          found = str[0].find(searchString, found + replaceString.length());
      }
      
      auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
      core::PlanNodeId p1;
      core::PlanNodeId p2;

      auto inputs = ROW({
            {"v", ARRAY(REAL())},
            {"v_row", BIGINT()},
            {"v_col", BIGINT()},
        });

      auto weights = ROW({
            {"w", ARRAY(REAL())},
            {"w_row", BIGINT()},
            {"w_col", BIGINT()},
        });

      auto planBuilder = exec::test::PlanBuilder(planNodeIdGenerator)
                      .tableScan(inputs)
                      .capturePlanNodeId(p1)
                      .hashJoin(
                          {"v_col"},
                          {"w_row"},
                        exec::test::PlanBuilder(planNodeIdGenerator)
                      .tableScan(weights)
                      .capturePlanNodeId(p2)
                      .planNode(),
                        "", // extra filter
                        {"v_row", "w_col", "v", "w"})
                      .project({"v_row", "w_col", "mat_mul_b(v, w) AS mp"})
                      .singleAggregation({"w_col","v_row"}, {"array_sum(mp) AS result"})
                      .project({str[0]})
                      .planNode();
      possiblePlanBuilder.replacePlan(planBuilder);
      }

      else if(test_action == "Split2Multi"){
        core::QueryConfig config({});
        // auto torch = std::dynamic_pointer_cast<TorchDNN>(exec::getVectorFunction("torchDNN", {ARRAY(REAL())}, {}, config));
        // // std::vector<int> dimensions;
        // auto dims = torch->getDims();

        // core::PlanNodeId p0;
        // std::string compute =  NNBuilder()
        //                       .denseLayer(dims[1], dims[0], data.weights[0], data.bias[0], NNBuilder::RELU)
        //                       .denseLayer(dims[2], dims[1], data.weights[1], data.bias[1], NNBuilder::SOFTMAX)
        //                       .build();


        // auto inputs = ROW({
        //     {"v", ARRAY(REAL())},
        // });
        // //connect with oldplan? maybe
        // auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
        // auto planBuilder = exec::test::PlanBuilder(planNodeIdGenerator)
        //               .tableScan(inputs)
        //               .capturePlanNodeId(p0)
        //               .project({fmt::format(compute, "v")}) 
        //               .planNode();
        // possiblePlanBuilder.replacePlan(planBuilder);

      }

      else {
        possiblePlanBuilder.replacePlan(originPlanBuilder->planNode());
      }
    }

    void mcts_optimizer(){

      // Send a list of strings to Python
      char str_buffer[1024];
      std::vector<std::string> action_strings;
      auto plan = originPlanBuilder->planNode();
      auto projectNode = std::dynamic_pointer_cast<const facebook::velox::core::ProjectNode>(plan);
      // auto udf_strings = std::vector<std::string>{};
      // //softmax5(mat_add4(mat_mul3(relu2(mat_add1(mat_mul0({/"x"}))))))
      // udf_strings.push_back(projectNode->projections()[0]->toString());
      // // std::vector<std::string> input_strings = {"string1", "string2", "string3"};
      
      Json::Value input_json;
      input_json["strings"] = Json::arrayValue;
      input_json["strings"].append(projectNode->projections()[0]->toString());
      std::string json_str = input_json.toStyledString();
      send(clientSocket, json_str.c_str(), json_str.length(), 0);
      // Wait for the MCTS result from Python
      while (true) {
          // Receive a string from Python
          ssize_t bytesRead = recv(clientSocket, str_buffer, sizeof(str_buffer), 0);
          if (bytesRead <= 0) {
              break;  // Connection closed or error
          }

          str_buffer[bytesRead] = '\0';  // Null-terminate the received data
          std::string received_str(str_buffer);

          Json::Value json_data;
          Json::CharReaderBuilder jsonReader;
          std::istringstream jsonStream(received_str);
          Json::parseFromStream(jsonReader, jsonStream, &json_data, nullptr);

          const Json::Value& action_strings_json = json_data["action_strings"];
          std::vector<std::string> action_strings;

          for (const auto& action_string_json : action_strings_json) {
              action_strings.push_back(action_string_json.asString());
          }

          if (json_data["end"].asString() == "F") {
              for (auto action : action_strings){
                rewriten_udf(action);
              }
              // Call your cost function with the vector of action strings
              int result = costFunction(action_strings);
            
              std::cout << "cost = " << result << std::endl;

              // Send the result back to Python
              int send_result = htonl(result);
              send(clientSocket, &send_result, sizeof(send_result), 0);
              action_strings.clear();
          }

          if (json_data["end"].asString() == "T") {
              break;  // End communication
          }
      }

      // Close the connection
      close(clientSocket);
    }
};






void test_optimizer_mcts(int argc, char** argv){
  folly::init(&argc, &argv, false);
  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  parse::registerTypeResolver();
  const std::string kHiveConnectorId = "test-hive";
  auto hiveConnector =
      connector::getConnectorFactory(
          connector::hive::HiveConnectorFactory::kHiveConnectorName)
          ->newConnector(kHiveConnectorId, std::make_shared<core::MemConfig>());
  connector::registerConnector(hiveConnector);

  filesystems::registerLocalFileSystem();
  dwrf::registerDwrfReaderFactory();

  int clientSocket = socket(AF_INET, SOCK_STREAM, 0);

  if (clientSocket == -1) {
      std::cerr << "Error creating socket\n";
  }

  sockaddr_in serverAddr;
  serverAddr.sin_family = AF_INET;
  serverAddr.sin_addr.s_addr = inet_addr("127.0.0.1");
  serverAddr.sin_port = htons(12345);

  if (connect(clientSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == -1) {
      std::cerr << "Error connecting to server\n";
      close(clientSocket);
  }

  int input_features_size = 597540;//597540
  int num_samples = 1000;
  int first_layer_output_size = 1024;
  int second_layer_output_size = 14588;
  auto data = data_generate(input_features_size, num_samples, first_layer_output_size, second_layer_output_size);
  // softmax5(mat_add4(mat_mul3(relu2(mat_add1(mat_mul0({}))))))
  auto udf_plan_builder = build_plan_udf(data, input_features_size, first_layer_output_size, second_layer_output_size);
  Opt optimizer(udf_plan_builder.planBuilder, clientSocket, data);
  optimizer.start();
  optimizer.mcts_optimizer();
  // mcts_optimizer(*(udf_plan_builder.planBuilder), clientSocket);
  // exec_plan_udf(udf_plan_builder, 1000, data.features, 4, 4);
  std::string test_action1 = "Merge2Single";
  std::string test_action2 = "Mul2JoinAgg";

  // auto plan_s1_builder_1 = rewriten_udf(*(udf_plan_builder.planBuilder), data, input_features_size, first_layer_output_size, second_layer_output_size, test_action1);
  // exec_plan_udf(plan_s1_builder_1, 1000, data.features, 4, 4);
  // auto plan_s2_builder_1 = rewriten_tradition(plan_s1_builder_1);
  // auto cost = cost_estimation(plan_s2_builder_1);
  // std::cout << cost << std::endl;

  // auto plan_s1_builder_2 = rewriten_udf(udf_plan_builder, test_action2);
  // auto plan_s2_builder_2 = rewriten_tradition(plan_s1_builder_2);
  // auto cost = cost_estimation(plan_s2_builder_2);
  // std::cout << cost << std::endl;

}


void rewrite_test_split2multi(int argc, char** argv){
  folly::init(&argc, &argv, false);
  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  parse::registerTypeResolver();
  const std::string kHiveConnectorId = "test-hive";
  auto hiveConnector =
      connector::getConnectorFactory(
          connector::hive::HiveConnectorFactory::kHiveConnectorName)
          ->newConnector(kHiveConnectorId, std::make_shared<core::MemConfig>());
  connector::registerConnector(hiveConnector);

  filesystems::registerLocalFileSystem();
  dwrf::registerDwrfReaderFactory();

  int input_features_size = 800;//597540
  int num_samples = 1000;
  int first_layer_output_size = 1024;
  int second_layer_output_size = 14588;
  auto data = data_generate(input_features_size, num_samples, first_layer_output_size, second_layer_output_size);
  auto featureArrayVector = maker.arrayVector<float>(data.features, REAL());
  auto inputRowVector = maker.rowVector({"v"}, {featureArrayVector});
  auto file = TempFilePath::create();
  ::MyFileTest myFile;
  auto config = std::make_shared<facebook::velox::dwrf::Config>();
  myFile.writeToFile(file->path, {inputRowVector}, config);

  core::PlanNodeId p0;

  std::vector<int> dimensions;
  dimensions.push_back(input_features_size);
  dimensions.push_back(first_layer_output_size);
  dimensions.push_back(second_layer_output_size);

  float* weights[2] = {data.weights[0], data.weights[1]};
  float* bias[2] = {data.bias[0], data.bias[1]};

  exec::registerVectorFunction(
    "torchDNN0",
    TorchDNN2Level::signatures(),
    std::make_unique<TorchDNN2Level>(weights, bias, dimensions)
  );

    // optimization::registerVectorFunction(
    //     "torchDNN0",
    //     TorchDNN::signatures(),
    //     std::make_unique<TorchDNN>(weights, bias, dimensions),
    //     {},
    //     true
    // );
  // Register reg;
  // reg.registerVectorFunction(
  //   "torchDNN0",
  //   TorchDNN::signatures(),
  //   std::make_unique<TorchDNN>(weights, bias, dimensions),
  //   {},
  //   true
  // );

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator)
                .tableScan(asRowType(inputRowVector->type()))
                .capturePlanNodeId(p0)
                .project({"torchDNN0(v)"})
                .planBuild();

  auto planNode = myPlan.planNode();
  CataLog cataLog;
  // std::shared_ptr<optimization::Split2MultiRewriteAction>
  //         myAction = std::make_shared<
  //             optimization::Split2MultiRewriteAction>();
  
  // myAction->apply(
  //         planNode, nullptr, maker, myPlan, pool_, planNodeIdGenerator, "torchDNN0");

      RuleManager ruleManager;
      PlanState planState(ruleManager);
      planState.getPossibleActions(planNode,cataLog);
      std::pair<std::string, std::string> testAction("torchdnn0", "Split2MultiRewriteAction");
      planState.takeAction(planNode, nullptr, maker, myPlan, pool_, planNodeIdGenerator, {testAction}, cataLog);
      planState.update(myPlan,cataLog);


  int numSplits = 8;
  int veloxThreads = 8;
  auto hiveSplits =  myFile.makeHiveConnectorSplits(file->path, numSplits, dwio::common::FileFormat::DWRF);

  std::shared_ptr<folly::Executor> executor_{
        std::make_shared<folly::CPUThreadPoolExecutor>(
            std::thread::hardware_concurrency())};

    std::shared_ptr<core::QueryCtx> queryCtx_{
        std::make_shared<core::QueryCtx>(executor_.get())};

    // queryCtx_->testingOverrideConfigUnsafe(
    //     {{core::QueryConfig::kPreferredOutputBatchBytes, "1000000"},
    //      {core::QueryConfig::kMaxOutputBatchRows, "10000"}});
    boost::interprocess::interprocess_semaphore semaphore(veloxThreads);

    auto task = exec::Task::create(
        "0",
        myPlan.planFragment(),
        0,
        queryCtx_,
        [&semaphore](RowVectorPtr result, ContinueFuture* /*unused*/) {
          if (result)
            semaphore.post();
          return exec::BlockingReason::kNotBlocked;
        });

    std::cout << "Hive splits:" << std::endl;

    for (auto& split : hiveSplits) {
      // std::cout << split->toString() << std::endl;
      task->addSplit(p0, exec::Split(std::move(split)));
    }
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();

    

    task->start(veloxThreads);

    task->noMoreSplits(p0);

    // Start task with 2 as maximum drivers and wait for execution to finish

    waitForFinishedDrivers(task);

    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();

    std::stringstream ss;


    std::cout << "Time for TorchDNN with Input Data (sec): "
              << std::endl;

    std::cout << ss.str()
              << (std::chrono::duration_cast<std::chrono::microseconds>(
                      end - begin)
                      .count()) /
            1000000.0
              << " secs" << std::endl;
  
}


void rewrite_test_merge2single(int argc, char** argv){
  folly::init(&argc, &argv, false);
  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  parse::registerTypeResolver();
  const std::string kHiveConnectorId = "test-hive";
  auto hiveConnector =
      connector::getConnectorFactory(
          connector::hive::HiveConnectorFactory::kHiveConnectorName)
          ->newConnector(kHiveConnectorId, std::make_shared<core::MemConfig>());
  connector::registerConnector(hiveConnector);

  filesystems::registerLocalFileSystem();
  dwrf::registerDwrfReaderFactory();

  int input_features_size = 800;//597540
  int num_samples = 1000;
  int first_layer_output_size = 1024;
  int second_layer_output_size = 14588;
  auto data = data_generate(input_features_size, num_samples, first_layer_output_size, second_layer_output_size);
  auto featureArrayVector = maker.arrayVector<float>(data.features, REAL());
  auto inputRowVector = maker.rowVector({"v"}, {featureArrayVector});
  auto file = TempFilePath::create();
  ::MyFileTest myFile;
  auto config = std::make_shared<facebook::velox::dwrf::Config>();
  myFile.writeToFile(file->path, {inputRowVector}, config);

  core::PlanNodeId p0;

  std::string compute =  NNBuilder()
                        .denseLayer(first_layer_output_size, input_features_size, data.weights[0], data.bias[0], NNBuilder::RELU)
                        .denseLayer(second_layer_output_size, first_layer_output_size, data.weights[1], data.bias[1], NNBuilder::SOFTMAX)
                        .build();

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator)
                .tableScan(asRowType(inputRowVector->type()))
                .capturePlanNodeId(p0)
                .project({fmt::format(compute, "v")}) 
                .planBuild();

  auto planNode = myPlan.planNode();
  
  // std::shared_ptr<optimization::Merge2SingleRewriteAction>
  //         myAction = std::make_shared<
  //             optimization::Merge2SingleRewriteAction>();
  
  // myAction->apply(
  //         planNode, nullptr, maker, myPlan, pool_, planNodeIdGenerator, "");


  int numSplits = 8;
  int veloxThreads = 8;
  auto hiveSplits =  myFile.makeHiveConnectorSplits(file->path, numSplits, dwio::common::FileFormat::DWRF);

  std::shared_ptr<folly::Executor> executor_{
        std::make_shared<folly::CPUThreadPoolExecutor>(
            std::thread::hardware_concurrency())};

    std::shared_ptr<core::QueryCtx> queryCtx_{
        std::make_shared<core::QueryCtx>(executor_.get())};

    // queryCtx_->testingOverrideConfigUnsafe(
    //     {{core::QueryConfig::kPreferredOutputBatchBytes, "1000000"},
    //      {core::QueryConfig::kMaxOutputBatchRows, "10000"}});
    boost::interprocess::interprocess_semaphore semaphore(veloxThreads);

    auto task = exec::Task::create(
        "0",
        myPlan.planFragment(),
        0,
        queryCtx_,
        [&semaphore](RowVectorPtr result, ContinueFuture* /*unused*/) {
          if (result)
            semaphore.post();
          return exec::BlockingReason::kNotBlocked;
        });

    std::cout << "Hive splits:" << std::endl;

    for (auto& split : hiveSplits) {
      // std::cout << split->toString() << std::endl;
      task->addSplit(p0, exec::Split(std::move(split)));
    }
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();

    

    task->start(veloxThreads);

    task->noMoreSplits(p0);

    // Start task with 2 as maximum drivers and wait for execution to finish

    waitForFinishedDrivers(task);

    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();

    std::stringstream ss;


    std::cout << "Time for TorchDNN with Input Data (sec): "
              << std::endl;

    std::cout << ss.str()
              << (std::chrono::duration_cast<std::chrono::microseconds>(
                      end - begin)
                      .count()) /
            1000000.0
              << " secs" << std::endl;
  
}

void rewrite_test_mul2joinagg(int argc, char** argv, int blocks){
  folly::init(&argc, &argv, false);
  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  parse::registerTypeResolver();
  const std::string kHiveConnectorId = "test-hive";
  auto hiveConnector =
      connector::getConnectorFactory(
          connector::hive::HiveConnectorFactory::kHiveConnectorName)
          ->newConnector(kHiveConnectorId, std::make_shared<core::MemConfig>());
  connector::registerConnector(hiveConnector);

  filesystems::registerLocalFileSystem();
  dwrf::registerDwrfReaderFactory();

  int input_features_size = 800;//597540
  int num_samples = 1000;
  int first_layer_output_size = 1024;
  int second_layer_output_size = 14588;
  auto data = data_generate(input_features_size, num_samples, first_layer_output_size, second_layer_output_size);
  auto featureArrayVector = maker.arrayVector<float>(data.features, REAL());
  auto inputRowVector = maker.rowVector({"v"}, {featureArrayVector});


  
  auto inputBlocks = ::create_input_block(num_samples*input_features_size, data.features, blocks);
  auto weightBlocks = ::create_weight_block(input_features_size*first_layer_output_size, data.weights[0], blocks);
  auto inputs = ::block_to_files(inputBlocks, blocks, 0);//0 denote values, 1 denote weight
  auto weights = ::block_to_files(weightBlocks, blocks, 1);

  auto file = TempFilePath::create();
  ::MyFileTest myFile;
  auto config = std::make_shared<facebook::velox::dwrf::Config>();
  myFile.writeToFile(file->path, {inputRowVector}, config);

  core::PlanNodeId p0;
  // core::PlanNodeId p1 = "2";
  // core::PlanNodeId p2 = "3";
  std::string compute =  NNBuilder()
                        .denseLayer(first_layer_output_size, input_features_size, data.weights[0], data.bias[0], NNBuilder::RELU)
                        .denseLayer(second_layer_output_size, first_layer_output_size, data.weights[1], data.bias[1], NNBuilder::SOFTMAX)
                        .build();

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator)
                .tableScan(asRowType(inputRowVector->type()))
                .capturePlanNodeId(p0)
                .project({fmt::format(compute, "v")}) 
                .planBuild();

  auto planNode = myPlan.planNode();
  core::PlanNodeId p1;
  core::PlanNodeId p2;
  // std::shared_ptr<optimization::Mul2JoinAggRewriteAction>
  //         myAction = std::make_shared<
  //             optimization::Mul2JoinAggRewriteAction>(blocks, &p1, &p2, inputs.schema, weights.schema);
  
  // myAction->apply(
  //         planNode, nullptr, maker, myPlan, pool_, planNodeIdGenerator, "");


  int veloxThreads = 4;
  auto inputHiveSplits = myFile.makeHiveConnectorSplits(inputs.paths);
  auto weightHiveSplits = myFile.makeHiveConnectorSplits(weights.paths);

  std::shared_ptr<folly::Executor> executor_{
        std::make_shared<folly::CPUThreadPoolExecutor>(
            std::thread::hardware_concurrency())};

    std::shared_ptr<core::QueryCtx> queryCtx_{
        std::make_shared<core::QueryCtx>(executor_.get())};

    queryCtx_->testingOverrideConfigUnsafe({{core::QueryConfig::kSpillEnabled, "true"}});
    // queryCtx_->testingOverrideConfigUnsafe(
    //     {{core::QueryConfig::kPreferredOutputBatchBytes, "1000000"},
    //      {core::QueryConfig::kMaxOutputBatchRows, "10000"}});
    boost::interprocess::interprocess_semaphore semaphore(veloxThreads*8);

    auto task = exec::Task::create(
        "0",
        myPlan.planFragment(), //replace
        0,
        queryCtx_,
        [&semaphore](RowVectorPtr result, ContinueFuture* /*unused*/) {
          if (result)
            semaphore.post();
          return exec::BlockingReason::kNotBlocked;
        });

    std::cout << "Hive splits:" << std::endl;
    task->start(veloxThreads);
    for (auto& split : inputHiveSplits) {
      semaphore.wait();
      // std::cout << split->toString() << std::endl;
      task->addSplit(p1, exec::Split(std::move(split)));
    }

    for (auto& split : weightHiveSplits) {
      semaphore.wait();
      // std::cout << split->toString() << std::endl;
      task->addSplit(p2, exec::Split(std::move(split)));
    }
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();

    

    // task->start(veloxThreads);

    task->noMoreSplits(p1);
    task->noMoreSplits(p2);

    // Start task with 2 as maximum drivers and wait for execution to finish

    waitForFinishedDrivers(task);

    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();

    std::stringstream ss;


    std::cout << "Time for TorchDNN with Input Data (sec): "
              << std::endl;

    std::cout << ss.str()
              << (std::chrono::duration_cast<std::chrono::microseconds>(
                      end - begin)
                      .count()) /
            1000000.0
              << " secs" << std::endl;
  
}

void rewrite_test_joinagg2mul(int argc, char** argv){
  folly::init(&argc, &argv, false);
  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  parse::registerTypeResolver();
  const std::string kHiveConnectorId = "test-hive";
  auto hiveConnector =
      connector::getConnectorFactory(
          connector::hive::HiveConnectorFactory::kHiveConnectorName)
          ->newConnector(kHiveConnectorId, std::make_shared<core::MemConfig>());
  connector::registerConnector(hiveConnector);

  filesystems::registerLocalFileSystem();
  dwrf::registerDwrfReaderFactory();

  int input_features_size = 800;//597540
  int num_samples = 1000;
  int first_layer_output_size = 1024;
  int second_layer_output_size = 14588;
  auto data = data_generate(input_features_size, num_samples, first_layer_output_size, second_layer_output_size);
  auto featureArrayVector = maker.arrayVector<float>(data.features, REAL());
  auto inputRowVector = maker.rowVector({"v"}, {featureArrayVector});

  auto valueSchema = asRowType(inputRowVector->type());
  
  auto inputBlocks = ::create_input_block(num_samples*input_features_size, data.features, 4);
  auto weightBlocks = ::create_weight_block(input_features_size*first_layer_output_size, data.weights[0], 4);
  auto inputs = ::block_to_files(inputBlocks, 4, 0);//0 denote values, 1 denote weight
  auto weights = ::block_to_files(weightBlocks, 4, 1);

  auto file = TempFilePath::create();
  ::MyFileTest myFile;
  auto config = std::make_shared<facebook::velox::dwrf::Config>();
  myFile.writeToFile(file->path, {inputRowVector}, config);

  core::PlanNodeId p0;
  // core::PlanNodeId p1 = "2";
  // core::PlanNodeId p2 = "3";
  std::string compute =  NNBuilder()
                        .denseLayer(first_layer_output_size, input_features_size, data.weights[0], data.bias[0], NNBuilder::RELU)
                        .denseLayer(second_layer_output_size, first_layer_output_size, data.weights[1], data.bias[1], NNBuilder::SOFTMAX)
                        .build();

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator)
                .tableScan(asRowType(inputRowVector->type()))
                .capturePlanNodeId(p0)
                .project({fmt::format(compute, "v")}) 
                .planBuild();

  auto planNode = myPlan.planNode();
  core::PlanNodeId p1;
  core::PlanNodeId p2;
  // std::shared_ptr<optimization::Mul2JoinAggRewriteAction>
  //         myAction = std::make_shared<
  //             optimization::Mul2JoinAggRewriteAction>(4, &p1, &p2, inputs.schema, weights.schema);
  
  // myAction->apply(
  //         planNode, nullptr, maker, myPlan, pool_, planNodeIdGenerator, ""); //rewrite mul2joinagg

  auto planNode2 = myPlan.planNode();
  core::PlanNodeId p3;
  // std::shared_ptr<optimization::JoinAgg2MulRewriteAction> 
  //         myAction2 = std::make_shared<optimization::JoinAgg2MulRewriteAction>(&p3, valueSchema);

  // myAction2->apply(planNode2, nullptr, maker, myPlan, pool_, planNodeIdGenerator, ""); //rewrite joinagg2mul


  int numSplits = 8;
  int veloxThreads = 8;
  auto hiveSplits =  myFile.makeHiveConnectorSplits(file->path, numSplits, dwio::common::FileFormat::DWRF);

  std::shared_ptr<folly::Executor> executor_{
        std::make_shared<folly::CPUThreadPoolExecutor>(
            std::thread::hardware_concurrency())};

    std::shared_ptr<core::QueryCtx> queryCtx_{
        std::make_shared<core::QueryCtx>(executor_.get())};

    // queryCtx_->testingOverrideConfigUnsafe(
    //     {{core::QueryConfig::kPreferredOutputBatchBytes, "1000000"},
    //      {core::QueryConfig::kMaxOutputBatchRows, "10000"}});
    boost::interprocess::interprocess_semaphore semaphore(veloxThreads);

    auto task = exec::Task::create(
        "0",
        myPlan.planFragment(),
        0,
        queryCtx_,
        [&semaphore](RowVectorPtr result, ContinueFuture* /*unused*/) {
          if (result)
            semaphore.post();
          return exec::BlockingReason::kNotBlocked;
        });

    std::cout << "Hive splits:" << std::endl;

    for (auto& split : hiveSplits) {
      // std::cout << split->toString() << std::endl;
      task->addSplit(p3, exec::Split(std::move(split)));
    }
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();

    

    task->start(veloxThreads);

    task->noMoreSplits(p3);

    // Start task with 2 as maximum drivers and wait for execution to finish

    waitForFinishedDrivers(task);

    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();

    std::stringstream ss;


    std::cout << "Time for TorchDNN with Input Data (sec): "
              << std::endl;

    std::cout << ss.str()
              << (std::chrono::duration_cast<std::chrono::microseconds>(
                      end - begin)
                      .count()) /
            1000000.0
              << " secs" << std::endl;
  
}

int main(int argc, char** argv) {
    memory::MemoryManager::initialize({});
    // test_optimizer_demo(argc, argv);
    // test_optimizer_mcts(argc, argv);
    rewrite_test_split2multi(argc, argv);
    // rewrite_test_merge2single(argc, argv);
    // rewrite_test_mul2joinagg(argc, argv, 4);
    // rewrite_test_joinagg2mul(argc, argv);
    // test_connecter();
}
