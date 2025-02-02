#define EIGEN_USE_BLAS
#include <folly/init/Init.h>
#include "velox/common/base/Fs.h"
#include "velox/connectors/tpch/TpchConnector.h"
#include "velox/connectors/tpch/TpchConnectorSplit.h"
#include "velox/core/Expressions.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/Expr.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/Expressions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"
#include "velox/tpch/gen/TpchGen.h"
#include "velox/vector/tests/utils/VectorTestBase.h"
#include "velox/expression/VectorFunction.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include <Eigen/Dense>
#include <cblas.h>
#include <chrono>
#include <torch/torch.h>
#include "velox/exec/Task.h"
#include "velox/ml_functions/NNBuilder.h"
#include <fstream>
#include <sstream>
#include <string>
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/ml_functions/UtilFunction.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;

class CNNTest : public HiveConnectorTestBase {
    public : 
        CNNTest(){
             // Register Presto scalar functions.
            functions::prestosql::registerAllScalarFunctions();

            // Register Presto aggregate functions.
            aggregate::prestosql::registerAllAggregateFunctions();

            // Register type resolver with DuckDB SQL parser.
            parse::registerTypeResolver();

            // HiveConnectorTestBase::SetUp();
            parquet::registerParquetReaderFactory();

            const std::string kHiveConnectorId = "test-hive";

            auto hiveConnector =
                connector::getConnectorFactory(
                    connector::hive::HiveConnectorFactory::kHiveConnectorName)
                    ->newConnector(
                        kHiveConnectorId, std::make_shared<core::MemConfig>());
            connector::registerConnector(hiveConnector);

        }

        ~CNNTest() {}

//         std::vector<std::shared_ptr<facebook::velox::RowVector>, std::allocator<std::shared_ptr<facebook::velox::RowVector>>> runtestidmeta();
//         std::vector<std::shared_ptr<facebook::velox::RowVector>, std::allocator<std::shared_ptr<facebook::velox::RowVector>>> runtestidlabel();
//         void runjointest(std::vector<std::shared_ptr<facebook::velox::RowVector>, std::allocator<std::shared_ptr<facebook::velox::RowVector>>>
//  idmeta_data, std::vector<std::shared_ptr<facebook::velox::RowVector>, std::allocator<std::shared_ptr<facebook::velox::RowVector>>>
//  idlabel_data);
        FlatVectorPtr<float> get_tensor(std::ifstream& file, int size, int lines);
        FlatVectorPtr<float> get_tensor(VectorMaker& m, std::ifstream& file, int size, int lines);
        void run(int numDriver, int memoryPoolSizeMB, int spillMemThresholdMB, bool enableSpill, int repeatRun);
        void test_conv2d();
        void fraud_detection_query();


        void TestBody() override {}

         void SetUp() override {
             HiveConnectorTestBase::SetUp(); // Call base class setup if needed
         }

        void TearDown() override {
            HiveConnectorTestBase::TearDown(); // Call base class teardown if needed
        }

        std::shared_ptr<memory::MemoryPool> pool_{
            memory::MemoryManager::getInstance()->addLeafPool()};
        VectorMaker maker{pool_.get()};

        // int numSamples = 10;
    
};

void CNNTest::run(int numDriver, int memoryPoolSizeMB, int spillMemThresholdMB, bool enableSpill, int repeatRun) {
  //  test_mat_mul();
  //  test_mat_add();
  //  test_relu();
  //  test_softmax()
  // test_argmax();
  //  test_dense_layer();
  //  test_torch_dense_layer_multithreading();
  //  test_mnist();
  //  test_multithreading();
  //  test_multithreading_oom();
  //  test_batching();
  //  test_conv2d();
  //  test_deep_bench_conv1();
  //  test_land_cover_conv3();
  //  test_spill(numDriver, memoryPoolSizeMB, spillMemThresholdMB, enableSpill, repeatRun);
  //  mytest();
  //  test_mnist_multithreading();
  //  test_mnist_oom_weights();
  // test_torch_dense_layer();
  // test_complex_torchnn();
    fraud_detection_query();
}

FlatVectorPtr<float> CNNTest::get_tensor(std::ifstream& file, int size, int lines){
    return get_tensor(maker,file,size,lines);
}

FlatVectorPtr<float> CNNTest::get_tensor(VectorMaker& m, std::ifstream& file, int size, int lines){
    //std::cout << "Loading tensor of size " << size << std::endl;
    FlatVectorPtr<float> tensor = m.flatVector<float>(size);
    int index = 0;
    std::string line;
    int ln = 0;
    while (lines--) { // Read a line from the file
        std::getline(file, line);
        std::istringstream iss(line); // Create an input string stream from the line
        std::string numberStr;
        while (std::getline(iss, numberStr, ',')) { // Read each number separated by comma
            float number = std::stof(numberStr);    // Convert the string to float
            tensor->set(index++, number);
        }
        
    }
    return tensor;
}

void CNNTest::test_conv2d() {

    std::ifstream weights_file("/home/cnn_weights.txt"); 
    std::ifstream bias_file("/home/cnn_bias.txt"); 
    std::ifstream test_file("/home/test_file.txt"); 

    // read inputs
    int input_dims[] = {32,32,1};
    int input_size = input_dims[0] * input_dims[1] * input_dims[2]; //32*32 = 1024
    int num_samples = 10;

    FlatVectorPtr<float> input = get_tensor(test_file, input_size * num_samples, num_samples * input_dims[2]); // 1024 in 1 line  
    float* data = input->values()->asMutable<float>();

    std::vector<std::vector<float>> featureVectors;
    for(int i=0, cursor = 0; i < num_samples; i++, cursor += input_size){
      std::vector<float> featureVector(data + cursor, data + cursor + input_size);
      featureVectors.push_back(featureVector);
    }

    auto featureArrayVector = maker.arrayVector<float>(featureVectors, REAL());
    auto inputRowVector = maker.rowVector({"x"}, {featureArrayVector});



    // Layer 1 Conv2d layer
    int cnn_layer1_filters = 6; //64; // 6 out channels 
    int cnn_layer1_filter_dims[] = {5,5,1}; //{3,3,1}; // height * width * channels
    int weights1_size = cnn_layer1_filter_dims[0] * cnn_layer1_filter_dims[1] * cnn_layer1_filter_dims[2] * cnn_layer1_filters; // 150
    int dims1[] = {cnn_layer1_filters, cnn_layer1_filter_dims[0], cnn_layer1_filter_dims[1], cnn_layer1_filter_dims[2], input_dims[0], input_dims[1]}; 
    FlatVectorPtr<float> weights_1 = get_tensor(weights_file, weights1_size, cnn_layer1_filters * cnn_layer1_filter_dims[2]); // (file, tensor_size = 150, lines = 6)
    FlatVectorPtr<float> bias_1 = get_tensor(bias_file, cnn_layer1_filters, 1); // (6,1)
    
    // Maxpool layer

    // Layer 2 Conv2d layer
    int cnn_layer2_filters = 16;//64; // 6 out channels 
    int cnn_layer2_filter_dims[] = {5,5,6};//{3,3,1}; // height * width * channels
    int weights2_size = cnn_layer2_filter_dims[0] * cnn_layer2_filter_dims[1] * cnn_layer2_filter_dims[2] * cnn_layer2_filters; // 2400
    int dims2[] = {cnn_layer2_filters, cnn_layer2_filter_dims[0], cnn_layer2_filter_dims[1], cnn_layer2_filter_dims[2], 14, 14}; 
    FlatVectorPtr<float> weights_2 = get_tensor(weights_file, weights2_size, cnn_layer2_filters * cnn_layer2_filter_dims[2]); // (file, tensor_size = 150, lines = 6)
    FlatVectorPtr<float> bias_2 = get_tensor(bias_file, cnn_layer2_filters, 1); // (6,1)

    // MAxpool Layer 

    // 3 linear layers

    int input3_size = 120; // num_features
    int layer3_size = 400; // num units in hidden layer 2
    int layer4_size = 84;
    int layer5_size = 2;


    FlatVectorPtr<float> weights_3 = get_tensor(weights_file, layer3_size * input3_size, input3_size);
    FlatVectorPtr<float> bias_3 = get_tensor(bias_file, layer3_size, 1);
    FlatVectorPtr<float> weights_4 = get_tensor(weights_file, layer4_size * layer3_size, layer3_size);
    FlatVectorPtr<float> bias_4 = get_tensor(bias_file, layer4_size, 1);
    FlatVectorPtr<float> weights_5 = get_tensor(weights_file, layer5_size * layer4_size, layer4_size);
    FlatVectorPtr<float> bias_5 = get_tensor(bias_file, layer5_size, 1);

    float* w3 =  weights_3->values()->asMutable<float>();
    float* w4 =  weights_4->values()->asMutable<float>();
    float* w5 =  weights_5->values()->asMutable<float>();
    // std::cout << "w1 : ";
    // int weights_1_size = weights_1->size();
    // for(int i = 0;i<weights_1_size;i++){
    //     std::cout << w1[i] << " ";
    // }
    // std::cout << "\n";

    weights_file.close();
    bias_file.close();

    float* bias_1_values = bias_1->values()->asMutable<float>();
    float* bias_2_values = bias_2->values()->asMutable<float>();
    float* bias_3_values = bias_3->values()->asMutable<float>();
    float* bias_4_values = bias_4->values()->asMutable<float>();
    float* bias_5_values = bias_5->values()->asMutable<float>();
    

    FlatVectorPtr<float> bias_3_mat = maker.flatVector<float>(num_samples * layer3_size);
    for(int i=0; i < bias_3_mat->size(); i++)
      bias_3_mat->set(i, bias_3_values[i%layer3_size]);
    
    FlatVectorPtr<float> bias_4_mat = maker.flatVector<float>(num_samples * layer4_size);
    for(int i=0; i < bias_4_mat->size(); i++)
      bias_4_mat->set(i, bias_4_values[i%layer4_size]);

    FlatVectorPtr<float> bias_5_mat = maker.flatVector<float>(num_samples * layer5_size);
    for(int i=0; i < bias_5_mat->size(); i++)
      bias_5_mat->set(i, bias_5_values[i%layer5_size]);
    
    std::string compute =  NNBuilder()
                          .convLayer(cnn_layer1_filters, dims1, weights_1->values()->asMutable<float>(), 
                            bias_1->values()->asMutable<float>(), NNBuilder::RELU)
                          .maxPoolLayer(2, 28, 28)
                          .convLayer(cnn_layer2_filters, dims2, weights_2->values()->asMutable<float>(), 
                            bias_2->values()->asMutable<float>(), NNBuilder::RELU)
                          .maxPoolLayer(2, 14, 14)
                          .denseLayer(layer3_size ,input3_size, weights_3->values()->asMutable<float>(), 
                            bias_3_mat->values()->asMutable<float>(), NNBuilder::RELU)
                          .denseLayer(layer4_size ,layer3_size, weights_4->values()->asMutable<float>(), 
                            bias_4_mat->values()->asMutable<float>(), NNBuilder::RELU)
                          .denseLayer(layer5_size ,layer4_size, weights_5->values()->asMutable<float>(), 
                            bias_5_mat->values()->asMutable<float>(), NNBuilder::SOFTMAX)
                          .build();

    std::cout << fmt::format(compute, "x") << std::endl; // softmax5(mat_add4(mat_mul3(relu2(mat_add1(mat_mul0({}))))))
  //   auto plan = exec::test::PlanBuilder(pool_.get())
  //                 .values({inputRowVector})
  //                 .project({fmt::format(compute, "x")}) 
	// 	              .planNode();
  //   auto results = exec::test::AssertQueryBuilder(plan).copyResults(pool_.get());
    
  //  std::cout << "Results:" << results->toString() << std::endl;
  //  std::cout << results->toString(0, results->size()) << std::endl;

}



void CNNTest::fraud_detection_query() {

  //register all the ml_related functions
  test_conv2d();
  exec::registerVectorFunction(
      "classify",
      Classify::signatures(),
      std::make_unique<Classify>());
  exec::registerVectorFunction(
      "convert_string_to_float_array",
      ConvertStringToFloatArray::signatures(),
      std::make_unique<ConvertStringToFloatArray>());
  
  // Define the schema for the metadata and image data
  auto metaRowType = ROW(
      {"id", "name", "address", "birthday", "gender", "ethnicity", "class",
       "issue_date", "expire_date", "height", "weight", "eye_color",
       "hair_color", "is_donor", "is_veteran", "license_number"},
      {VARCHAR(), VARCHAR(), VARCHAR(), VARCHAR(), VARCHAR(), VARCHAR(),
       VARCHAR(), VARCHAR(), VARCHAR(), VARCHAR(), VARCHAR(), VARCHAR(),
       VARCHAR(), BOOLEAN(), BOOLEAN(), VARCHAR()});

  auto imageRowType = ROW({"name", "imagedata"}, {VARCHAR(), VARCHAR()});

  auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();

  core::PlanNodeId metaScanId;
  core::PlanNodeId imageScanId;

  // query
  auto read_meta_plan = PlanBuilder(planNodeIdGenerator)
                          .tableScan(metaRowType)
                          .capturePlanNodeId(metaScanId)
                          .hashJoin(
                              {"id"}, {"name"},
                              PlanBuilder(planNodeIdGenerator)
                                  .tableScan(imageRowType)
                                  .capturePlanNodeId(imageScanId)
                                  .project({"name", "imagedata"}) // Select specific columns
                                  .planNode(),
                                  "",
                                  {"id","gender","imagedata"},
                                  core::JoinType::kInner
                          )
                          .project({"id", "gender", "imagedata",
                                    "classify(softmax16(mat_add15(mat_mul14(relu13(mat_add12(mat_mul11(relu10(mat_add9(mat_mul8(max_pool7(relu6(vec_scal_add5(conv2d4(max_pool3(relu2(vec_scal_add1(conv2d0(convert_string_to_float_array(imagedata))))))))))))))))))) as softmax_output",
                          })
                          .partialAggregation(
                              {"gender"}, {"sum(softmax_output) as fraud_count"})
                          .localPartition({})
                          .finalAggregation()
                          .planNode();


  int numSplitsPerFile = 1;
  auto metaSplits = HiveConnectorTestBase::makeHiveConnectorSplits(
    {"file:./data/idmeta.parquet"},
    numSplitsPerFile,
    dwio::common::FileFormat::PARQUET);
  auto imageSplits = HiveConnectorTestBase::makeHiveConnectorSplits(
    {"file:./data/idimage_1000.parquet"},
    numSplitsPerFile,
    dwio::common::FileFormat::PARQUET);

  auto start = std::chrono::high_resolution_clock::now();
  RowVectorPtr metaResults = AssertQueryBuilder(read_meta_plan).split(metaScanId,exec::Split(metaSplits[0])).split(imageScanId,exec::Split(imageSplits[0])).copyResults(pool_.get());
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
  //RowVectorPtr imageResults = AssertQueryBuilder(read_image_plan).copyResults(pool_.get());

  // Process the results
  std::cout << "Grouped and Aggregated Results:" << std::endl;
  for (size_t i = 0; i < metaResults->size(); ++i) {
    // Extract gender and the aggregated user count (usercount is the aggregation of ids)
    auto gender = metaResults->childAt(0)->asFlatVector<StringView>()->valueAt(i).str();
    auto userCount = metaResults->childAt(1)->asFlatVector<int64_t>()->valueAt(i); // assuming count is stored as int64_t
    
    std::cout << "Gender: " << gender << ", User Count: " << userCount << std::endl;
  }

  // std::cout << "Image Results:" << std::endl;
  // for (size_t i = 0; i < imageResults->size(); ++i) {
  //   auto name = imageResults->childAt(0)->asFlatVector<StringView>()->valueAt(i);
  //   auto imagedata = imageResults->childAt(1)->asFlatVector<StringView>()->valueAt(i);
  //   std::cout << "Name: " << name << ", Image Data: " << imagedata << std::endl;
  // }
}

DEFINE_bool(spill, false, "Whether enable spilling");
DEFINE_int32(spill_threshold, 30, "Set spill memory threshold");
DEFINE_int32(memory_pool, 100, "Set memory pool size");
DEFINE_int32(repeat, 5, "Number of repeat run");
DEFINE_int32(num_driver, 1, "Number of driver");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});

  bool enableSpill = FLAGS_spill;
  int memoryPoolSizeMB = FLAGS_memory_pool;
  int spillMemoryThresholdMB = FLAGS_spill_threshold;
  int repeatRun = FLAGS_repeat;
  int numDriver = FLAGS_num_driver;
  CNNTest demo;
  demo.run(numDriver, memoryPoolSizeMB, spillMemoryThresholdMB, enableSpill, repeatRun);
}