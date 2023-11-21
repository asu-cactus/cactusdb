
struct DataFrame {
    std::vector<std::vector<float>> features;
    std::vector<float*> weights;
    std::vector<float*> bias;
    float* featuresFloat;
};

struct PlanBuilderExec{
  std::shared_ptr<PlanBuilder> planBuilder;
  std::vector<core::PlanNodeId> p;
  PlanBuilderExec(std::shared_ptr<PlanBuilder> builder, std::vector<core::PlanNodeId> ids):planBuilder(builder), p(ids){}
};

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

PlanBuilderExec build_plan_udf(DataFrame data, int features, int first_layer, int second_layer, int memoryLimit, std::vector<std::vector<float>> feature, int splitsNum, int threadsNum){
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
  std::shared_ptr<PlanBuilder> planBuilderShared = std::make_shared<PlanBuilder>(planBuilder);
  PlanBuilderExec planBuilderExec(planBuilderShared, {p0});
  return planBuilderExec;
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
          ->newConnector(kHiveConnectorId, nullptr);
    connector::registerConnector(hiveConnector);

    filesystems::registerLocalFileSystem();
    dwrf::registerDwrfReaderFactory();
    OptimizerRewriter optimizer;
    int input_features_size = 597540;
    int num_samples = 1000;

    int first_layer_output_size = 1024;
    int second_layer_output_size = 14588;

    int memory_limit_udf = 1000;//mb
    int splits_num_udf = 4;
    int threads_num_udf = 4;

    int memory_limit_rela = 100000;//mb
    int splits_num_rela = 4;
    int threads_num_rela = 4;

    auto data = data_generate(input_features_size, num_samples, first_layer_output_size, second_layer_output_size);
    auto udf_plan_builder = build_plan_udf(data, input_features_size, first_layer_output_size, second_layer_output_size, memory_limit_udf, data.features, splits_num_udf, threads_num_udf);
    auto new_plan = optimizer.build(udf_plan_builder);
    exec_plan_relational(relational_plan.planBuilderExec, memory_limit_rela, relational_plan.inputsPaths, relational_plan.weightPaths, threads_num_rela);
}


int main(int argc, char** argv) {
    test_optimizer_demo(argc, argv);
}