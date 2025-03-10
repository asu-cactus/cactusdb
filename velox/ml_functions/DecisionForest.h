/*
 * Copyright (c) 2025 ASU Cactus Lab.
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
/**
 * @class Forest
 * @brief A class that represents a random forest model for prediction tasks.
 *
 * This class provides functionality to construct a random forest from a folder of tree files
 * and make predictions using the forest.
 */
class Forest {
public:
    Node forest[MAX_NUM_TREES][MAX_NUM_NODES_PER_TREE]; ///< Array of trees in the forest.
    int numTrees; ///< Number of trees in the forest.
    bool isClassification; ///< Flag indicating whether the forest is used for classification.

    /**
     * @brief Default constructor.
     */
    Forest() {}

    /**
     * @brief Constructor that initializes the forest from a folder of tree files.
     *
     * @param pathToFolder The path to the folder containing tree files.
     * @param isClassification Flag indicating whether the forest is used for classification.
     */
    Forest(std::string pathToFolder, bool isClassification)
        : isClassification{isClassification} {
        this->constructForestFromFolder(pathToFolder);
    }

    /**
     * @brief Scans a folder and collects paths to tree files.
     *
     * @param pathToFolder The path to the folder containing tree files.
     * @param pathVector A vector to store the paths of tree files.
     */
    static void vectorizeForestFolder(
        std::string pathToFolder,
        std::vector<std::string>& pathVector) {
    if (pathToFolder[pathToFolder.length() - 1] != '/') {
      pathToFolder = pathToFolder + std::string("/");
    }

    DIR* dr = opendir(pathToFolder.c_str());

    struct dirent* file = NULL;

    while ((file = readdir(dr)) != NULL) {
      if ((strcmp(file->d_name, ".") != 0) && (strcmp(file->d_name, ".."))) {
        std::string path = pathToFolder + std::string(file->d_name);

        pathVector.push_back(path);
      }
    }

    closedir(dr);
    }

    /**
     * @brief Constructs the forest from a folder of tree files.
     *
     * @param pathToFolder The path to the folder containing tree files.
     */
    void constructForestFromFolder(std::string pathToFolder) {
    std::vector<std::string> treePaths;

    vectorizeForestFolder(pathToFolder, treePaths);

    constructForestFromPaths(treePaths);
    }

    /**
     * @brief Constructs the forest from a list of tree file paths.
     *
     * @param treesPathIn A vector of paths to tree files.
     */
    void constructForestFromPaths(std::vector<std::string>& treesPathIn) {
    this->numTrees = treesPathIn.size();

    for (int n = 0; n < numTrees; ++n) {
      Tree::constructTreeFromPath(treesPathIn[n], &(forest[n][0]));
    }

    // STATS ABOUT THE FOREST
    LOG(INFO)
        << "[Forest-constructForestFromPaths] Number of trees in the forest: "
        << numTrees << std::endl;
    }

    /**
     * @brief Makes predictions using the forest.
     *
     * @param input The input vector containing feature values.
     * @param resultVector A vector to store the prediction results.
     * @param numInputs The number of input samples.
     * @param numFeatures The number of features in each input sample.
     */
    inline void predict(
        VectorPtr& input,
        std::vector<float>& resultVector,
        int numInputs,
        int numFeatures) {
    auto inputFeatures = input->as<ArrayVector>()->elements();

    float* inputValues = inputFeatures->values()->asMutable<float>();

    float* outData = resultVector.data();

    for (int rowIndex = 0; rowIndex < numInputs; rowIndex++) {
      int curBase = rowIndex * numFeatures;

      float accumulatedResult = 0.0;

      for (int treeIndex = 0; treeIndex < numTrees; treeIndex++) {
        int curIndex = 0;

        Node* tree = forest[treeIndex];

        while (!tree[curIndex].isLeaf) {
          const float featureValue =
              inputValues[curBase + tree[curIndex].indexID];

          curIndex = featureValue < tree[curIndex].threshold
              ? tree[curIndex].leftChild
              : tree[curIndex].rightChild;
        }

        accumulatedResult += (float)(tree[curIndex].leafValue);
      }

      accumulatedResult /= numTrees;

      if (isClassification) {
        accumulatedResult = (accumulatedResult > 0.0) ? 1.0 : 0.0;
      }

      outData[rowIndex] = accumulatedResult;
    }
    }
};
/**
 * @class ForestPrediction
 * @brief A class that implements a random forest prediction function, inheriting from MLFunction.
 *
 * This class provides functionality to load a random forest model and make predictions using it.
 */
class ForestPrediction : public MLFunction {
public:
    /**
     * @brief Constructor that initializes the random forest prediction function.
     *
     * @param forestPath The path to the folder containing the forest model.
     * @param numFeatures The number of features in each input sample.
     * @param isClassification Flag indicating whether the forest is used for classification.
     */
    ForestPrediction(
        std::string forestPath,
        int numFeatures,
        bool isClassification) {
    if (!std::filesystem::exists(forestPath)) {
      throw std::runtime_error(
          "[ForestPrediction] Path not exists: " + forestPath);
    }

    this->forest = std::make_shared<Forest>(forestPath, isClassification);

    this->numFeatures = numFeatures;

    this->forestPath = forestPath;

    this->isClassification = isClassification;
    }

    /**
     * @brief Applies the random forest prediction function to the input array.
     *
     * This method processes the input array, makes predictions using the random forest,
     * and stores the result in the output vector.
     *
     * @param rows A SelectivityVector specifying the rows to process.
     * @param args A vector of input arguments (e.g., the input array).
     * @param type The type of the output vector.
     * @param context The execution context.
     * @param output The output vector where the prediction results will be stored.
     */
    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    int numInputs = rows.size();

    std::vector<float> resultVector(numInputs);

    this->forest->predict(args[0], resultVector, numInputs, this->numFeatures);

    VectorMaker maker{context.pool()};

    output = maker.flatVector<float>(resultVector, REAL());
    }

    /**
     * @brief Returns the function signatures supported by this class.
     *
     * @return A vector of shared pointers to FunctionSignature objects.
     */
    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                    .argumentType("array(REAL)")
                    .returnType("REAL")
                    .build()};
    }

    /**
     * @brief Returns the tensor associated with this function.
     *
     * @return A pointer to an empty float array (no weights for ForestPrediction).
     */
    float* getTensor() const override {
        return new float[0];
    }

    /**
     * @brief Returns the name of the function.
     *
     * @return The name of the function as a string ("tree_predict").
     */
    static std::string getName() {
        return "tree_predict";
    }

    /**
     * @brief Returns the number of features in each input sample.
     *
     * @return The number of features.
     */
    int getNumFeatures() {
        return numFeatures;
    }

    /**
     * @brief Returns the path to the forest model folder.
     *
     * @return The path to the forest model folder.
     */
    std::string& getForestPath() {
        return this->forestPath;
    }

    /**
     * @brief Returns whether the forest is used for classification.
     *
     * @return True if the forest is used for classification, false otherwise.
     */
    bool getClassification() {
        return this->isClassification;
    }

private:
    ForestPtr forest; ///< Pointer to the random forest model.
    int numFeatures; ///< Number of features in each input sample.
    std::string forestPath; ///< Path to the forest model folder.
    bool isClassification; ///< Flag indicating whether the forest is used for classification.
