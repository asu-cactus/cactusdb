#pragma once
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <memory>
#include <cmath>
#include <stdlib.h>
#include <string>
#include <dirent.h>
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/ml_functions/functions.h"
#include "velox/vector/tests/utils/VectorTestBase.h"
#include "velox/ml_functions/DecisionTree.h"

using namespace std;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::memory;

namespace ml{

#define MAX_NUM_TREES 1600

class Forest;
   
typedef std::shared_ptr<Forest> ForestPtr;

class Forest {

public:
 
    Node forest[MAX_NUM_TREES][MAX_NUM_NODES_PER_TREE];

    int numTrees;

    bool isClassification;

    Forest() {}

    Forest(std::string pathToFolder, bool isClassification):isClassification{isClassification} {

	this->constructForestFromFolder(pathToFolder);

    }

    static void vectorizeForestFolder(std::string pathToFolder, std::vector<std::string> & pathVector) {

        if (pathToFolder[pathToFolder.length()-1] != '/') {

            pathToFolder = pathToFolder + std::string("/");

        }

        std::cout << pathToFolder << std::endl;

        DIR *dr = opendir(pathToFolder.c_str());

        struct dirent * file = NULL;

        while ((file = readdir(dr)) != NULL) {

            std::cout << file->d_name << std::endl;

            if ((strcmp(file->d_name, ".") != 0)&&(strcmp(file->d_name, ".."))) {

                std::string path = pathToFolder + std::string(file->d_name);

		std::cout << path << std::endl;

                pathVector.push_back(path);

            }
        }

        closedir(dr);
    
    }

    void constructForestFromFolder(std::string pathToFolder) {

        std::vector<std::string> treePaths;

        vectorizeForestFolder(pathToFolder, treePaths);

        constructForestFromPaths(treePaths);
    }

    void constructForestFromPaths(std::vector<std::string> &treesPathIn) {

        this->numTrees = treesPathIn.size();

        for (int n = 0; n < numTrees; ++n) {

            std::cout << treesPathIn[n] << std::endl;

            Tree::constructTreeFromPath(treesPathIn[n], &(forest[n][0]));
        }

        // STATS ABOUT THE FOREST
        std::cout << "Number of trees in the forest: " << numTrees << std::endl;
     }

     inline void predict(VectorPtr& input, std::vector<float>& resultVector, int numInputs, int numFeatures) {

        //get the input features
        auto inputFeatures = input->as<ArrayVector>()->elements();

        float* inputValues = inputFeatures->values()->asMutable<float>();

        float * outData = resultVector.data();

        for (int rowIndex = 0; rowIndex < numInputs; rowIndex++) {

         int curBase = rowIndex * numFeatures;

	 float accumulatedResult = 0.0;
         
	 std::cout << curBase << std::endl;
	 
	 for (int treeIndex = 0; treeIndex < numTrees; treeIndex++) {
         
	 	int curIndex = 0;

 		Node * tree = forest[treeIndex];

                while (!tree[curIndex].isLeaf) {

                    const float featureValue = inputValues[curBase + tree[curIndex].indexID];

                    std::cout << curIndex << "--" << tree[curIndex].indexID << "--" << featureValue << "--" << tree[curIndex].threshold << std::endl;

                    curIndex = featureValue < tree[curIndex].threshold ? tree[curIndex].leftChild : tree[curIndex].rightChild;

                 }

                 accumulatedResult += (float)(tree[curIndex].leafValue);

	 }

	 accumulatedResult /= numTrees;

	 if (isClassification) {
	 
	 	 accumulatedResult = (accumulatedResult > 0.0) ? 1.0 : 0.0;

	 }

         outData[rowIndex] = accumulatedResult; 

         std::cout << rowIndex << ":" <<(float)(accumulatedResult) << std::endl;

	 }

        
      }

   };




class ForestPrediction : public MLFunction {

  public:

  ForestPrediction(std::string forestPath, int numFeatures, bool isClassification) {

   this->forest = std::make_shared<Forest>(forestPath, isClassification);

   this->numFeatures = numFeatures;

  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {

    BaseVector::ensureWritable(rows, type, context.pool(), output);

    int numInputs = rows.size();

    std::vector<float> resultVector (numInputs);

    this->forest->predict(args[0], resultVector, numInputs, this->numFeatures);

    VectorMaker maker{context.pool()};

    output = maker.flatVector<float>(resultVector, REAL());

  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {

    return {exec::FunctionSignatureBuilder()
                .argumentType("array(REAL)")
                .returnType("REAL")
                .build()};
  }

  // TODO: add get and set for bias or we have a better way to store the two
  // parameters in a single file
  float* getTensor() const override {

      	  return new float[0]; //will this lead to memory leak?
  
  }

  static std::string getName() {

       return "tree_predict";

  }

 private:

  ForestPtr forest;

  int numFeatures;

};

}
