#pragma once
#include "velox/cost_model/CostModel.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;



// perhaps only estimate cost for changes made and not the whole tree again
// mcts can retrun the nodeId of the changes made (it makes one change only)
// we need a way to get the next node for any node
// 
class CostEstimator {

    protected:
        std::unique_ptr<CostModel> costModel;

    public:
        virtual ~CostEstimator() = default;

        // pass different cost models
        // use a simple one for quick testing
   
        CostEstimator(std::unique_ptr<CostModel> model) : costModel(std::move(model)) {}

        virtual CostEstimate estimateCost(std::shared_ptr<const core::PlanNode>& plan) const = 0; 
};


class SimpleCostEstimator : public CostEstimator {

    public:

        SimpleCostEstimator(std::unique_ptr<CostModel> model) : CostEstimator(std::move(model)) {}

        CostEstimate estimateCost(std::shared_ptr<const core::PlanNode>& plan) const override {
            LOG(INFO) << fmt::format("[INFO] SimpleCostEstimator estimateCost Start. Node: {}, # Sources: {}", plan->name(), plan->sources().size()) << std::endl;
            if(!plan)
                return CostEstimate(0,0,0);
            std::vector<Source> sources; 
            // total cost of all the sources for the current node
            // this will be added to the cost of the current node
            // to get the total cost so far including the current node

            // hashjoin
            float srcCost = 0.0; // the current node
            for (auto source : plan->sources()) {
                LOG(INFO) << fmt::format("[INFO] Current iterated plan source: {}", source->name()) << std::endl;
                CostEstimate estimate =  estimateCost(source);
                
                srcCost += estimate.cost;
                LOG(INFO) << fmt::format("[INFO] Finished estimateCost for source: {} of node: {}, outputRows: {}, outputCols: {}, cost: {}, accumulated cost: {}", source->name(), plan->name(), estimate.outputRows, estimate.outputCols, estimate.cost, srcCost) << std::endl;
                std::shared_ptr<OutputStat> stat = std::make_shared<OutputStat>(OutputStat(estimate.outputRows , estimate.outputCols));
                // inputs to the current node
                // output from previous step is input to the current node
                // hence the stat is the output stat of the sources
                sources.push_back(Source(std::string(plan->name()), Source::Type::NODE, stat));
            }
            CostEstimate estimate = costModel->getCost(plan, sources);
            // total cost so far
            estimate.cost += srcCost;
            LOG(INFO) << fmt::format("[INFO] SimpleCostEstimator estimateCost Finished. Node: {} estimate.cost: {}", plan->name(), estimate.cost) << std::endl;
            return estimate;
    }
};

