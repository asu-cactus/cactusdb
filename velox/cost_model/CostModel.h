#pragma once
#include<vector>
#include "velox/cost_model/Source.h"
#include "velox/cost_model/Catalog.h"
#include "velox/optimizer/CataLog.h"
#include "velox/cost_model/UdfCostCoefficient.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;


class CostModel {

    public:

        enum PlanNodeType {
            Filter,
            Project,
            Aggregation,
            HashJoin,
            OrderBy,
            TableScan,
            NONE,
            LocalPartition,
            Unnest,
            RowNumber,
            NestedLoopJoin
        };

        virtual ~CostModel() = default;

        CostModel(CataLog catalog) : catalog(catalog) {}

        // node needs a stat (which isn't necessarilty a source stat) 
        virtual CostEstimate getCost(std::shared_ptr<const core::PlanNode>& node, std::vector<Source> sources) = 0;
        // if else on node type and handle each type .. e.g projections, filters, etc

        PlanNodeType hashPlanNode(std::string_view name) {
            if (name == "Filter") return Filter;
            if (name == "Project") return Project;
            if (name == "TableScan") return TableScan;
            if (name == "HashJoin") return HashJoin;
            if (name == "OrderBy") return OrderBy;
            if (name == "Aggregation") return Aggregation;
            if (name == "LocalPartition") return LocalPartition;
            if (name == "Unnest") return Unnest;
            if (name == "RowNumber") return RowNumber;
            if (name == "NestedLoopJoin") return NestedLoopJoin;
            return NONE;
        }

    protected:
        CataLog catalog;

};

class SimpleCostModel : public CostModel {

    public:
        SimpleCostModel(CataLog catalog) : CostModel(catalog) {}

        CostEstimate getCost(std::shared_ptr<const core::PlanNode>& node, std::vector<Source> sources) override {

            if(!node)
                return CostEstimate(0,0,0);
            LOG(INFO) << fmt::format("[INFO] CostModel - getCost, Node: {}, isSourceEmpty:{}", node->name(), sources.empty()) << std::endl;
            // it is a leaf node
            if(sources.empty()){
                return handleLeafNode(node);
            } else {
                return handleInternalNode(node, sources);
            }

        }
    
    private:

        // Leaf Node is usually a data source
        // Returned CostEstimate object will store the cardinality of the data source
        CostEstimate handleLeafNode(std::shared_ptr<const core::PlanNode>& node){
            
            // Retrieve the source object from catalog based on node's id in velox query plan
            std::shared_ptr<Source> src = catalog.getSource(node->id());
            LOG(INFO) << fmt::format("[INFO] LeafNode: CostModel get src from catalog based on node id: {}, srcType: {}", node->id(), src->getType()) << std::endl;
            if(!src)
                throw std::runtime_error("Source not found for node: " + node->id() + ":" + std::string(node->name()));

            switch (src->getType()) {
                case Source::Type::FILE:
                case Source::Type::DATABASE:
                case Source::Type::NODE:
                case Source::Type::VECTOR: {
                    
                    // for now we're just using output stat but
                    // we can use different stats here based on the src type
                    
                    std::shared_ptr<OutputStat> stat = std::static_pointer_cast<OutputStat>(src->getStats());

                    // the cost for source node is 0 or constant for a plan
                    return CostEstimate(0.0, stat->getRows(), stat->getCols());
                } break;
                
                default:    
                    throw std::runtime_error("Source type not supported");
              
                
            }
             
        }

        CostEstimate handleInternalNode(std::shared_ptr<const core::PlanNode>& node, std::vector<Source> sources){
            
            if(sources.empty())
                throw std::runtime_error("Source not found for node: " + node->id() + ":" + std::string(node->name()));
            LOG(INFO) << fmt::format("[INFO] InternalNode: {}", node->name()) << std::endl;
            std::vector<double> coefficientVector = UdfCostCoefficient::getInstance().getCoefficient(std::string(node->name())); 
            switch (hashPlanNode(node->name())) {
                case Aggregation:
                case OrderBy:{
                    float lambda = 0.2;
                    std::shared_ptr<OutputStat> stats = std::static_pointer_cast<OutputStat>(sources[0].getStats());
                    return CostEstimate(coefficientVector[0] * stats->getRows(), lambda * stats->getRows(), stats->getCols());
                } break;
                case Filter: {
                    float lambda = 0.5;
                    std::shared_ptr<OutputStat> stats =  std::static_pointer_cast<OutputStat>(sources[0].getStats());
                    return CostEstimate(lambda * stats->getRows() , lambda * stats->getRows(), stats->getCols());

                } break;
                case HashJoin: {
                    // Left table: MxN, Right table: OxP
                    // Cost: alpha * M + beta * M * N + gamma * O * P
                    float alpha = coefficientVector[0];
                    float beta = coefficientVector[1];
                    float gamma = coefficientVector[2];
                    std::shared_ptr<OutputStat> stats1 =  std::static_pointer_cast<OutputStat>(sources[0].getStats());
                    std::shared_ptr<OutputStat> stats2 =  std::static_pointer_cast<OutputStat>(sources[1].getStats());
                    float cost = alpha * stats1->getRows() + beta * stats1->getRows() * stats1->getCols() + gamma * stats2->getRows() * stats2->getCols();
                    return CostEstimate(cost, stats1->getRows(), stats1->getCols() + stats2->getCols());
                } break;
                case Project: {
                    
                    std::shared_ptr<const ProjectNode> projectNode = std::dynamic_pointer_cast<const ProjectNode> (node);
                    
	                const std::vector<TypedExprPtr> & projections = projectNode->projections();

                    LOG(INFO) << "[INFO] CostModel - handleInternalNode : There are " << projections.size() << " projections." << std::endl;
                    LOG(INFO) << "\t\t: ";
                    for (auto exprPtr: projections) {
                        LOG(INFO) << exprPtr->toString() << ",";
                    }

                    // if multiple projection we should add the cost here
                    CostEstimate projectionCostEstimate(0,0,0);
                    for(auto projection : projections){
                        CostEstimate partialCostEstimate = handleProjection(projection, sources);
                        projectionCostEstimate.cost += partialCostEstimate.cost ;
                        projectionCostEstimate.outputRows = (partialCostEstimate.outputRows > projectionCostEstimate.outputRows) ? partialCostEstimate.outputRows : projectionCostEstimate.outputRows;
                        projectionCostEstimate.outputCols = (partialCostEstimate.outputCols > projectionCostEstimate.outputCols) ? partialCostEstimate.outputCols : projectionCostEstimate.outputCols;
                    }


                    return projectionCostEstimate;
                } break;

                case LocalPartition: {
                    //FIXME
                    std::shared_ptr<OutputStat> stats1 =  std::static_pointer_cast<OutputStat>(sources[0].getStats());
                    // std::shared_ptr<OutputStat> stats2 =  std::static_pointer_cast<OutputStat>(sources[1].getStats());
                    return CostEstimate(coefficientVector[0] * stats1->getRows() , stats1->getRows() , stats1->getCols());
                } break;

                case Unnest: {
                    //FIXME CostModel - Init Sta
                    std::shared_ptr<OutputStat> stats1 =  std::static_pointer_cast<OutputStat>(sources[0].getStats());
                    // std::shared_ptr<OutputStat> stats2 =  std::static_pointer_cast<OutputStat>(sources[1].getStats());
                    return CostEstimate(1 * stats1->getRows(), 1 * stats1->getRows() , stats1->getCols());
                } break;

                case RowNumber: {
                    //FIXME CostModel - Init Sta
                    std::shared_ptr<OutputStat> stats1 =  std::static_pointer_cast<OutputStat>(sources[0].getStats());
                    return CostEstimate(coefficientVector[0] * stats1->getRows(), stats1->getRows(), stats1->getCols());
                } break;
                
                case NestedLoopJoin: {
                    // Left table: MxN, Right table: OxP
                    // Cost: alpha * M + beta * M * N + gamma * M * O * P
                    float alpha = coefficientVector[0];
                    float beta = coefficientVector[1];
                    float gamma = coefficientVector[2];
                    std::shared_ptr<OutputStat> stats1 =  std::static_pointer_cast<OutputStat>(sources[0].getStats());
                    std::shared_ptr<OutputStat> stats2 =  std::static_pointer_cast<OutputStat>(sources[1].getStats());
                    LOG(INFO) << fmt::format("[INFO] - NestedLoopJoin Stat1: r: {}, c: {}",stats1->getRows(), stats1->getCols());
                    LOG(INFO) << fmt::format("[INFO] - NestedLoopJoin Stat2: r: {}, c: {}",stats2->getRows(), stats2->getCols());
                    float cost = alpha * stats1->getRows() + beta * stats1->getRows() * stats1->getCols() + gamma * stats2->getRows() * stats2->getCols();
                    int maxRowNumber = (stats1->getRows() > stats2->getRows()) ? stats1->getRows() : stats2->getRows();
                    return CostEstimate(cost, maxRowNumber, stats2->getCols());
                    // return CostEstimate(cost, stats1->getRows() * stats2->getRows(), stats1->getCols() + stats2->getCols());
                } break;

                default:    
                    throw std::runtime_error(fmt::format("Node type not supported: {}", node->name()));
                
            }
             
        }


        CostEstimate getUDFCost(const std::vector<std::string> udfs, std::vector<Source> sources) {

            // get the stat for the source of this node
            std::shared_ptr<OutputStat> stat = std::static_pointer_cast<OutputStat>(sources[0].getStats());
            CostEstimate finalEstimate(0, stat->getRows(), stat->getCols());

            LOG(INFO) << fmt::format("[INFO] CostModel - Init Stat: Rows:{}, Cols:{}, Cost: {}", finalEstimate.outputRows, finalEstimate.outputCols, finalEstimate.cost) << std::endl;

            vectorFunctionFactories().withRLock([&](auto& functionMap) { 
            
                for(std::string udf : udfs){
                                 
                    auto it = functionMap.find(udf);

                    if (it != functionMap.end()) {
                    
                        LOG(INFO) << fmt::format("[INFO] getUDFCost, processing udf: {}", udf) << std::endl;
                        core::QueryConfig config({});
                        std::shared_ptr<VectorFunction> func;
                        // FIXME dynamically choose the input type to get the right function pointer
                        if (udf == "mat_mul_h") {
                            func = getVectorFunction(udf, {ARRAY(REAL()), ARRAY(REAL())}, {}, config);
                        } else if (udf == "velox_tree_predict" || udf == "velox_decision_tree_construct" || udf == "velox_decision_tree_predict" || udf  == "decision_forest_predict") {
                            // FIXME, bug here failed passing custom type
                            continue;
                            // func = getVectorFunction(udf, {ARRAY(REAL()), OpaqueType()}, {}, config);
                        } else if (udf == "gt" || udf == "if") {
                            continue;
                        } else {
                          func = getVectorFunction(udf, {ARRAY(REAL())}, {}, config);
                        }
                        
                        std::shared_ptr<MLFunction> mlFunc = std::dynamic_pointer_cast<MLFunction>(func);
                        CostEstimate curCost = mlFunc->getCost({finalEstimate.outputRows, finalEstimate.outputCols});
                        finalEstimate.cost += curCost.cost;
                        finalEstimate.outputRows = curCost.outputRows;
                        finalEstimate.outputCols = curCost.outputCols;
                        LOG(INFO) << fmt::format("[INFO] CostModel - getUDFCost: {}, cost: {}, accumulated cost: {}", mlFunc->getFuncName(), curCost.cost, finalEstimate.cost) << std:: endl;
                        LOG(INFO) << fmt::format("\t\t outputRows: {}, outputCols: {}", finalEstimate.outputRows, finalEstimate.outputCols) << std::endl;
                    }    
                }
            });
            LOG(INFO) << fmt::format("[INFO] CostModel :getUDFCost done, cost: {} ", finalEstimate.cost) << std::endl;
            return finalEstimate;
	    }
	 
	    
        CostEstimate handleProjection(const core::TypedExprPtr expression,  std::vector<Source> sources) {

            if (!expression) return CostEstimate(0,1,1);
            // entire expression for udf
            std::string exp = expression->toString();
            LOG(INFO) << "[INFO] CostModel - handleProjection: " << exp << std::endl;
            // relu(mat_add(mat_mul())
            // (mat_mul -> mat_add -> relu) ()
            std::vector<std::string> udfs = getUDFs(exp);

            // to estimate entire expression cost
            // we have to start with the innermost udf
            // because the input size applies to the
            // innermost udf
            // hence we have to reverse the exp
            // relu(mat_add(mat_mul())) -> mat_mul(mad_add(relu))
            // now we can lookup the udf cost map
            std::reverse(udfs.begin(), udfs.end());
            return getUDFCost(udfs, sources);
        }

        std::vector<std::string> getUDFs(std::string expression) {
            std::istringstream stream(expression);
            std::string token;
            std::vector<std::string> udfs;
            char delimiter = '(';
         
            while (std::getline(stream, token, delimiter)) {
                udfs.push_back(token);
            }
            // last element is the input
            // hence we have to delete it
            // relu(mat_add(mat_mul(x)))
            if(!udfs.empty())
                udfs.pop_back();

            return udfs;
        }


};