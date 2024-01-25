#include<vector>
#include "velox/cost_model/Source.h"
#include "velox/cost_model/Catalog.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;


struct CostEstimate {
    float cost;
    int outputRows;
    int outputCols;

    CostEstimate(float cost, int rows, int cols) : cost(cost) , outputRows(rows), outputCols(cols) {}
};


class CostModel {

    public:

        enum PlanNodeType {
            Filter,
            Project,
            Aggregation,
            HashJoin,
            OrderBy,
            TableScan,
            NONE
        };

        virtual ~CostModel() = default;

        CostModel(std::shared_ptr<Catalog> catalog) : catalog(catalog) {}

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
            return NONE;
        }

    protected:
        std::shared_ptr<Catalog> catalog;

};

class SimpleCostModel : public CostModel {

    public:
        SimpleCostModel(std::shared_ptr<Catalog> catalog) : CostModel(catalog) {}

        CostEstimate getCost(std::shared_ptr<const core::PlanNode>& node, std::vector<Source> sources) override {

            if(!node)
                return CostEstimate(0,0,0);

            // it is a leaf node
            if(sources.empty()){
                return handleLeafNode(node);
            } else {
                return handleInternalNode(node, sources);
            }

            return CostEstimate(1.0, 1, 1);



            // std::cout << "Sources for " << node->name() << " - " << node->id() << std::endl;
            // for(Source s : sources){
            //     std::cout << s.getName() << " ";
            // }
            // std::cout << std::endl;
            // return CostEstimate(1.0, 1, 1);
        }
    
    private:
    
        CostEstimate handleLeafNode(std::shared_ptr<const core::PlanNode>& node){
            
            std::shared_ptr<Source> src = catalog->getSource(node->id());
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

            switch (hashPlanNode(node->name())) {
                case Aggregation:
                case Filter:
                case OrderBy:{
                    
                    std::shared_ptr<OutputStat> stats = std::static_pointer_cast<OutputStat>(sources[0].getStats());
                    return CostEstimate(stats->getCols() + stats->getRows(), stats->getRows(), stats->getCols());
                } break;
                case HashJoin: {

                    std::shared_ptr<OutputStat> stats1 =  std::static_pointer_cast<OutputStat>(sources[0].getStats());
                    std::shared_ptr<OutputStat> stats2 =  std::static_pointer_cast<OutputStat>(sources[1].getStats());

                    return CostEstimate(stats1->getRows() * stats2->getRows(), stats1->getRows() * stats2->getRows()  , stats1->getCols() + stats2->getCols());
                } break;
                case Project: {
                    
                    return CostEstimate(0,1,1);
                } break;
                
                default:    
                    throw std::runtime_error("Node type not supported");
                
            }
             
        }

};