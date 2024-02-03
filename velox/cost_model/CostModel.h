#include<vector>
#include "velox/cost_model/Source.h"
#include "velox/cost_model/Catalog.h"

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
                case OrderBy:{
                    
                    std::shared_ptr<OutputStat> stats = std::static_pointer_cast<OutputStat>(sources[0].getStats());
                    return CostEstimate(stats->getCols() + stats->getRows(), stats->getRows(), stats->getCols());
                } break;
                case Filter: {
                    float lambda = 0.5;
                    std::shared_ptr<OutputStat> stats =  std::static_pointer_cast<OutputStat>(sources[0].getStats());
                    return CostEstimate(lambda * stats->getRows() , lambda * stats->getRows(), stats->getCols());

                } break;
                case HashJoin: {
                    float lambda = 0.7;
                    std::shared_ptr<OutputStat> stats1 =  std::static_pointer_cast<OutputStat>(sources[0].getStats());
                    std::shared_ptr<OutputStat> stats2 =  std::static_pointer_cast<OutputStat>(sources[1].getStats());
                    return CostEstimate(lambda * stats1->getRows() * stats2->getRows(), lambda * stats1->getRows() * stats2->getRows(), stats1->getCols() + stats2->getCols());
                } break;
                case Project: {
                    
                    std::shared_ptr<const ProjectNode> projectNode = std::dynamic_pointer_cast<const ProjectNode> (node);
                    
	                const std::vector<TypedExprPtr> & projections = projectNode->projections();

                    std::cout << "There are " << projections.size() << " projections." << std::endl;

                    for(auto projection : projections){
                        handleProjection(projection, sources);
                    }


                    return CostEstimate(0,1,1);
                } break;

                default:    
                    throw std::runtime_error("Node type not supported");
                
            }
             
        }


        CostEstimate getUDFCost(const std::vector<std::string> udfs, std::vector<Source> sources) {

            // get the stat for the source of this node
            std::shared_ptr<OutputStat> stat = std::static_pointer_cast<OutputStat>(sources[0].getStats());
            CostEstimate finalEstimate(0, stat->getRows(), stat->getCols());

            vectorFunctionFactories().withRLock([&](auto& functionMap) { 
            
                for(std::string udf : udfs){
                                 
                    auto it = functionMap.find(udf);

                    if (it != functionMap.end()) {
                    
                        std::cout << udf << std::endl;
                        core::QueryConfig config({});
                        std::shared_ptr<VectorFunction> func = getVectorFunction(udf, {ARRAY(REAL())}, {}, config);
                        std::shared_ptr<MLFunction> mlFunc = std::dynamic_pointer_cast<MLFunction>(func);
                        CostEstimate curCost = mlFunc->getCost({finalEstimate.outputRows, finalEstimate.outputCols});
                     
                        finalEstimate.cost += curCost.cost;
                        std::cout << finalEstimate.cost << std:: endl;
                        finalEstimate.outputCols = curCost.outputRows;
                        finalEstimate.outputCols = curCost.outputCols;
                    }    
                }
            });
            std::cout<< finalEstimate.cost;
            return finalEstimate;
	    }
	 
	    
        void handleProjection(const core::TypedExprPtr expression,  std::vector<Source> sources) {

            if (!expression) return;
            // entire expression for udf
            std::string exp = expression->toString();
            std::cout << exp << std::endl;
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
            getUDFCost(udfs, sources);


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