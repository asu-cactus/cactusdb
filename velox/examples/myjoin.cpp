

auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();
core::PlanNodeId nationScanId;
core::PlanNodeId regionScanId;
plan = PlanBuilder(planNodeIdGenerator)
           .tableScan(
               tpch::Table::TBL_NATION, {"n_regionkey"}, 1 /*scaleFactor*/)
           .capturePlanNodeId(nationScanId)
           .hashJoin(
               {"n_regionkey"},
               {"r_regionkey"},
               PlanBuilder(planNodeIdGenerator)
                   .tableScan(
                       tpch::Table::TBL_REGION,
                       {"r_regionkey", "r_name"},
                       1 /*scaleFactor*/)
                   .capturePlanNodeId(regionScanId)
                   .planNode(),
               "", // extra filter
               {"r_name"})
           .singleAggregation({"r_name"}, {"count(1) as nation_cnt"})
           .orderBy({"r_name"}, false)
           .planNode();

auto nationCnt = AssertQueryBuilder(plan)
                     .split(nationScanId, makeTpchSplit())
                     .split(regionScanId, makeTpchSplit())
                     .copyResults(pool());

std::cout << std::endl
          << "> number of nations per region in TPC-H: "
          << nationCnt->toString() << std::endl;
std::cout << nationCnt->toString(0, 10) << std::endl;