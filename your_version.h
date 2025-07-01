commit d319c4caa6506209aed3916a2c484fd6fe3dd391
Author: jrt1899 <jaytandel1899@gmail.com>
Date:   Mon Jun 30 17:44:03 2025 +0000

    Filters added to TPCXAi queries

diff --git a/velox/optimizer/tests/BenchmarkQueryTemplates.h b/velox/optimizer/tests/BenchmarkQueryTemplates.h
index ad1a0193f..671d3f5dd 100644
--- a/velox/optimizer/tests/BenchmarkQueryTemplates.h
+++ b/velox/optimizer/tests/BenchmarkQueryTemplates.h
@@ -399,17 +399,14 @@ PlanBuilder setupProfileQueryPlanFromTemplate1(
                    "u_age",
                    "u_gender",
                    "u_occupation",
-                   "u_zipcode",
                    "m_movie_id",
-                   "m_genres",
-                    "m_title",
-                    "m_spoken_languages","m_popularity","m_vote_average","m_vote_count"   }
+                   "m_genres"  }
         );
       
         //Filter here
       if (generateFilter) {
         std::vector<std::string> filterExpr =
-            sampleUserMovieFilterExpr("movie_user", timestampSeed);
+            sampleUserMovieFilterExpr("age_gender_occupation_genre", timestampSeed);
         for (auto expr : filterExpr) {
           queryPlan = queryPlan.filter(expr);
         }
@@ -482,8 +479,7 @@ PlanBuilder setupProfileQueryPlanFromTemplate1(
                     {
                     "m_movie_id",
                    "m_genres",
-                    "m_title",
-                    "m_spoken_languages","m_popularity","m_vote_average","m_vote_count",
+                    "m_popularity","m_vote_average","m_vote_count",
                     "r_rating" 
                     },
                     /*joinType=*/core::JoinType::kInner  
@@ -492,7 +488,7 @@ PlanBuilder setupProfileQueryPlanFromTemplate1(
                 //filter expressions
                 if (generateFilter) {
             std::vector<std::string> filterExpr =
-                sampleUserMovieFilterExpr("movie", timestampSeed);
+                sampleUserMovieFilterExpr("genre_rating", timestampSeed);
             for (auto expr : filterExpr) {
             queryPlan = queryPlan.filter(expr);
             }
@@ -800,8 +796,15 @@ PlanBuilder setupProfileQueryPlanFromTemplate1(
               .project({
                 "id",
                 "extract_tf_features(hf_tokenizer(text)) as feature"
-            }).
-            project({fmt::format(modelStr, "feature")});
+            });
+
+        if (generateFilter) {
+            std::vector<std::string> filterExpr = sampleTPCxAIFilterExpr("idReview", timestampSeed);
+            for (auto expr : filterExpr) {
+                queryPlan = queryPlan.filter(expr);
+            }
+        }
+        queryPlan = queryPlan.project({fmt::format(modelStr, "feature")});
 
         //review Data read 
         cataLog.setIdAddressMap(
@@ -1001,7 +1004,7 @@ PlanBuilder setupProfileQueryPlanFromTemplate1(
             .project({"cast(c_customer_sk as BIGINT) as c_customer_sk",
                 "c_birth_day",
                 "c_birth_month",
-                "c_birth_year"})
+                "c_birth_year","c_birth_country"})
             .planNode(),
      "",
       {
@@ -1010,7 +1013,8 @@ PlanBuilder setupProfileQueryPlanFromTemplate1(
         "c_customer_sk",
         "c_birth_day",
         "c_birth_month",
-        "c_birth_year"
+        "c_birth_year",
+        "c_birth_country"
       },
       JoinType::kInner)
     .hashJoin(
@@ -1028,10 +1032,19 @@ PlanBuilder setupProfileQueryPlanFromTemplate1(
         "c_birth_day",
         "c_birth_month",
         "c_birth_year",
+        "c_birth_country",
         "department"
       },
-      JoinType::kInner)
-    .project({
+      JoinType::kInner);
+    
+      if (generateFilter) {
+            std::vector<std::string> filterExpr = sampleTPCxAIFilterExpr("department_birthDay_birthCountry", timestampSeed);
+            for (auto expr : filterExpr) {
+                queryPlan = queryPlan.filter(expr);
+            }
+        }
+    
+    queryPlan = queryPlan.project({
       "department_encoder(department) department_",
       "(1922.0 -   cast(c_birth_year as double))/(79.0) AS birth_year",
       "(12.0   -   cast(c_birth_month as double))/(11.0) AS birth_month",
