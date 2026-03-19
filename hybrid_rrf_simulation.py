"""
ANNSIM 2026 Camera-Ready Code Repository
Paper: OPTIMIZING BUILDING CODE COMPLIANCE CHECKING THROUGH SIMULATION AND LLMS
Description: This script simulates the Hybrid Retrieval Optimization Framework (RRF) 
             balancing Dense (Semantic) and Sparse (Lexical) retrieval streams.
"""

import numpy as np
from rank_bm25 import BM25Okapi

class HybridRetrievalSimulator:
    def __init__(self, k_constant=60):
        self.k = k_constant
        self.bm25_k1 = 1.5
        self.bm25_b = 0.75

    def reciprocal_rank_fusion(self, dense_rank_dict, sparse_rank_dict, alpha):
        fusion_scores = {}
        all_docs = set(dense_rank_dict.keys()).union(set(sparse_rank_dict.keys()))

        for doc_id in all_docs:
            r_dense = dense_rank_dict.get(doc_id, float('inf'))
            r_sparse = sparse_rank_dict.get(doc_id, float('inf'))

            semantic_score = 1.0 / (self.k + r_dense) if r_dense != float('inf') else 0
            lexical_score = 1.0 / (self.k + r_sparse) if r_sparse != float('inf') else 0

            fusion_scores[doc_id] = (alpha * semantic_score) + ((1 - alpha) * lexical_score)

        return sorted(fusion_scores.items(), key=lambda item: item[1], reverse=True)

    def run_alpha_sweep(self, dense_results, sparse_results):
        print("Starting Simulation Loop: Parameter Tuning (Alpha 0.0 to 1.0)")
        results = {}
        for alpha in np.arange(0.0, 1.1, 0.1):
            alpha = round(alpha, 1)
            fused_ranking = self.reciprocal_rank_fusion(dense_results, sparse_results, alpha)
            results[alpha] = fused_ranking
        return results

if __name__ == "__main__":
    simulator = HybridRetrievalSimulator(k_constant=60)
    mock_dense_ranks = {"Clause_C2.3": 8, "Clause_C2.4": 1, "Clause_C2.5": 2} 
    mock_sparse_ranks = {"Clause_C2.3": 1, "Clause_C2.5": 12, "Clause_C2.4": 15}
    optimization_curve_data = simulator.run_alpha_sweep(mock_dense_ranks, mock_sparse_ranks)
    print("Golden Ratio Validation: At alpha=0.4, correct clause score:", optimization_curve_data[0.4][0])
