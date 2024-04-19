#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <thread>

#include "robin_hood.h"
#include "engine.h"


CSRData _TFIDF::fit_transform(const std::vector<std::string>& documents) {
	std::string token = "";

    int total_cols = 0;

	uint32_t max_occurrences = (uint32_t)max_df;
	if (max_df <= 1.0f) {
		max_occurrences = (uint32_t)(max_df * documents.size());
	}

	vocabulary.reserve(documents.size() / 2);
	doc_frequency.reserve(documents.size() / 2);

	auto start = std::chrono::high_resolution_clock::now();
    // Build the vocabulary and document frequency
    for (const auto& doc : documents) {
		robin_hood::unordered_flat_set<uint32_t> seen_terms;

		for (const char& c : doc) {
			if (c != ' ') {
				token += toupper(c);
				continue;
			}

			if (token.empty()) {
				continue;
			}

			auto [it, inserted] = vocabulary.try_emplace(token, total_cols);
			if (inserted) {
				doc_frequency.push_back(1);
				++total_cols;
				seen_terms.insert(it->second);
				token.clear();
				continue;
			}
			doc_frequency[it->second] += seen_terms.insert(it->second).second;
			token.clear();
        }
		token.clear();
    }

	for (auto it = doc_frequency.begin(); it != doc_frequency.end();) {
		if (*it < (uint32_t)min_df || *it > max_occurrences) {
			it = doc_frequency.erase(it);
		} else {
			++it;
		}
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "Time taken to build vocabulary and document frequency: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

	start = std::chrono::high_resolution_clock::now();
	// CSRData csr_data = transform(documents, true);
	CSRData csr_data = transform_map_reduce(documents);
	end = std::chrono::high_resolution_clock::now();

	std::cout << "Time taken to transform documents: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

	return csr_data;
}

CSRData _TFIDF::transform(const std::vector<std::string>& documents, bool is_first) {
	CSRData csr_data;

	if (is_first) csr_data.ptrs.push_back(0);
    int index = 0;
    float norm_factor;

    for (const auto& doc : documents) {
		robin_hood::unordered_flat_map<uint32_t, float> word_count;
        norm_factor = 0.0f;

		std::string token = "";
		for (const char& c : doc) {
			if (c != ' ') {
				token += toupper(c);
				continue;
			}
			if (token.empty()) {
				continue;
			}

			auto it = vocabulary.find(token);
			if (it != vocabulary.end()) {
				++word_count[it->second];
			}
			token.clear();
		}

		for (auto& wc : word_count) {
            wc.second *= log((documents.size() + 1.0f) / (doc_frequency[wc.first] + 1.0f));
            norm_factor += wc.second * wc.second;
        }
        norm_factor = 1.0f / sqrt(norm_factor);

        for (const auto& wc : word_count) {
            csr_data.values.push_back(wc.second * norm_factor);
            csr_data.idxs.push_back(wc.first);
        }

        index += word_count.size();
        csr_data.ptrs.push_back(index);
    }

	return csr_data;
}


CSRData _TFIDF::transform_map_reduce(const std::vector<std::string>& documents) {
	// Do transform accross n threads and then collect the results together
	uint32_t num_threads = std::thread::hardware_concurrency();
	std::cout << "Using " << num_threads << " threads" << std::endl;

	uint32_t num_docs = documents.size();
	uint32_t docs_per_thread = num_docs / num_threads;

	std::vector<std::thread> threads;
	std::vector<CSRData> thread_results(num_threads);

	for (uint32_t i = 0; i < num_threads; ++i) {
		uint32_t start = i * docs_per_thread;
		uint32_t end = (i + 1) * docs_per_thread;
		if (i == num_threads - 1) {
			end = num_docs;
		}
		threads.emplace_back(std::thread([this, &documents, &thread_results, i, start, end]() {
			std::vector<std::string> docs(documents.begin() + start, documents.begin() + end);
			if (i == 0) {
				thread_results[i] = transform(docs, true);
			}
			else {
				thread_results[i] = transform(docs);
			}
		}));
	}

	for (auto& t : threads) {
		t.join();
	}

	CSRData csr_data;
	for (const auto& tr : thread_results) {
		csr_data.ptrs.insert(csr_data.ptrs.end(), tr.ptrs.begin(), tr.ptrs.end());
		csr_data.idxs.insert(csr_data.idxs.end(), tr.idxs.begin(), tr.idxs.end());
		csr_data.values.insert(csr_data.values.end(), tr.values.begin(), tr.values.end());
	}

	return csr_data;
}
