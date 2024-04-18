#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <chrono>

#include "robin_hood.h"
#include "engine.h"

inline std::vector<std::string> _TFIDF::tokenize(const std::string& text) {
    std::istringstream iss(text);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token) {
        std::transform(token.begin(), token.end(), token.begin(), ::tolower);
        tokens.push_back(token);
    }
    return tokens;
}

CSRData _TFIDF::fit_transform(const std::vector<std::string>& documents) {
	CSRData csr_data;

	std::string token = "";

    int total_cols = 0;

	auto start = std::chrono::high_resolution_clock::now();
    // Build the vocabulary and document frequency
    for (const auto& doc : documents) {
		robin_hood::unordered_flat_set<std::string> seen_terms;

		for (const char& c : doc) {
			if (c == ' ') {
				// if (vocabulary.find(token) == vocabulary.end()) {
					// vocabulary[token] = total_cols++;
				// }
				auto [it, inserted] = vocabulary.try_emplace(token, total_cols);
				total_cols += (int)inserted;
				// ++doc_frequency[vocabulary[token]];
				if (seen_terms.insert(token).second) {
					++doc_frequency[it->second];
				}
				token.clear();
				continue;
			}
			token += toupper(c);
        }
		token.clear();
    }
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "Time taken to build vocabulary and document frequency: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    csr_data.ptrs.push_back(0);
    int index = 0;
    float norm_factor;

	start = std::chrono::high_resolution_clock::now();
    for (const auto& doc : documents) {
		robin_hood::unordered_flat_map<uint32_t, float> word_count;
        norm_factor = 0.0;

		std::string token = "";
		for (const char& c : doc) {
			if (c == ' ') {
				if (vocabulary.find(token) != vocabulary.end()) {
					uint32_t col_index = vocabulary[token];
					++word_count[col_index];
				}
				token.clear();
				continue;
			}
			token += toupper(c);
		}

		for (auto& wc : word_count) {
            wc.second *= log((documents.size() + 1.0) / (doc_frequency[wc.first] + 1.0));
            norm_factor += wc.second * wc.second;
        }
        norm_factor = sqrt(norm_factor);

        for (const auto& wc : word_count) {
            csr_data.values.push_back(wc.second / norm_factor);
            csr_data.idxs.push_back(wc.first);
        }

        index += word_count.size();
        csr_data.ptrs.push_back(index);
    }
	end = std::chrono::high_resolution_clock::now();
	std::cout << "Time taken to transform documents: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

	return csr_data;
}

CSRData _TFIDF::transform(const std::vector<std::string>& documents) {
	CSRData csr_data;

	csr_data.ptrs.push_back(0);
	int index = 0;
	float norm_factor;

	for (const auto& doc : documents) {
		robin_hood::unordered_flat_map<uint32_t, float> word_count;
		std::vector<std::string> tokens = tokenize(doc);
		norm_factor = 0.0;

		for (const auto& token : tokens) {
			if (vocabulary.find(token) != vocabulary.end()) {
				uint32_t col_index = vocabulary[token];
				word_count[col_index]++;
			}
		}

		for (auto& wc : word_count) {
			wc.second *= log((documents.size() + 1.0) / (doc_frequency[wc.first] + 1.0));
			norm_factor += wc.second * wc.second;
		}
		norm_factor = sqrt(norm_factor);

		for (const auto& wc : word_count) {
			csr_data.values.push_back(wc.second / norm_factor);
			csr_data.idxs.push_back(wc.first);
		}
		index += word_count.size();
		csr_data.ptrs.push_back(index);
	}

	return csr_data;
}
