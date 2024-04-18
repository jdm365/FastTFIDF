#include <stdint.h>
#include <vector>
#include <string>

#include "robin_hood.h"

struct CSRData {
	std::vector<float> values;
	std::vector<uint32_t> idxs;
	std::vector<uint32_t> ptrs;
};

struct _TFIDF {
	robin_hood::unordered_map<std::string, uint32_t> vocabulary;
	robin_hood::unordered_map<uint32_t, uint32_t> doc_frequency;
	int min_df;
	float max_df;

	_TFIDF(
			int min_df=1,
			float max_df=1
			) : min_df(min_df), max_df(max_df) {}

	std::vector<std::string> tokenize(const std::string& text);
	CSRData fit_transform(
		const std::vector<std::string>& documents
	);
	CSRData transform(
		const std::vector<std::string>& documents
	);
};
