//#include "Infer.h"
//
//#include <fstream>
//#include <filesystem>
//
//namespace DC {
//	static std::unordered_map<std::string, Infer::ConstructEngine> s_engineRegistry;
//	static std::string s_defaultEngine;
//
//
//    std::unique_ptr<Infer> Infer::Create(const std::vector<std::byte>& modelData, size_t maxParallelCount, const std::string& engine){
//		std::string engineName = engine.empty() ? s_defaultEngine : engine;
//		auto it = s_engineRegistry.find(engineName);
//
//		if (it != s_engineRegistry.end()) {
//			Infer* base = it->second(modelData, maxParallelCount);
//			return std::unique_ptr<Infer>(base);
//		}
//
//        return std::unique_ptr<Infer>();
//    }
//
//    std::unique_ptr<Infer> Infer::Create(const std::filesystem::path& modelPath, size_t maxParallelCount, const std::string& engine){
//		std::string engineName = engine.empty() ? s_defaultEngine : engine;
//		auto it = s_engineRegistry.find(engineName);
//
//		if (it != s_engineRegistry.end()) {
//			std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
//			if (!file.is_open()) {
//				return nullptr;
//			}
//			auto fileSize = file.tellg();
//			file.seekg(0, std::ios::beg);
//			std::vector<std::byte> buffer(fileSize);
//			if (!file.read(reinterpret_cast<char*>(buffer.data()), fileSize)) {
//				return nullptr;
//			}
//			Infer* base = it->second(buffer, maxParallelCount);
//			return std::unique_ptr<Infer>(base);
//		}
//
//        return std::unique_ptr<Infer>();
//    }
//
//    void Infer::addEngine(const std::string& name, ConstructEngine factory) {
//        if (s_engineRegistry.empty()) {
//            s_defaultEngine = name;
//        }
//        s_engineRegistry[name] = factory;
//    }
//
//    std::unordered_set<std::string> Infer::getEngines() {
//        std::unordered_set<std::string> names;
//        for (const auto& pair : s_engineRegistry) {
//            names.insert(pair.first);
//        }
//        return names;
//    }
//}