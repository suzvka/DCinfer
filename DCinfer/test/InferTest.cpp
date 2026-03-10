#include "Infer.h"

#include <cassert>
#include <stdexcept>

class TestEnvA {
public:
	TestEnvA(int v) : value(v) {}
	int value;
};

class TestInferA {
public:
	class Value {
		public:
		Value(int v, const std::string& n) : value(v), name(n) {}
		int value;
		std::string name;
		std::vector<int64_t> shape = { 1 };
	};

	TestInferA(const TestEnvA& env) {
		_env = env.value;
	}

	std::vector<Value> run(std::vector<Value>&& input){
		std::vector<Value> output;
		int sum = 0;
		for (auto& v : input) {
			sum += v.value;
		}
		output.emplace_back(sum, "output");
		return output;
	}

	int _env;
	std::vector<std::string> inputNames = { "input_1", "input_2", "input_3" };
	std::vector<std::string> outputNames = { "output" };
	std::vector<int64_t> shape = { 1 };
};

class TestToolsA {
public:
	static bool loadModel(const std::vector<std::byte>& modelData, TestInferA& engine) {
		return true;
	}

	static std::vector<TestInferA::Value> inferFunc(std::vector<TestInferA::Value>& input, TestInferA& engine) {
		return engine.run(std::move(input));
	}

	static TestInferA::Value toInternal(const TestInferA::Value& value) {
		return value;
	}

	static TestInferA::Value toExternal(const TestInferA::Value& value) {
		return value;
	}

	static std::string getName(const TestInferA::Value& value) {
		return value.name;
	}

	static DC::Tensor::TensorType getType(const TestInferA::Value& value) {
		return DC::Tensor::TensorType::Int;
	}

	static std::vector<int64_t> getShape(const TestInferA::Value& value) {
		return value.shape;
	}

	static std::vector<std::string> getInputNames(const TestInferA& engine) {
		return engine.inputNames;
	}

	static std::vector<std::string> getOutputNames(const TestInferA& engine) {
		return engine.outputNames;
	}
};

int main() {
	using namespace DC;

	TestEnvA testEnv_A(10);

	TestInferA testInfer_A(testEnv_A);

	InferTools<TestInferA::Value, TestInferA> inferTools;
	inferTools.getInputNames = TestToolsA::getInputNames;
	inferTools.getOutputNames = TestToolsA::getOutputNames;
	inferTools.getName = TestToolsA::getName;
	inferTools.getType = TestToolsA::getType;
	inferTools.getShape = TestToolsA::getShape;


	std::vector<TestInferA::Value> inputs = {
		{ 1, "input_1" },
		{ 2, "input_2" },
		{ 3, "input_3" }
	};

	auto infer = CreateInfer("TestTypeA", "TestInferA", inferTools);
	auto& inferA = dynamic_cast<Infer<TestInferA::Value, TestInferA>&>(*infer);

	inferA.run(inputs);

	return 0;
}