#include "TensorSlot.h"

#include <iostream>

static void runTensorSlotTests() {
	using namespace DC;
	// 1) Create a TensorSlot with specific rules and verify properties
	TensorSlot slot("input", TensorMeta::TensorType::Float, sizeof(float), {2, 3});
	if (slot.name() != "input") throw std::runtime_error("TensorSlot name mismatch");
	if (slot.type() != TensorMeta::TensorType::Float) throw std::runtime_error("TensorSlot type mismatch");
	if (slot.typeSize() != sizeof(float)) throw std::runtime_error("TensorSlot type size mismatch");

	// 2) Create a matching Tensor and verify slot acceptance
	Tensor t(TensorMeta::TensorType::Float, sizeof(float), {2, 3}, {});
	try {
		slot << t; // should has throw
		throw std::runtime_error("TensorSlot accepted matching tensor");
	}
	catch (const TensorException&) {
		// expected
	}
	t.fill(42.0f);
	try {
		slot << t; // should not throw
	}
	catch (const TensorException&) {
		throw std::runtime_error("TensorSlot rejected matching tensor");
	}
	
	// 3) Create a non-matching Tensor (wrong shape) and verify rejection
	try {
		Tensor tWrongShape(TensorMeta::TensorType::Float, sizeof(float), {3, 2}, {});
		slot << tWrongShape;
		throw std::runtime_error("TensorSlot accepted wrong shape");
	} catch (const TensorException&) {
		// expected
	}

	// 4) Create a non-matching Tensor (wrong type) and verify rejection
	try {
		Tensor tWrongType(TensorMeta::TensorType::Int, sizeof(int), {2, 3}, {});
		slot << std::move(tWrongType);
		throw std::runtime_error("TensorSlot accepted wrong type");
	} catch (const TensorException&) {
		// expected
	}

	// 5) Set default tensor and verify retrieval
	{
		TensorSlot defaultSlot("default", TensorMeta::TensorType::Int, sizeof(int), { 2,3 });
		Tensor defaultT(TensorMeta::TensorType::Int, sizeof(int), { 2, 3 }, {});
		defaultT.fill(123);
		defaultSlot.setDefaultTensor(defaultT);
		auto retrieved = defaultSlot.view();
		if (
			retrieved.type() != TensorMeta::TensorType::Int ||
			retrieved.shape() != std::vector<int64_t>{2, 3} ||
			retrieved.data<int>()[0] != 123
			) {
			throw std::runtime_error("TensorSlot default tensor mismatch");
		}
	}

	// 6) getTensor on an empty slot should throw
	try {
		TensorSlot empty("empty", TensorMeta::TensorType::Float, sizeof(float), {1});
		empty.view();
		throw std::runtime_error("getTensor did not throw on empty slot");
	}
	catch (const TensorException&) {
		// expected
	}

	// 7) setDefaultTensor with mismatched type should throw (implementation throws std::runtime_error)
	try {
		TensorSlot defWrong("defWrong", TensorMeta::TensorType::Int, sizeof(int), {2,3});
		Tensor wrongDefault(TensorMeta::TensorType::Float, sizeof(float), {2,3}, {});
		defWrong.setDefaultTensor(wrongDefault);
		throw std::runtime_error("setDefaultTensor accepted mismatched default tensor");
	}
	catch (const std::runtime_error&) {
		// expected
	}

	// 8) clearData() should remove input data and getTensor() should return default again
	{
		TensorSlot mix("mix", TensorMeta::TensorType::Int, sizeof(int), {2,3});
		Tensor def(TensorMeta::TensorType::Int, sizeof(int), {2,3}, {});
		def.fill(7);
		mix.setDefaultTensor(def);
		Tensor t2(TensorMeta::TensorType::Int, sizeof(int), {2,3}, {});
		t2.fill(9);
		try {
			mix << (std::move(t2));
		}
		catch (const TensorException&) {
			throw std::runtime_error("mix.input rejected valid tensor");
		}
		mix.clearData();
		auto r = mix.view();
		if (r.data<int>()[0] != 7) {
			throw std::runtime_error("clear did not restore default tensor");
		}
	}

	// 9) Test move semantics of input
	{
		TensorSlot moveSlot("move", TensorMeta::TensorType::Int, sizeof(int), { 2,3 });
		Tensor t3(TensorMeta::TensorType::Int, sizeof(int), { 2,3 }, {});
		t3.fill(5);
		try {
			moveSlot << std::move(t3);
		}
		catch (const TensorException&) {
			throw std::runtime_error("moveSlot.input rejected valid tensor");
		}
		auto t4 = Tensor::Create<int>();
		moveSlot >> t4;
		if (t4.data<int>()[0] != 5) {
			throw std::runtime_error("moveSlot did not store moved tensor correctly");
		}
	}

	
}



int main() {
	try {
		runTensorSlotTests();

		std::cout << "TensorSlot tests passed" << std::endl;
		return 0;
	}
	catch(const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}
}
