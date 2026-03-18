#include "TensorSlot.h"

#include <iostream>
#include <string>

static void runTensorSlotTests() {
	using namespace DC;

	struct DummyExternalTensor {
		std::string payload;
	};

	// Test 1: write to input slot and inspect
	{
		TensorSlotBase::Config cfg = TensorSlotBase::CreateConfig();
		cfg.setPosition(TensorSlotBase::Config::Position::Input);
		cfg.setType(TensorSlotBase::Config::Type::Value);

		auto slot = CreateSlot<float>("in", {2,2}, cfg);
		if (!slot.isInput()) throw std::runtime_error("slot should be input");
		if (!slot.isType<float>()) throw std::runtime_error("slot type should be float");

		// prepare tensor
		Tensor t = Tensor::Create<float>({2,2});
		t.fill<float>(1.5f);

		slot << t; // write

		if (!slot.hasData()) throw std::runtime_error("slot should have data after write");

		const Tensor& view = slot.view();
		auto sp = view.data<float>();
		if (sp.size() != 4) throw std::runtime_error("unexpected data size");
		for (auto v : sp) {
			if (v != 1.5f) throw std::runtime_error("unexpected value in tensor");
		}
	}

	// Test 2: default data and read from output slot
	{
		TensorSlotBase::Config cfg = TensorSlotBase::CreateConfig();
		cfg.setPosition(TensorSlotBase::Config::Position::Output);

		auto slot = CreateSlot<float>("out", {1,2}, cfg);

		Tensor def = Tensor::Create<float>({1,2});
		def.fill<float>(2.5f);
		slot.setDefaultTensor(def);

		if (!slot.hasDefaultData()) throw std::runtime_error("slot should have default data");

		Tensor out;
		slot >> out; // read

		auto sp = out.data<float>();
		if (sp.size() != 2) throw std::runtime_error("unexpected default tensor size");
		for (auto v : sp) if (v != 2.5f) throw std::runtime_error("unexpected default tensor value");
	}

	// Test 3: shape mismatch should throw when loading invalid shape
	{
		TensorSlotBase::Config cfg = TensorSlotBase::CreateConfig();
		cfg.setPosition(TensorSlotBase::Config::Position::Input);

		auto slot = CreateSlot<float>("badshape", {2,2}, cfg);
		Tensor t = Tensor::Create<float>({1,2});
		t.fill<float>(0.0f);
		bool thrown = false;
		try {
			slot << t;
		}
		catch (const std::exception& e) {
			thrown = true;
		}
		if (!thrown) throw std::runtime_error("expected exception on shape mismatch");
	}

	// Test 4: TensorSlot should keep/own external tensors and provide zero-copy external view
	{
		TensorSlotBase::Config cfg = TensorSlotBase::CreateConfig();
		cfg.setPosition(TensorSlotBase::Config::Position::Input);
		auto toInternal = [](const DummyExternalTensor& t) {
			// placeholder conversion: produce a float tensor from external
			(void)t;
			Tensor x = Tensor::Create<float>({ 1 });
			x.fill<float>(0.0f);
			return x;
		};
		auto toExternal = [](const Tensor&) {
			return DummyExternalTensor{ "fromInternal" };
		};

		TensorSlot<DummyExternalTensor> slot(
			"ext",
			Type::getType<TensorMeta::TensorType, float>(),
			Type::getSize<TensorMeta::TensorType, float>(),
			{ 1 },
			toInternal,
			toExternal,
			cfg
		);

		// write external by rvalue: slot should take ownership
		DummyExternalTensor ext{ "moved" };
		slot << std::move(ext);

		// read back external view (zero-copy ownership should preserve payload)
		DummyExternalTensor got;
		slot >> got;
		if (got.payload != "moved") throw std::runtime_error("TensorSlot failed to preserve moved external payload");

		// Now test conversion path: write an internal tensor and read as external
		TensorSlot<DummyExternalTensor> slot2(
			"ext2",
			Type::getType<TensorMeta::TensorType, float>(),
			Type::getSize<TensorMeta::TensorType, float>(),
			{ 1 },
			toInternal,
			toExternal,
			cfg
		);

		// prepare internal tensor matching slot2 type and shape
		Tensor internal = Tensor::Create<float>({ 1 });
		internal.fill<float>(3.14f);
		slot2 << std::move(internal); // write internal data

		DummyExternalTensor outExt;
		slot2 >> outExt; // convert internal -> external
		if (outExt.payload != "fromInternal") throw std::runtime_error("TensorSlot conversion to external failed");

	}
}



int main() {
	try {
		runTensorSlotTests();

		std::cout << "TensorSlotBase tests passed" << std::endl;
		return 0;
	}
	catch(const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}
}
