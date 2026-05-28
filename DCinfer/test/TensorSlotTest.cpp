#include "TensorSlot.h"
#include "SlotType.h"

#include <iostream>
#include <string>

// 向全局注册表注册测试类型的映射（由 EngineRegistry.cpp 的静态初始化保障 DC::Tensor 和 NativeTensor）
// DummyExternalTensor 在此通过 ValiatorRegistry（不做校验）自动放行

static void runTensorSlotTests() {
	using namespace DC;

	struct DummyExternalTensor {
		std::string payload;
	};

	// Test 1: store Tensor and peek
	{
		TensorSlot::Config cfg = TensorSlot::CreateConfig();
		cfg.setPosition(TensorSlot::Config::Position::Input);

		TensorSlot slot("in", TensorMeta::TensorType::Float, sizeof(float), {2, 2}, cfg);

		if (!slot.isInput())
			throw std::runtime_error("slot should be input");
		if (!slot.isType<float>())
			throw std::runtime_error("slot type should be float");

		// prepare tensor
		Tensor t = Tensor::Create<float>({2, 2});
		t.fill<float>(1.5f);

		slot.store(std::move(t)); // store via type-erased API

		if (!slot.hasData())
			throw std::runtime_error("slot should have data after store");

		// peek for read-only access
		auto* viewPtr = slot.peek<Tensor>();
		if (!viewPtr)
			throw std::runtime_error("peek<Tensor> returned null");

		auto sp = viewPtr->data<float>();
		if (sp.size() != 4)
			throw std::runtime_error("unexpected data size");
		for (auto v : sp) {
			if (v != 1.5f)
				throw std::runtime_error("unexpected value in tensor");
		}

		// view() backward compat
		const auto& v = slot.view();
		if (std::abs(v.item<float>() - 1.5f) < 1e-6f) { /* ok */
		}
	}

	// Test 2: default data and take output
	{
		TensorSlot::Config cfg = TensorSlot::CreateConfig();
		cfg.setPosition(TensorSlot::Config::Position::Output);

		TensorSlot slot("out", TensorMeta::TensorType::Float, sizeof(float), {1, 2}, cfg);

		Tensor def = Tensor::Create<float>({1, 2});
		def.fill<float>(2.5f);
		slot.setDefaultTensor(def);

		if (!slot.hasDefaultData())
			throw std::runtime_error("slot should have default data");

		// Take via view (backward compat for default data)
		const auto& out = slot.view();
		auto sp = out.data<float>();
		if (sp.size() != 2)
			throw std::runtime_error("unexpected default tensor size");
		for (auto v : sp)
			if (v != 2.5f)
				throw std::runtime_error("unexpected default tensor value");
	}

	// Test 3: shape mismatch should throw when storing
	{
		TensorSlot::Config cfg = TensorSlot::CreateConfig();
		cfg.setPosition(TensorSlot::Config::Position::Input);

		TensorSlot slot("badshape", TensorMeta::TensorType::Float, sizeof(float), {2, 2}, cfg);
		Tensor t = Tensor::Create<float>({1, 2});
		t.fill<float>(0.0f);
		bool thrown = false;
		try {
			slot.store(std::move(t));
		} catch (const std::exception& e) {
			thrown = true;
		}
		if (!thrown)
			throw std::runtime_error("expected exception on shape mismatch");
	}

	// Test 4: store and take arbitrary external type (DummyExternalTensor)
	{
		TensorSlot::Config cfg = TensorSlot::CreateConfig();
		cfg.setPosition(TensorSlot::Config::Position::Input);

		TensorSlot slot("ext", TensorMeta::TensorType::Float, sizeof(float), {1}, cfg);

		// store external type directly (no validator registered → pass-through)
		DummyExternalTensor ext{"moved"};
		slot.store(std::move(ext));

		if (!slot.hasData())
			throw std::runtime_error("slot should have external data");
		if (slot.storedType() == SlotDataType::DCTensor)
			throw std::runtime_error("stored type should not be DCTensor");

		// take back
		auto got = slot.take<DummyExternalTensor>();
		if (got.payload != "moved")
			throw std::runtime_error("take<DummyExternalTensor> payload mismatch");

		// slot should be empty after take
		if (slot.hasData())
			throw std::runtime_error("slot should be empty after take");
	}

	// Test 5: type mismatch on take should throw
	{
		TensorSlot::Config cfg = TensorSlot::CreateConfig();
		cfg.setPosition(TensorSlot::Config::Position::Output);

		TensorSlot slot("typemismatch", TensorMeta::TensorType::Float, sizeof(float), {}, cfg);

		Tensor t(TensorMeta::TensorType::Float, sizeof(float));
		t = 42.0f;
		slot.store(std::move(t));

		if (slot.storedType() != SlotDataType::DCTensor)
			throw std::runtime_error("expected DCTensor type");

		// Try to take as wrong type
		bool thrown = false;
		try {
			slot.take<DummyExternalTensor>();
		} catch (const std::exception&) {
			thrown = true;
		}
		if (!thrown)
			throw std::runtime_error("expected exception on type mismatch take");
	}
}

int main() {
	try {
		runTensorSlotTests();

		std::cout << "TensorSlotBase tests passed" << std::endl;
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}
}
